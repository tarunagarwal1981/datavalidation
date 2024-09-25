import pandas as pd
import numpy as np
from database import get_db_engine
from sqlalchemy.exc import SQLAlchemyError
import streamlit as st

# Configuration
COLUMN_NAMES = {
    'VESSEL_NAME': 'VESSEL_NAME',
    'REPORT_DATE': 'REPORT_DATE',
    'LATITUDE': 'LATITUDE',
    'LONGITUDE': 'LONGITUDE',
    'OBSERVED_DISTANCE': 'OBSERVERD_DISTANCE',
    'STEAMING_TIME_HRS': 'STEAMING_TIME_HRS'
}

VALIDATION_THRESHOLDS = {
    'max_distance': 500,
    'distance_alignment_lower': 0.9,
    'distance_alignment_upper': 1.1
}

@st.cache_data
def fetch_validation_data():
    engine = get_db_engine()
    try:
        query = f"""
        SELECT {', '.join(COLUMN_NAMES.values())}
        FROM sf_consumption_logs
        ORDER BY {COLUMN_NAMES['VESSEL_NAME']}, {COLUMN_NAMES['REPORT_DATE']}
        """
        return pd.read_sql_query(query, engine)
    except SQLAlchemyError as e:
        st.error(f"Error fetching validation data: {str(e)}")
        return pd.DataFrame()

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth's radius in kilometers

    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    return R * c

def validate_distance_batch(df):
    failure_reasons = []
    
    # Calculate distances
    df['calc_distance'] = haversine_distance(
        df[COLUMN_NAMES['LATITUDE']].shift(), 
        df[COLUMN_NAMES['LONGITUDE']].shift(),
        df[COLUMN_NAMES['LATITUDE']], 
        df[COLUMN_NAMES['LONGITUDE']]
    )
    
    # Validations
    mask_negative = df[COLUMN_NAMES['OBSERVED_DISTANCE']] < 0
    mask_too_high = df[COLUMN_NAMES['OBSERVED_DISTANCE']] > VALIDATION_THRESHOLDS['max_distance']
    mask_zero_when_steaming = (df[COLUMN_NAMES['OBSERVED_DISTANCE']] == 0) & (df[COLUMN_NAMES['STEAMING_TIME_HRS']] > 0)
    mask_not_aligned = ~((VALIDATION_THRESHOLDS['distance_alignment_lower'] * df['calc_distance'] <= 
                          df[COLUMN_NAMES['OBSERVED_DISTANCE']]) & 
                         (df[COLUMN_NAMES['OBSERVED_DISTANCE']] <= 
                          VALIDATION_THRESHOLDS['distance_alignment_upper'] * df['calc_distance']))
    
    failure_reasons.extend([
        {'Vessel Name': row[COLUMN_NAMES['VESSEL_NAME']], 
         'Report Date': row[COLUMN_NAMES['REPORT_DATE']], 
         'Remarks': "Observed Distance is negative"}
        for _, row in df[mask_negative].iterrows()
    ])
    
    failure_reasons.extend([
        {'Vessel Name': row[COLUMN_NAMES['VESSEL_NAME']], 
         'Report Date': row[COLUMN_NAMES['REPORT_DATE']], 
         'Remarks': "Observed Distance too high"}
        for _, row in df[mask_too_high].iterrows()
    ])
    
    failure_reasons.extend([
        {'Vessel Name': row[COLUMN_NAMES['VESSEL_NAME']], 
         'Report Date': row[COLUMN_NAMES['REPORT_DATE']], 
         'Remarks': "Observed Distance is zero when steaming"}
        for _, row in df[mask_zero_when_steaming].iterrows()
    ])
    
    failure_reasons.extend([
        {'Vessel Name': row[COLUMN_NAMES['VESSEL_NAME']], 
         'Report Date': row[COLUMN_NAMES['REPORT_DATE']], 
         'Remarks': "Observed Distance not aligned with calculated distance"}
        for _, row in df[mask_not_aligned].iterrows()
    ])
    
    return failure_reasons

def validate_distance_data(batch_size=1000):
    df = fetch_validation_data()
    validation_results = []

    if df.empty:
        return pd.DataFrame(columns=['Vessel Name', 'Report Date', 'Remarks'])

    total_batches = len(df) // batch_size + (1 if len(df) % batch_size > 0 else 0)
    progress_bar = st.progress(0)
    progress_text = st.empty()

    for i in range(total_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(df))
        batch = df.iloc[start_idx:end_idx]
        
        batch_results = validate_distance_batch(batch)
        validation_results.extend(batch_results)
        
        progress = (i + 1) / total_batches
        progress_bar.progress(progress)
        progress_text.text(f"Validating: {progress:.0%}")

    progress_bar.empty()
    progress_text.empty()

    return pd.DataFrame(validation_results)

if __name__ == "__main__":
    results = validate_distance_data()
    print(results)
