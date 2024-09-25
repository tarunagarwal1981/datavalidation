import pandas as pd
import math
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
    'container_max_distance': 600,
    'non_container_max_distance': 500,
    'distance_alignment_lower': 0.9,
    'distance_alignment_upper': 1.1
}

@st.cache_data
def fetch_validation_data():
    engine = get_db_engine()
    try:
        query = f"""
        SELECT *
        FROM sf_consumption_logs
        ORDER BY "VESSEL_NAME", "REPORT_DATE"
        """
        return pd.read_sql_query(query, engine)
    except SQLAlchemyError as e:
        st.error(f"Error fetching validation data: {str(e)}")
        return pd.DataFrame()

def chord_distance(lat1, lon1, lat2, lon2, radius=6371):
    phi1, lambda1, phi2, lambda2 = map(math.radians, [lat1, lon1, lat2, lon2])
    d_lambda = lambda2 - lambda1
    
    a = (math.sin((phi2-phi1)/2)**2 + 
         math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda/2)**2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return radius * c

def validate_distance(row, prev_row):
    failure_reasons = []
    
    required_columns = [COLUMN_NAMES['OBSERVED_DISTANCE'], COLUMN_NAMES['STEAMING_TIME_HRS'], 
                        COLUMN_NAMES['LATITUDE'], COLUMN_NAMES['LONGITUDE']]
    if not all(col in row.index for col in required_columns):
        return ["Missing required columns for distance validation"]

    observed_distance = row[COLUMN_NAMES['OBSERVED_DISTANCE']]
    steaming_time_hrs = row[COLUMN_NAMES['STEAMING_TIME_HRS']]

    if pd.isna(observed_distance):
        return ["Observed Distance data is missing"]

    if observed_distance < 0:
        failure_reasons.append("Observed Distance cannot be negative")

    if observed_distance > VALIDATION_THRESHOLDS['non_container_max_distance']:
        failure_reasons.append("Observed Distance too high")

    if pd.notna(steaming_time_hrs) and steaming_time_hrs > 0:
        if observed_distance == 0:
            failure_reasons.append("Observed Distance cannot be zero when streaming")
        else:
            expected_distance = steaming_time_hrs * 1  # Assuming average speed of 1 for now
            if not (VALIDATION_THRESHOLDS['distance_alignment_lower'] * expected_distance <= observed_distance <= VALIDATION_THRESHOLDS['distance_alignment_upper'] * expected_distance):
                failure_reasons.append("Observed Distance not aligned with steaming hours")

    if prev_row is not None:
        try:
            calculated_distance = chord_distance(
                row[COLUMN_NAMES['LATITUDE']], row[COLUMN_NAMES['LONGITUDE']],
                prev_row[COLUMN_NAMES['LATITUDE']], prev_row[COLUMN_NAMES['LONGITUDE']]
            )
            if observed_distance < calculated_distance:
                failure_reasons.append("Observed Distance less than calculated distance between positions")
        except Exception as e:
            failure_reasons.append(f"Error calculating distance: {str(e)}")

    return failure_reasons

def validate_distance_data():
    df = fetch_validation_data()
    validation_results = []

    if df.empty:
        return pd.DataFrame(columns=['Vessel Name', 'Report Date', 'Remarks'])

    total_rows = len(df)
    progress_bar = st.progress(0)
    progress_text = st.empty()

    for i, (vessel_name, vessel_data) in enumerate(df.groupby('VESSEL_NAME')):
        prev_row = None
        for _, row in vessel_data.iterrows():
            failure_reasons = validate_distance(row, prev_row)
            if failure_reasons:
                validation_results.append({
                    'Vessel Name': vessel_name,
                    'Report Date': row['REPORT_DATE'],
                    'Remarks': ", ".join(failure_reasons)
                })
            prev_row = row
        
        # Update progress
        progress = (i + 1) / len(df.groupby('VESSEL_NAME'))
        progress_bar.progress(progress)
        progress_text.text(f"Validating: {progress:.0%}")

    progress_bar.empty()
    progress_text.empty()

    return pd.DataFrame(validation_results)

if __name__ == "__main__":
    results = validate_distance_data()
    print(results)
