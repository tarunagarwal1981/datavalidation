import pandas as pd
from database import get_db_engine
from sqlalchemy.exc import SQLAlchemyError
import streamlit as st

# Configuration
COLUMN_NAMES = {
    'VESSEL_NAME': 'VESSEL_NAME',
    'REPORT_DATE': 'REPORT_DATE',
    'ROB_HSFO': 'ROB_HSFO',
    'ROB_LSMGO': 'ROB_LSMGO',
    'ROB_ULSFO': 'ROB_ULSFO',
    'ROB_VLSFO': 'ROB_VLSFO',
    'ROB_MDO': 'ROB_MDO',
    'ROB_LNG': 'ROB_LNG',
    'BUNKERED_QTY_HSFO': 'BUNKERED_QTY_HSFO',
    'BUNKERED_QTY_LSMGO': 'BUNKERED_QTY_LSMGO',
    'BUNKERED_QTY_VLSFO': 'BUNKERED_QTY_VLSFO',
    'BUNKERED_QTY_ULSFO': 'BUNKERED_QTY_ULSFO',
    'BUNKERED_QTY_MDO': 'BUNKERED_QTY_MDO',
    'BUNKERED_QTY_LNG': 'BUNKERED_QTY_LNG',
    'TOTAL_CONSUMPTION_HSFO': 'TOTAL_CONSUMPTION_HSFO',
    'TOTAL_CONSUMPTION_LSMGO': 'TOTAL_CONSUMPTION_LSMGO',
    'TOTAL_CONSUMPTION_MDO': 'TOTAL_CONSUMPTION_MDO',
    'TOTAL_CONSUMPTION_ULSFO': 'TOTAL_CONSUMPTION_ULSFO',
    'TOTAL_CONSUMPTION_VLSFO': 'TOTAL_CONSUMPTION_VLSFO',
    'TOTAL_CONSUMPTION_LNG': 'TOTAL_CONSUMPTION_LNG'
}

FUEL_TYPES = ['HSFO', 'LSMGO', 'ULSFO', 'VLSFO', 'MDO', 'LNG']

@st.cache_data
def fetch_sf_consumption_logs(date_filter):
    engine = get_db_engine()
    try:
        query = f"""
        SELECT {', '.join(f'"{col}"' for col in COLUMN_NAMES.values())}
        FROM sf_consumption_logs
        WHERE "{COLUMN_NAMES['REPORT_DATE']}" >= %s
        ORDER BY "{COLUMN_NAMES['VESSEL_NAME']}", "{COLUMN_NAMES['REPORT_DATE']}"
        """
        df = pd.read_sql_query(query, engine, params=(date_filter,))
        # Replace null values with 0 for numeric columns
        numeric_columns = [col for col in df.columns if df[col].dtype in ['float64', 'int64']]
        df[numeric_columns] = df[numeric_columns].fillna(0)
        return df
    except SQLAlchemyError as e:
        st.error(f"Error fetching sf_consumption_logs data: {str(e)}")
        return pd.DataFrame()

def safe_float(value):
    """Convert value to float, returning 0 if conversion fails."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0

def validate_fuel_rob_batch(df):
    failure_reasons = []

    for i in range(1, len(df)):
        current_row = df.iloc[i]
        previous_row = df.iloc[i-1]

        for fuel_type in FUEL_TYPES:
            current_rob = safe_float(current_row[COLUMN_NAMES[f'ROB_{fuel_type}']])
            prev_rob = safe_float(previous_row[COLUMN_NAMES[f'ROB_{fuel_type}']])
            bunkered_qty = safe_float(current_row[COLUMN_NAMES[f'BUNKERED_QTY_{fuel_type}']])
            total_consumption = safe_float(current_row[COLUMN_NAMES[f'TOTAL_CONSUMPTION_{fuel_type}']])

            calculated_rob = round(prev_rob + bunkered_qty - total_consumption, 2)
            current_rob = round(current_rob, 2)

            if calculated_rob != current_rob:
                failure_reasons.append({
                    'Vessel Name': current_row[COLUMN_NAMES['VESSEL_NAME']],
                    'Report Date': current_row[COLUMN_NAMES['REPORT_DATE']],
                    'Remarks': f"{fuel_type} ROB validation failed. Calculated: {calculated_rob:.2f}, Actual: {current_rob:.2f}, Difference: {abs(current_rob - calculated_rob):.2f}"
                })

    return failure_reasons

def validate_fuel_rob_for_vessel(df, vessel_name, batch_size=1000):
    vessel_df = df[df[COLUMN_NAMES['VESSEL_NAME']] == vessel_name].sort_values(COLUMN_NAMES['REPORT_DATE'])
    validation_results = []

    total_batches = len(vessel_df) // batch_size + (1 if len(vessel_df) % batch_size > 0 else 0)
    progress_bar = st.progress(0)
    progress_text = st.empty()

    for i in range(total_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(vessel_df))
        batch = vessel_df.iloc[start_idx:end_idx]
        
        batch_results = validate_fuel_rob_batch(batch)
        validation_results.extend(batch_results)
        
        progress = (i + 1) / total_batches
        progress_bar.progress(progress)
        progress_text.text(f"Validating Fuel ROB for {vessel_name}: {progress:.0%}")

    progress_bar.empty()
    progress_text.empty()

    return validation_results

if __name__ == "__main__":
    from datetime import datetime, timedelta
    
    date_filter = datetime.now() - timedelta(days=30)  # Example: last 30 days
    df = fetch_sf_consumption_logs(date_filter)
    
    if not df.empty:
        vessel_name = df[COLUMN_NAMES['VESSEL_NAME']].iloc[0]  # Example: validate for the first vessel in the data
        results = validate_fuel_rob_for_vessel(df, vessel_name)
        print(pd.DataFrame(results))
    else:
        print("No data available for validation.")
