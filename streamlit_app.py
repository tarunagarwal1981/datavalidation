import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime, timedelta
import urllib.parse

# Define the column names based on your table structure
ME_CONSUMPTION_COL = 'actual_me_consumption'
ME_POWER_COL = 'actual_me_power'
ME_RPM_COL = 'me_rpm'
VESSEL_IMO_COL = 'vessel_imo'
RUN_HOURS_COL = 'steaming_time_hrs'
CURRENT_LOAD_COL = 'me_load_pct'
CURRENT_SPEED_COL = 'observed_speed'
STREAMING_HOURS_COL = 'steaming_time_hrs'
REPORT_DATE_COL = 'reportdate'
LOAD_TYPE_COL = 'load_type'
VESSEL_NAME_COL = 'vessel_name'
VESSEL_TYPE_COL = 'vessel_type'

# Sidebar information
st.sidebar.write("Data validation happened for the last 3 months.")

# Supabase connection details
supabase_host = "aws-0-ap-south-1.pooler.supabase.com"
supabase_database = "postgres"
supabase_user = "postgres.conrxbcvuogbzfysomov"
supabase_password = "wXAryCC8@iwNvj#"
supabase_port = "6543"

# URL encode the password
encoded_password = urllib.parse.quote(supabase_password)

# Function to create the SQLAlchemy engine using Supabase credentials
@st.cache_resource
def get_db_engine():
    db_url = f"postgresql+psycopg2://{supabase_user}:{encoded_password}@{supabase_host}:{supabase_port}/{supabase_database}"
    engine = create_engine(db_url)
    return engine

# Fetch the data for the last 3 months from the vessel_performance_summary table and join with vessel_particulars
def fetch_vessel_performance_data(engine):
    query = """
    SELECT vps.*, vp.vessel_type
    FROM vessel_performance_summary vps
    LEFT JOIN vessel_particulars vp ON vps.vessel_name = vp.vessel_name
    WHERE vps.reportdate >= %s;
    """
    three_months_ago = datetime.now() - timedelta(days=90)
    df = pd.read_sql_query(query, engine, params=(three_months_ago,))
    return df

# Calculate average consumption for the last 30 non-null data points for each vessel and load type
def calculate_avg_consumption(df, vessel_name, load_type):
    vessel_df = df[(df[VESSEL_NAME_COL] == vessel_name) & (df[LOAD_TYPE_COL] == load_type)]
    vessel_df = vessel_df.dropna(subset=[ME_CONSUMPTION_COL]).sort_values(by=REPORT_DATE_COL).tail(30)

    if not vessel_df.empty:
        total_consumption = vessel_df[ME_CONSUMPTION_COL].sum()
        total_steaming_time = vessel_df[RUN_HOURS_COL].sum()
        if total_steaming_time > 0:
            return total_consumption / total_steaming_time
    return None

# Run the validation logic for each vessel
def validate_data(df):
    validation_results = []
    
    for vessel_name, vessel_data in df.groupby(VESSEL_NAME_COL):
        vessel_type = vessel_data[VESSEL_TYPE_COL].iloc[0]  # Get vessel type for this vessel
        for _, row in vessel_data.iterrows():
            failure_reasons = []
            
            me_consumption = row[ME_CONSUMPTION_COL]
            me_power = row[ME_POWER_COL]
            me_rpm = row[ME_RPM_COL]
            run_hours = row[RUN_HOURS_COL]
            load_type = row[LOAD_TYPE_COL]
            
            if me_consumption < 0 or me_consumption > 300:
                failure_reasons.append("ME Consumption out of range")
            
            if me_consumption >= (250 * me_power * run_hours / 10**6):
                failure_reasons.append("ME Consumption too high for the Reported power")
            
            if me_rpm > 0 and me_consumption == 0:
                failure_reasons.append("ME Consumption cannot be zero when underway")
            
            if vessel_type == "CONTAINER" and me_consumption > 150:
                failure_reasons.append("ME Consumption too high for container vessel")
            elif vessel_type != "CONTAINER" and me_consumption > 60:
                failure_reasons.append("ME Consumption too high for non-container vessel")

            avg_consumption = calculate_avg_consumption(df, vessel_name, load_type)
            if avg_consumption is not None:
                if not (0.8 * avg_consumption <= me_consumption <= 1.2 * avg_consumption):
                    failure_reasons.append(f"ME Consumption outside typical range of {load_type} condition")

            if failure_reasons:
                validation_results.append({
                    'Vessel Name': vessel_name,
                    'Report Date': row[REPORT_DATE_COL],
                    'Remarks': ", ".join(failure_reasons)
                })
    
    return validation_results

# Main section of the Streamlit app
st.title('ME Consumption Validation')

# Button to validate data
if st.button('Validate Data'):
    engine = get_db_engine()

    try:
        df = fetch_vessel_performance_data(engine)
        
        if not df.empty:
            validation_results = validate_data(df)
            
            if validation_results:
                result_df = pd.DataFrame(validation_results)
                st.write("Validation Results:")
                st.dataframe(result_df)
                
                csv = result_df.to_csv(index=False)
                st.download_button(label="Download validation report as CSV", data=csv, file_name='validation_report.csv', mime='text/csv')
            else:
                st.write("All data passed the validation checks!")
        else:
            st.write("No data found for the last 3 months.")
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
