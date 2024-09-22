import streamlit as st
import pandas as pd
import psycopg2
from datetime import datetime, timedelta

# Define the column names based on your table structure
ME_CONSUMPTION_COL = 'actual_me_consumption'  # Column containing ME consumption
ME_POWER_COL = 'actual_me_power'  # Column containing ME power (kW)
ME_RPM_COL = 'me_rpm'  # Column containing ME RPM
VESSEL_TYPE_COL = 'vessel_type'  # Column containing vessel type (e.g., container)
RUN_HOURS_COL = 'steaming_time_hrs'  # Column containing engine run hours
CURRENT_LOAD_COL = 'me_load_pct'  # Column for load reference
CURRENT_SPEED_COL = 'observed_Speed'  # Column for vessel speed
STREAMING_HOURS_COL = 'steaming_time_hrs'  # Column for streaming hours
REPORT_DATE_COL = 'reportdate'  # Column for report date
VESSEL_NAME_COL = 'vessel_name'  # Column for vessel name

# Sidebar information
st.sidebar.write("Data validation happened for the last 6 months.")

# Database connection details
koyeb_host = "ep-rapid-wind-a1jdywyi.ap-southeast-1.pg.koyeb.app"
koyeb_database = "koyebdb"
koyeb_user = "koyeb-adm"
koyeb_password = "YBK7jd6wLaRD"
koyeb_port = "5432"

# Connect to the PostgreSQL database
@st.cache_resource
def get_db_connection():
    conn = psycopg2.connect(
        host=koyeb_host,
        database=koyeb_database,
        user=koyeb_user,
        password=koyeb_password,
        port=koyeb_port
    )
    return conn

# Fetch the data for the last 6 months with vessel_type from vessel_particulars
def fetch_data():
    conn = get_db_connection()
    
    # SQL Query that joins vessel_performance_summary and vessel_particulars based on vessel_name
    query = """
    SELECT sf.*, vp.vessel_type
    FROM vessel_performance_summary sf
    LEFT JOIN vessel_particulars vp ON sf.vessel_name = vp.vessel_name
    WHERE sf.reportdate >= %s;
    """
    
    six_months_ago = datetime.now() - timedelta(days=180)
    df = pd.read_sql_query(query, conn, params=[six_months_ago])
    conn.close()
    return df

# Check if the required columns are present in the DataFrame
def check_required_columns(df):
    required_columns = [ME_CONSUMPTION_COL, ME_POWER_COL, ME_RPM_COL, VESSEL_TYPE_COL,
                        RUN_HOURS_COL, CURRENT_LOAD_COL, CURRENT_SPEED_COL, STREAMING_HOURS_COL,
                        REPORT_DATE_COL, VESSEL_NAME_COL]
    missing_columns = [col for col in required_columns if col not in df.columns]
    return missing_columns

# Run the validation logic for each vessel group
def validate_data(df):
    validation_results = []
    
    if df.shape[0] < 20:
        validation_results.append({
            'Vessel Name': 'N/A',
            'Report Date': 'N/A',
            'Remarks': "Less than 20 data points in the filtered dataset"
        })
        return validation_results
    
    missing_columns = check_required_columns(df)
    if missing_columns:
        validation_results.append({
            'Vessel Name': 'N/A',
            'Report Date': 'N/A',
            'Remarks': f"Missing columns: {', '.join(missing_columns)}"
        })
        return validation_results

    # Group by vessel name and apply validation to each group
    grouped = df.groupby(VESSEL_NAME_COL)

    for vessel_name, vessel_data in grouped:
        for index, row in vessel_data.iterrows():
            failure_reason = []
            try:
                me_consumption = row[ME_CONSUMPTION_COL]
                me_power = row[ME_POWER_COL]
                me_rpm = row[ME_RPM_COL]
                vessel_type = row[VESSEL_TYPE_COL]  # Retrieved from vessel_particulars
                run_hours = row[RUN_HOURS_COL]
                current_load = row[CURRENT_LOAD_COL]
                current_speed = row[CURRENT_SPEED_COL]
                streaming_hours = row[STREAMING_HOURS_COL]
            except KeyError as e:
                failure_reason.append(f"Missing required column: {str(e)}")
                continue
            
            # Apply validation logic within the vessel group
            if me_consumption < 0 or me_consumption > 300:
                failure_reason.append("ME Consumption out of range")
            
            if me_consumption <= (250 / me_power * run_hours * 10**6):
                failure_reason.append("ME Consumption too high for the Reported power")
            
            if me_rpm > 0 and me_consumption == 0:
                failure_reason.append("ME Consumption cannot be zero when underway")
            
            if vessel_type == "container" and me_consumption > 150:
                failure_reason.append("ME Consumption too high for container vessel")
            elif vessel_type != "container" and me_consumption > 60:
                failure_reason.append("ME Consumption too high for non-container vessel")
            
            # Add other validation logics from your file here (e.g., avg_consumption, expected_consumption, etc.)
            
            # Collect the result if any validation failed
            if failure_reason:
                validation_results.append({
                    'Vessel Name': row[VESSEL_NAME_COL],
                    'Report Date': row[REPORT_DATE_COL],
                    'Remarks': ", ".join(failure_reason)
                })
    
    return validation_results

# Main section of the Streamlit app
st.title('ME Consumption Validation')

# Button to validate data
if st.button('Validate Data'):
    # Fetch data from the last 6 months with vessel_type from vessel_particulars
    df = fetch_data()
    
    if not df.empty:
        # Validate the data for each vessel group
        validation_results = validate_data(df)
        
        if validation_results:
            result_df = pd.DataFrame(validation_results)
            st.write("Validation Results:")
            st.dataframe(result_df)
            
            # Option to download results as CSV
            csv = result_df.to_csv(index=False)
            st.download_button(label="Download validation report as CSV", data=csv, file_name='validation_report.csv', mime='text/csv')
        else:
            st.write("All data passed the validation checks!")
    else:
        st.write("No data found for the last 6 months.")
