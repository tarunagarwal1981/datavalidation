import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime, timedelta
import urllib.parse

# Define the column names based on your table structure
ME_CONSUMPTION_COL = 'actual_me_consumption'  # Column containing ME consumption
ME_POWER_COL = 'actual_me_power'  # Column containing ME power (kW)
ME_RPM_COL = 'me_rpm'  # Column containing ME RPM
VESSEL_IMO_COL = 'vessel_imo'  # Column for vessel IMO in vessel_performance_summary
RUN_HOURS_COL = 'steaming_time_hrs'  # Column containing engine run hours
CURRENT_LOAD_COL = 'me_load_pct'  # Column for load reference
CURRENT_SPEED_COL = 'observed_Speed'  # Column for vessel speed
STREAMING_HOURS_COL = 'steaming_time_hrs'  # Column for streaming hours
REPORT_DATE_COL = 'reportdate'  # Column for report date
LOAD_TYPE_COL = 'load_type'  # Column for load type

# Columns in vessel_particulars table
VESSEL_NAME_COL = 'vessel_name'  # Column for vessel name in vessel_particulars table
VESSEL_TYPE_COL = 'vessel_type'  # Vessel type from vessel_particulars table
VESSEL_IMO_PARTICULARS_COL = 'vessel_imo'  # IMO column in vessel_particulars table

# Sidebar information
st.sidebar.write("Data validation happened for the last 6 months.")

# Supabase connection details
supabase_host = "aws-0-ap-south-1.pooler.supabase.com"
supabase_database = "postgres"
supabase_user = "postgres.conrxbcvuogbzfysomov"
supabase_password = "wXAryCC8@iwNvj#"  # Your original password
supabase_port = "6543"

# URL encode the password
encoded_password = urllib.parse.quote(supabase_password)

# Function to create the SQLAlchemy engine using Supabase credentials
@st.cache_resource
def get_db_engine():
    db_url = f"postgresql+psycopg2://{supabase_user}:{encoded_password}@{supabase_host}:{supabase_port}/{supabase_database}"
    engine = create_engine(db_url)
    return engine

# Fetch the data for the last 6 months from the vessel_performance_summary table
def fetch_vessel_performance_data(engine):
    query = """
    SELECT * FROM vessel_performance_summary
    WHERE reportdate >= %s;
    """
    six_months_ago = datetime.now() - timedelta(days=180)
    df = pd.read_sql_query(query, engine, params=(six_months_ago,))
    return df

# Fetch vessel name and type information from the vessel_particulars table using vessel_imo
def fetch_vessel_type_data(engine):
    query = """
    SELECT vessel_imo, vessel_name, vessel_type FROM vessel_particulars;
    """
    vessel_type_df = pd.read_sql_query(query, engine)
    return vessel_type_df

# Merge vessel type data with vessel performance data using vessel_imo
def merge_vessel_type(df_performance, df_particulars):
    # Convert both vessel_imo columns to string to avoid merge issues
    df_performance[VESSEL_IMO_COL] = df_performance[VESSEL_IMO_COL].astype(str)
    df_particulars[VESSEL_IMO_PARTICULARS_COL] = df_particulars[VESSEL_IMO_PARTICULARS_COL].astype(str)
    
    # Merge the two dataframes on the vessel_imo column
    merged_df = pd.merge(df_performance, df_particulars, left_on=VESSEL_IMO_COL, right_on=VESSEL_IMO_PARTICULARS_COL, how='left')
    return merged_df

# Calculate average consumption for the last 30 non-null data points for each vessel and load type
def calculate_avg_consumption(df, vessel_imo, load_type):
    # Filter for the vessel and load type
    vessel_df = df[(df[VESSEL_IMO_COL] == vessel_imo) & (df[LOAD_TYPE_COL] == load_type)]
    # Sort by report date and filter the last 30 non-null ME consumption data points
    vessel_df = vessel_df.dropna(subset=[ME_CONSUMPTION_COL]).sort_values(by=REPORT_DATE_COL).tail(30)

    if not vessel_df.empty:
        # Calculate avg_consumption as total ME consumption / total steaming time
        total_consumption = vessel_df[ME_CONSUMPTION_COL].sum()
        total_steaming_time = vessel_df[RUN_HOURS_COL].sum()
        if total_steaming_time > 0:
            avg_consumption = total_consumption / total_steaming_time
            return avg_consumption
    return None

# Check if the required columns are present in the DataFrame
def check_required_columns(df):
    required_columns = [ME_CONSUMPTION_COL, ME_POWER_COL, ME_RPM_COL, VESSEL_IMO_COL, 
                        RUN_HOURS_COL, CURRENT_LOAD_COL, CURRENT_SPEED_COL, STREAMING_HOURS_COL,
                        REPORT_DATE_COL, VESSEL_TYPE_COL, LOAD_TYPE_COL]
    missing_columns = [col for col in required_columns if col not in df.columns]
    return missing_columns

# Run the validation logic for each vessel group
def validate_data(df):
    validation_results = []
    
    if df.shape[0] < 20:
        validation_results.append({
            'Vessel IMO': 'N/A',
            'Report Date': 'N/A',
            'Remarks': "Less than 20 data points in the filtered dataset"
        })
        return validation_results
    
    missing_columns = check_required_columns(df)
    if missing_columns:
        validation_results.append({
            'Vessel IMO': 'N/A',
            'Report Date': 'N/A',
            'Remarks': f"Missing columns: {', '.join(missing_columns)}"
        })
        return validation_results

    # Group by vessel IMO and apply validation to each group
    grouped = df.groupby(VESSEL_IMO_COL)

    for vessel_imo, vessel_data in grouped:
        for index, row in vessel_data.iterrows():
            failure_reason = []
            try:
                me_consumption = row[ME_CONSUMPTION_COL]
                me_power = row[ME_POWER_COL]
                me_rpm = row[ME_RPM_COL]
                vessel_type = row[VESSEL_TYPE_COL]
                run_hours = row[RUN_HOURS_COL]
                current_load = row[CURRENT_LOAD_COL]
                current_speed = row[CURRENT_SPEED_COL]
                streaming_hours = row[STREAMING_HOURS_COL]
                load_type = row[LOAD_TYPE_COL]
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

            # Calculate the average consumption for the last 30 points
            avg_consumption = calculate_avg_consumption(df, vessel_imo, load_type)
            if avg_consumption is not None:
                if not (0.8 * avg_consumption <= me_consumption <= 1.2 * avg_consumption):
                    failure_reason.append(f"ME Consumption outside typical range of {load_type} condition")

            # New validation for expected consumption based on speed
            if streaming_hours > 0:
                expected_consumption = get_speed_consumption_table(current_speed)  # Simulated speed-based consumption
                if not (0.8 * expected_consumption <= me_consumption <= 1.2 * expected_consumption):
                    failure_reason.append(f"ME Consumption not aligned with speed consumption table")

            # Collect the result if any validation failed
            if failure_reason:
                validation_results.append({
                    'Vessel IMO': row[VESSEL_IMO_COL],
                    'Report Date': row[REPORT_DATE_COL],
                    'Remarks': ", ".join(failure_reason)
                })
    
    return validation_results

# Main section of the Streamlit app
st.title('ME Consumption Validation')

# Button to validate data
if st.button('Validate Data'):
    # Create the database engine using SQLAlchemy
    engine = get_db_engine()

    try:
        # Fetch data from vessel performance and particulars tables
        df_performance = fetch_vessel_performance_data(engine)
        df_particulars = fetch_vessel_type_data(engine)

        # Merge vessel type from particulars into the performance data using vessel_imo
        df = merge_vessel_type(df_performance, df_particulars)
        
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
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
