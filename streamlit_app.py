import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from validators.me_consumption_validation import validate_me_consumption, fetch_vessel_performance_data, fetch_vessel_coefficients, fetch_hull_performance_data
from validators.ae_consumption_validation import validate_ae_consumption
from validators.boiler_consumption_validation import validate_boiler_consumption, fetch_mcr_data
from validators.distance_validation import validate_distance_data

def main():
    st.set_page_config(layout="wide")
    
    # Create three columns: left_sidebar, main_content, right_sidebar
    left_sidebar, main_content, right_sidebar = st.columns([1, 2, 1])

    with left_sidebar:
        st.sidebar.title('Validation Settings')
        time_range = st.sidebar.selectbox(
            "Validation Time Range",
            ("Last 1 Month", "Last 3 Months", "Last 6 Months"),
            key="time_range_select"
        )

        if time_range == "Last 1 Month":
            date_filter = datetime.now() - timedelta(days=30)
        elif time_range == "Last 3 Months":
            date_filter = datetime.now() - timedelta(days=90)
        else:  # Last 6 Months
            date_filter = datetime.now() - timedelta(days=180)

        st.sidebar.write("Validation Criteria:")
        me_consumption_check = st.sidebar.checkbox("ME Consumption", value=True, key="me_consumption_check")
        ae_consumption_check = st.sidebar.checkbox("AE Consumption", value=True, key="ae_consumption_check")
        boiler_consumption_check = st.sidebar.checkbox("Boiler Consumption", value=True, key="boiler_consumption_check")
        observed_distance_check = st.sidebar.checkbox("Observed Distance", value=True, key="observed_distance_check")

        max_vessels = st.sidebar.number_input("Maximum number of vessels to process (0 for all)", min_value=0, value=0, key="max_vessels")
        batch_size = st.sidebar.number_input("Batch size for distance validation", min_value=100, max_value=10000, value=1000, step=100, key="batch_size")

    with main_content:
        st.title('Vessel Data Validation')

        if st.button('Validate Data', key="validate_button"):
            try:
                validation_results = []

                if me_consumption_check or ae_consumption_check or boiler_consumption_check:
                    df = fetch_vessel_performance_data(date_filter)
                    coefficients_df = fetch_vessel_coefficients()
                    hull_performance_df = fetch_hull_performance_data()
                    mcr_df = fetch_mcr_data(date_filter)
                    
                    if not df.empty:
                        vessel_groups = list(df.groupby('vessel_name'))
                        if max_vessels > 0:
                            vessel_groups = vessel_groups[:max_vessels]
                        
                        progress_bar = st.progress(0)
                        progress_text = st.empty()
                        
                        for i, (vessel_name, vessel_data) in enumerate(vessel_groups):
                            vessel_type = vessel_data['vessel_type'].iloc[0]
                            vessel_coefficients = coefficients_df[coefficients_df['vessel_name'] == vessel_name].iloc[0] if not coefficients_df[coefficients_df['vessel_name'] == vessel_name].empty else None
                            
                            hull_performance = hull_performance_df[hull_performance_df['vessel_name'] == vessel_name]['hull_rough_power_loss_pct_ed'].iloc[0] if not hull_performance_df[hull_performance_df['vessel_name'] == vessel_name].empty else 0
                            hull_performance_factor = 1 + (hull_performance / 100)
                            
                            mcr_value = mcr_df[mcr_df['Vessel_Name'] == vessel_name]['ME_1_MCR_kW'].iloc[0] if not mcr_df[mcr_df['Vessel_Name'] == vessel_name].empty else None
                            mcr_value = float(mcr_value) if pd.notna(mcr_value) else None
                            
                            for _, row in vessel_data.iterrows():
                                failure_reasons = []
                                
                                if me_consumption_check:
                                    me_failure_reasons = validate_me_consumption(row, vessel_data, vessel_type, vessel_coefficients, hull_performance_factor)
                                    failure_reasons.extend(me_failure_reasons)
                                
                                if ae_consumption_check:
                                    ae_failure_reasons = validate_ae_consumption(row, vessel_data, date_filter)
                                    failure_reasons.extend(ae_failure_reasons)
                                
                                if boiler_consumption_check:
                                    boiler_failure_reasons = validate_boiler_consumption(row, mcr_value)
                                    failure_reasons.extend(boiler_failure_reasons)
                                
                                if failure_reasons:
                                    validation_results.append({
                                        'Vessel Name': vessel_name,
                                        'Report Date': row['reportdate'],
                                        'Remarks': ", ".join(failure_reasons)
                                    })
                            
                            progress = (i + 1) / len(vessel_groups)
                            progress_bar.progress(progress)
                            progress_text.text(f"Validating: {progress:.0%}")
                        
                        progress_bar.empty()
                        progress_text.empty()
                
                if observed_distance_check:
                    with st.spinner('Performing distance validation...'):
                        distance_validation_results = validate_distance_data(date_filter, batch_size)
                        validation_results.extend(distance_validation_results.to_dict('records'))
                
                all_results = pd.DataFrame(validation_results)
                
                if not all_results.empty:
                    st.write("Validation Results:")
                    st.dataframe(all_results)
                    
                    csv = all_results.to_csv(index=False)
                    st.download_button(label="Download validation report as CSV", data=csv, file_name='validation_report.csv', mime='text/csv', key="download_button")
                else:
                    st.write("All data passed the validation checks!")
            
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

        st.write("This application validates vessel performance data based on selected criteria.")
        st.write("Use the checkboxes in the sidebar to select which validations to run, then click the 'Validate Data' button to start the validation process.")

    with right_sidebar:
        st.sidebar.title("Validation Checks")
        
        st.sidebar.subheader("ME Consumption Validations")
        st.sidebar.write("1. Out of range: Checks if ME consumption is between 0 and 50.")
        st.sidebar.write("2. High for reported power: Compares ME consumption to a calculated maximum based on reported power.")
        st.sidebar.write("3. Zero when underway: Flags if ME consumption is zero when the vessel is moving.")
        st.sidebar.write("4. Vessel type limit: Checks against maximum limits for container and non-container vessels.")
        st.sidebar.write("5. Historical comparison: Compares to average consumption for the same load type in the last 30 days.")
        st.sidebar.write("6. Speed consumption alignment: Compares to expected consumption based on speed, displacement, and hull performance.")

        st.sidebar.subheader("AE Consumption Validations")
        st.sidebar.write("1. Out of range: Checks if AE consumption is between 0 and 50.")
        st.sidebar.write("2. High for reported power: Compares AE consumption to a calculated maximum based on reported power.")
        st.sidebar.write("3. Zero when generating: Flags if AE consumption is zero when AE power is being generated.")
        st.sidebar.write("4. Historical comparison: Compares to average consumption in the last 30 days.")
        st.sidebar.write("5. Zero total consumption: Flags if total AE consumption is zero (assuming no shaft generator).")

        st.sidebar.subheader("Boiler Consumption Validations")
        st.sidebar.write("1. Out of range: Checks if boiler consumption is between 0 and 100.")
        st.sidebar.write("2. Below cargo heating: Flags if consumption is less than expected cargo heating consumption.")
        st.sidebar.write("3. Non-zero at high ME load: Checks if consumption is non-zero when ME load is high during sea passage.")

        st.sidebar.subheader("Observed Distance Validations")
        st.sidebar.write("1. Negative distance: Flags if observed distance is negative.")
        st.sidebar.write("2. Excessive distance: Checks if observed distance is above a maximum threshold.")
        st.sidebar.write("3. Zero distance when steaming: Flags if observed distance is zero when steaming time is non-zero.")
        st.sidebar.write("4. Alignment with calculated: Compares observed distance with calculated distance based on GPS coordinates.")

if __name__ == "__main__":
    main()
