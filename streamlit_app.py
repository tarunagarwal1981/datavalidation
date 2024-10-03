import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from validators.me_consumption_validation import validate_me_consumption, fetch_vessel_performance_data, fetch_vessel_coefficients, fetch_hull_performance_data
from validators.ae_consumption_validation import validate_ae_consumption
from validators.boiler_consumption_validation import validate_boiler_consumption, fetch_mcr_data
from validators.distance_validation import validate_distance_data
from validators.speed_validation import validate_speed, fetch_speed_data
from validators.fuel_rob_validation import validate_fuel_rob_for_vessel, fetch_sf_consumption_logs
from validators.advanced_validation import run_advanced_validation  # Import the advanced validation function
from database import get_db_engine  # Import the database engine function

st.set_page_config(layout="wide")

def main():
    # Create columns for main content and right sidebar
    main_content, right_sidebar = st.columns([3, 1])

    with st.sidebar:
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
        speed_check = st.sidebar.checkbox("Speed", value=True, key="speed_check")
        fuel_rob_check = st.sidebar.checkbox("Fuel ROB", value=True, key="fuel_rob_check")
        
        # New: Advanced Validation checkbox
        advanced_validation_check = st.sidebar.checkbox("Run Advanced Validations", value=False, key="advanced_validation_check")

        max_vessels = st.sidebar.number_input("Maximum number of vessels to process (0 for all)", min_value=0, value=0, key="max_vessels")
        batch_size = st.sidebar.number_input("Batch size for distance validation", min_value=100, max_value=10000, value=1000, step=100, key="batch_size")

    with main_content:
        st.title('Vessel Data Validation')

        if st.button('Validate Data', key="validate_button"):
            try:
                validation_results = []
                engine = get_db_engine()  # Get the database engine

                if me_consumption_check or ae_consumption_check or boiler_consumption_check or speed_check or fuel_rob_check or advanced_validation_check:
                    df = fetch_vessel_performance_data(date_filter)
                    coefficients_df = fetch_vessel_coefficients()
                    hull_performance_df = fetch_hull_performance_data()
                    mcr_df = fetch_mcr_data(date_filter)
                    sf_consumption_logs = fetch_sf_consumption_logs(date_filter)
                    
                    if not df.empty:
                        vessel_groups = list(df.groupby('vessel_name'))
                        if max_vessels > 0:
                            vessel_groups = vessel_groups[:max_vessels]
                        
                        progress_bar = st.progress(0)
                        progress_text = st.empty()
                        
                        vessel_type_cache = {}
                        
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
                                
                                if speed_check:
                                    speed_failure_reasons = validate_speed(row, vessel_type_cache)
                                    failure_reasons.extend(speed_failure_reasons)
                                
                                if fuel_rob_check:
                                    fuel_rob_failures = validate_fuel_rob_for_vessel(sf_consumption_logs, vessel_name)
                                    failure_reasons.extend([failure['Remarks'] for failure in fuel_rob_failures if failure['Report Date'] == row['reportdate']])
                                
                                if failure_reasons:
                                    validation_results.append({
                                        'Vessel Name': vessel_name,
                                        'Report Date': row['reportdate'],
                                        'Remarks': ", ".join(failure_reasons)
                                    })
                            
                            # New: Advanced Validation
                            if advanced_validation_check:
                            advanced_results = run_advanced_validation(engine, vessel_name, date_filter)
                            
                            st.write(f"Advanced Validation Results for {vessel_name}:")
                            st.write(f"Anomalies detected: {len(advanced_results['anomalies'])}")
                            st.write("Drift detected in features:", ", ".join([f for f, d in advanced_results['drift'].items() if d]))
                            
                            st.write("Change points detected:")
                            for feature, points in advanced_results['change_points'].items():
                                st.write(f"  {feature}: {points}")
                            
                            st.write("Feature relationships (Mutual Information):")
                            for feature, mi in advanced_results['relationships'].items():
                                st.write(f"  {feature}: {mi:.4f}")

                            st.write("---")
                            
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
        st.markdown("<h2 style='font-size: 18px;'>Validation Checks</h2>", unsafe_allow_html=True)
        
        st.markdown("<h3 style='font-size: 14px;'>ME Consumption Validations</h3>", unsafe_allow_html=True)
        st.markdown("...")  # Add content as needed

        st.markdown("<h3 style='font-size: 14px;'>AE Consumption Validations</h3>", unsafe_allow_html=True)
        st.markdown("...")  # Add content as needed

        st.markdown("<h3 style='font-size: 14px;'>Boiler Consumption Validations</h3>", unsafe_allow_html=True)
        st.markdown("...")  # Add content as needed

        st.markdown("<h3 style='font-size: 14px;'>Observed Distance Validations</h3>", unsafe_allow_html=True)
        st.markdown("...")  # Add content as needed

        st.markdown("<h3 style='font-size: 14px;'>Speed Validations</h3>", unsafe_allow_html=True)
        st.markdown("...")  # Add content as needed

        st.markdown("<h3 style='font-size: 14px;'>Advanced Validations</h3>", unsafe_allow_html=True)
        st.markdown("...")  # Add content as needed

        # New: Advanced Validations description
        st.markdown("<h3 style='font-size: 14px;'>Advanced Validations</h3>", unsafe_allow_html=True)
        st.markdown("""
        <div style='font-size: 10px;'>
        1. Anomaly Detection: Uses ensemble of Isolation Forest and Local Outlier Factor to detect unusual data points.<br>
        2. Drift Detection: Identifies significant changes in data distribution over time.<br>
        3. Change Point Detection: Detects abrupt changes in time series data.<br>
        4. Feature Relationship Analysis: Measures mutual information between features to understand complex dependencies.
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
    
