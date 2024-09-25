import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from validators.me_consumption_validation import validate_me_consumption, fetch_vessel_performance_data, fetch_vessel_coefficients, fetch_hull_performance_data
from validators.ae_consumption_validation import validate_ae_consumption
from validators.boiler_consumption_validation import validate_boiler_consumption, fetch_mcr_data
from validators.distance_validation import validate_distance_data

st.title('Vessel Data Validation')

# Sidebar information and filters
st.sidebar.write("Select the time range for data validation:")
time_range = st.sidebar.selectbox(
    "Validation Time Range",
    ("Last 1 Month", "Last 3 Months", "Last 6 Months")
)

# Convert time range to timedelta
if time_range == "Last 1 Month":
    date_filter = datetime.now() - timedelta(days=30)
elif time_range == "Last 3 Months":
    date_filter = datetime.now() - timedelta(days=90)
else:  # Last 6 Months
    date_filter = datetime.now() - timedelta(days=180)

# Validation criteria checkboxes
st.sidebar.write("Validation Criteria:")
me_consumption_check = st.sidebar.checkbox("ME Consumption", value=True)
ae_consumption_check = st.sidebar.checkbox("AE Consumption", value=True)
boiler_consumption_check = st.sidebar.checkbox("Boiler Consumption", value=True)
observed_distance_check = st.sidebar.checkbox("Observed Distance", value=True)

# Option to limit number of vessels
max_vessels = st.sidebar.number_input("Maximum number of vessels to process (0 for all)", min_value=0, value=0)

# Batch size for distance validation
batch_size = st.sidebar.number_input("Batch size for distance validation", min_value=100, max_value=10000, value=1000, step=100)

if st.button('Validate Data'):
    try:
        validation_results = []

        # Perform non-distance validations
        if me_consumption_check or ae_consumption_check or boiler_consumption_check:
            df = fetch_vessel_performance_data(date_filter)
            coefficients_df = fetch_vessel_coefficients()
            hull_performance_df = fetch_hull_performance_data()
            mcr_df = fetch_mcr_data()
            
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
                            boiler_failure_reasons = validate_boiler_consumption(row, mcr_value, date_filter)
                            failure_reasons.extend(boiler_failure_reasons)
                        
                        if failure_reasons:
                            validation_results.append({
                                'Vessel Name': vessel_name,
                                'Report Date': row['reportdate'],
                                'Remarks': ", ".join(failure_reasons)
                            })
                    
                    # Update progress
                    progress = (i + 1) / len(vessel_groups)
                    progress_bar.progress(progress)
                    progress_text.text(f"Validating: {progress:.0%}")
                
                progress_bar.empty()
                progress_text.empty()
        
        # Perform distance validation if selected
        if observed_distance_check:
            with st.spinner('Performing distance validation...'):
                distance_validation_results = validate_distance_data(date_filter, batch_size)
                validation_results.extend(distance_validation_results.to_dict('records'))
        
        # Combine all validation results
        all_results = pd.DataFrame(validation_results)
        
        if not all_results.empty:
            st.write("Validation Results:")
            st.dataframe(all_results)
            
            csv = all_results.to_csv(index=False)
            st.download_button(label="Download validation report as CSV", data=csv, file_name='validation_report.csv', mime='text/csv')
        else:
            st.write("All data passed the validation checks!")
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

st.write("This application validates vessel performance data based on selected criteria.")
st.write("Use the checkboxes in the sidebar to select which validations to run, then click the 'Validate Data' button to start the validation process.")
