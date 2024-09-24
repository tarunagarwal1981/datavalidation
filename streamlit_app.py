import streamlit as st
import pandas as pd
from database import get_db_engine, fetch_vessel_performance_data, fetch_vessel_coefficients, fetch_hull_performance_data, fetch_mcr_data
from validators.me_consumption_validation import validate_me_consumption
from validators.ae_consumption_validation import validate_ae_consumption
from validators.boiler_consumption_validation import validate_boiler_consumption
from validators.distance_validation import validate_distance

st.title('Vessel Data Validation')

# Sidebar information
st.sidebar.write("Data validation happened for the last 3 months.")

if st.button('Validate Data'):
    engine = get_db_engine()

    try:
        df = fetch_vessel_performance_data(engine)
        coefficients_df = fetch_vessel_coefficients(engine)
        hull_performance_df = fetch_hull_performance_data(engine)
        mcr_df = fetch_mcr_data(engine)
        
        if not df.empty:
            validation_results = []
            
            for vessel_name, vessel_data in df.groupby('vessel_name'):
                vessel_type = vessel_data['vessel_type'].iloc[0]
                vessel_coefficients = coefficients_df[coefficients_df['vessel_name'] == vessel_name].iloc[0] if not coefficients_df[coefficients_df['vessel_name'] == vessel_name].empty else None
                
                hull_performance = hull_performance_df[hull_performance_df['vessel_name'] == vessel_name]['hull_rough_power_loss_pct_ed'].iloc[0] if not hull_performance_df[hull_performance_df['vessel_name'] == vessel_name].empty else 0
                hull_performance_factor = 1 + (hull_performance / 100)
                
                mcr_value = mcr_df[mcr_df['Vessel_Name'] == vessel_name]['ME_1_MCR_kW'].iloc[0] if not mcr_df[mcr_df['Vessel_Name'] == vessel_name].empty else None
                mcr_value = float(mcr_value) if pd.notna(mcr_value) else None
                
                for _, row in vessel_data.iterrows():
                    me_failure_reasons = validate_me_consumption(row, vessel_data, vessel_type, vessel_coefficients, hull_performance_factor)
                    ae_failure_reasons = validate_ae_consumption(row, vessel_data)
                    boiler_failure_reasons = validate_boiler_consumption(row, mcr_value)
                    distance_failure_reasons = validate_distance(row, vessel_type)
                    
                    failure_reasons = me_failure_reasons + ae_failure_reasons + boiler_failure_reasons + distance_failure_reasons
                    
                    if failure_reasons:
                        validation_results.append({
                            'Vessel Name': vessel_name,
                            'Report Date': row['reportdate'],
                            'Remarks': ", ".join(failure_reasons)
                        })
            
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

# Optional: Add more Streamlit components for user interaction or data visualization
st.sidebar.write("Validation Criteria:")
st.sidebar.write("- ME Consumption")
st.sidebar.write("- AE Consumption")
st.sidebar.write("- Boiler Consumption")
st.sidebar.write("- Observed Distance")

st.write("This application validates vessel performance data based on multiple criteria.")
st.write("Click the 'Validate Data' button to start the validation process.")
