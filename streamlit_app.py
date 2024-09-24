import streamlit as st
import pandas as pd
from database import get_db_engine, fetch_vessel_performance_data, fetch_vessel_coefficients, fetch_hull_performance_data
from validators import run_all_validations

st.title('Vessel Data Validation')

# Sidebar information
st.sidebar.write("Data validation happened for the last 3 months.")

if st.button('Validate Data'):
    engine = get_db_engine()

    try:
        df = fetch_vessel_performance_data(engine)
        coefficients_df = fetch_vessel_coefficients(engine)
        hull_performance_df = fetch_hull_performance_data(engine)
        
        if not df.empty:
            validation_results = run_all_validations(df, coefficients_df, hull_performance_df)
            
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
