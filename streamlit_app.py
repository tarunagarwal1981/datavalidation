import streamlit as st
import pandas as pd
import logging
from datetime import datetime, timedelta
from sqlalchemy.exc import SQLAlchemyError

# Import validations
from validators import (
    me_consumption_validation as me_validator,
    ae_consumption_validation as ae_validator,
    boiler_consumption_validation as boiler_validator,
    distance_validation as distance_validator,
    speed_validation as speed_validator,
    fuel_rob_validation as fuel_validator,
    slip_validation as slip_validator,
    advanced_validation as advanced_validator
)
from database import get_db_engine

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_page():
    """Configure Streamlit page settings"""
    st.set_page_config(layout="wide")
    st.title('Vessel Data Validation')

def get_date_filter(time_range: str) -> datetime:
    """Convert time range selection to date filter"""
    days_map = {
        "Last 1 Month": 30,
        "Last 3 Months": 90,
        "Last 6 Months": 180
    }
    return datetime.now() - timedelta(days=days_map[time_range])

def setup_sidebar():
    """Configure sidebar controls"""
    with st.sidebar:
        st.title('Validation Settings')
        
        # Time range selection
        time_range = st.selectbox(
            "Validation Time Range",
            ("Last 1 Month", "Last 3 Months", "Last 6 Months")
        )
        date_filter = get_date_filter(time_range)
        
        # Validation selections
        st.write("Validation Criteria:")
        settings = {
            'me_consumption': st.checkbox("ME Consumption", value=True),
            'ae_consumption': st.checkbox("AE Consumption", value=True),
            'boiler_consumption': st.checkbox("Boiler Consumption", value=True),
            'observed_distance': st.checkbox("Observed Distance", value=True),
            'speed': st.checkbox("Speed", value=True),
            'fuel_rob': st.checkbox("Fuel ROB", value=True),
            'slip': st.checkbox("Slip Validation", value=True),
            'advanced': st.checkbox("Advanced Validation", value=False),
            'max_vessels': st.number_input(
                "Maximum vessels to process (0 for all)",
                min_value=0, value=0
            ),
            'batch_size': st.number_input(
                "Batch size for distance validation",
                min_value=100, max_value=10000,
                value=1000, step=100
            )
        }
        
        return date_filter, settings

def fetch_data(engine, date_filter):
    """Fetch all required data from database"""
    try:
        with st.spinner('Fetching data...'):
            data = {
                'vessel_performance': me_validator.fetch_vessel_performance_data(engine, date_filter),
                'coefficients': me_validator.fetch_vessel_coefficients(engine),
                'hull_performance': me_validator.fetch_hull_performance_data(engine),
                'mcr': boiler_validator.fetch_mcr_data(engine, date_filter),
                'sf_consumption': fuel_validator.fetch_sf_consumption_logs(engine, date_filter),
                'slip': slip_validator.fetch_slip_data(engine, date_filter)
            }
            
            if data['vessel_performance'].empty:
                st.warning("No vessel performance data found for the selected period.")
                return None
                
            return data
            
    except SQLAlchemyError as e:
        logger.error(f"Database error: {str(e)}")
        st.error("Failed to fetch data from database.")
        return None
    except Exception as e:
        logger.error(f"Error fetching data: {str(e)}")
        st.error("An error occurred while fetching data.")
        return None

def process_vessel(vessel_name, vessel_data, all_data, settings):
    """Process validation for a single vessel"""
    validation_results = []
    try:
        # Get vessel-specific data
        coefficients_df = all_data['coefficients']
        vessel_coefficients = (
            coefficients_df[coefficients_df['vessel_name'] == vessel_name].iloc[0]
            if not coefficients_df[coefficients_df['vessel_name'] == vessel_name].empty
            else None
        )

        hull_df = all_data['hull_performance']
        hull_performance = (
            hull_df[hull_df['vessel_name'] == vessel_name]['hull_rough_power_loss_pct_ed'].iloc[0]
            if not hull_df[hull_df['vessel_name'] == vessel_name].empty
            else 0
        )
        hull_factor = 1 + (hull_performance / 100)

        mcr_df = all_data['mcr']
        mcr_value = None
        if not mcr_df[mcr_df['Vessel_Name'] == vessel_name].empty:
            mcr_raw = mcr_df[mcr_df['Vessel_Name'] == vessel_name]['ME_1_MCR_kW'].iloc[0]
            mcr_value = float(mcr_raw) if pd.notna(mcr_raw) else None

        vessel_type = vessel_data['vessel_type'].iloc[0]
        vessel_type_cache = {}

        # Process each row
        for _, row in vessel_data.iterrows():
            failures = []
            
            try:
                if settings['me_consumption']:
                    me_results = me_validator.validate_me_consumption(
                        row, vessel_data, vessel_type,
                        vessel_coefficients, hull_factor
                    )
                    failures.extend(me_results or [])

                if settings['ae_consumption']:
                    ae_results = ae_validator.validate_ae_consumption(row, vessel_data)
                    failures.extend(ae_results or [])

                if settings['boiler_consumption']:
                    boiler_results = boiler_validator.validate_boiler_consumption(row, mcr_value)
                    failures.extend(boiler_results or [])

                if settings['speed']:
                    speed_results = speed_validator.validate_speed(row, vessel_type_cache)
                    failures.extend(speed_results or [])

                if settings['fuel_rob']:
                    fuel_results = fuel_validator.validate_fuel_rob_for_vessel(
                        all_data['sf_consumption'], vessel_name
                    )
                    failures.extend([
                        f['Remarks'] for f in fuel_results
                        if f['Report Date'] == row['reportdate']
                    ])

                if settings['slip']:
                    slip_results = slip_validator.validate_slip_percentage(row)
                    failures.extend(slip_results or [])

                if failures:
                    validation_results.append({
                        'Vessel Name': vessel_name,
                        'Report Date': row['reportdate'],
                        'Remarks': ", ".join(failures)
                    })

            except Exception as e:
                logger.error(f"Error processing row for {vessel_name}: {str(e)}")
                continue

    except Exception as e:
        logger.error(f"Error processing vessel {vessel_name}: {str(e)}")
        st.warning(f"Error processing vessel {vessel_name}")

    return validation_results

def run_validation(engine, data, settings, date_filter):
    """Run validation process for all vessels"""
    validation_results = []
    
    # Get vessel groups
    vessel_groups = list(data['vessel_performance'].groupby('vessel_name'))
    if settings['max_vessels'] > 0:
        vessel_groups = vessel_groups[:settings['max_vessels']]

    # Setup progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Process each vessel
    for i, (vessel_name, vessel_data) in enumerate(vessel_groups):
        try:
            # Regular validations
            results = process_vessel(vessel_name, vessel_data, data, settings)
            validation_results.extend(results)

            # Advanced validation
            if settings['advanced']:
                try:
                    advanced_results = advanced_validator.run_advanced_validation(
                        engine, vessel_name, date_filter
                    )
                    if advanced_results:
                        st.write(f"Advanced Validation - {vessel_name}:")
                        st.write(f"Anomalies: {len(advanced_results['anomalies'])}")
                        st.write("Drift detected:", 
                               ", ".join([f for f, d in advanced_results['drift'].items() if d]))
                except Exception as e:
                    logger.error(f"Advanced validation error for {vessel_name}: {str(e)}")

        except Exception as e:
            logger.error(f"Error processing vessel {vessel_name}: {str(e)}")
            continue

        finally:
            # Update progress
            progress = (i + 1) / len(vessel_groups)
            progress_bar.progress(progress)
            status_text.text(f"Processing vessel {i+1} of {len(vessel_groups)}")

    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()

    # Distance validation
    if settings['observed_distance']:
        try:
            with st.spinner('Running distance validation...'):
                distance_results = distance_validator.validate_distance_data(
                    engine, date_filter, settings['batch_size']
                )
                if not distance_results.empty:
                    validation_results.extend(distance_results.to_dict('records'))
        except Exception as e:
            logger.error(f"Distance validation error: {str(e)}")
            st.warning("Error during distance validation")

    return validation_results

def display_results(results):
    """Display validation results"""
    if not results:
        st.success("All data passed validation!")
        return

    df = pd.DataFrame(results)
    st.write("Validation Results:")
    st.dataframe(df)

    csv = df.to_csv(index=False)
    st.download_button(
        "Download Results",
        csv,
        "validation_results.csv",
        "text/csv"
    )

def main():
    setup_page()
    
    # Get settings from sidebar
    date_filter, settings = setup_sidebar()
    
    if st.button('Validate Data'):
        try:
            # Initialize database connection
            engine = get_db_engine()
            if engine is None:
                st.error("Database connection failed")
                return

            # Fetch required data
            data = fetch_data(engine, date_filter)
            if data is None:
                return

            # Run validation
            results = run_validation(engine, data, settings, date_filter)
            
            # Display results
            display_results(results)

        except Exception as e:
            logger.error(f"Validation error: {str(e)}")
            st.error("An error occurred during validation")

if __name__ == "__main__":
    main()
