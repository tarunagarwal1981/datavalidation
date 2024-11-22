import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_log,
    after_log
)
from sqlalchemy.exc import SQLAlchemyError

# Import validators
from validators.me_consumption_validation import (
    validate_me_consumption, fetch_vessel_performance_data,
    fetch_vessel_coefficients, fetch_hull_performance_data
)
from validators.ae_consumption_validation import validate_ae_consumption
from validators.boiler_consumption_validation import (
    validate_boiler_consumption, fetch_mcr_data
)
from validators.distance_validation import validate_distance_data
from validators.speed_validation import validate_speed, fetch_speed_data
from validators.fuel_rob_validation import (
    validate_fuel_rob_for_vessel, fetch_sf_consumption_logs
)
from validators.advanced_validation import run_advanced_validation
from validators.slip_validation import validate_slip_percentage, fetch_slip_data
from database import get_db_engine, DatabaseConnection

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataFetcher:
    """Handle all data fetching operations with proper error handling"""
    
    def __init__(self, engine):
        self.engine = engine

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(SQLAlchemyError),
        before=before_log(logger, logging.DEBUG),
        after=after_log(logger, logging.DEBUG)
    )
    def fetch_vessel_data(self, date_filter: datetime) -> pd.DataFrame:
        """Fetch vessel performance data with retries"""
        try:
            return fetch_vessel_performance_data(self.engine, date_filter)
        except Exception as e:
            logger.error(f"Error fetching vessel data: {str(e)}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(SQLAlchemyError),
        before=before_log(logger, logging.DEBUG),
        after=after_log(logger, logging.DEBUG)
    )
    def fetch_all_data(self, date_filter: datetime) -> Dict[str, pd.DataFrame]:
        """Fetch all required data with retries"""
        try:
            logger.info("Starting data fetch process")
            
            vessel_data = self.fetch_vessel_data(date_filter)
            if vessel_data.empty:
                logger.warning("No vessel performance data found")
                return {}

            data = {
                'vessel_performance': vessel_data,
                'coefficients': fetch_vessel_coefficients(self.engine),
                'hull_performance': fetch_hull_performance_data(self.engine),
                'mcr': fetch_mcr_data(self.engine, date_filter),
                'sf_consumption': fetch_sf_consumption_logs(self.engine, date_filter),
                'slip': fetch_slip_data(self.engine, date_filter)
            }
            
            logger.info("Successfully fetched all required data")
            return data
            
        except Exception as e:
            logger.error(f"Error in fetch_all_data: {str(e)}")
            raise

class ValidationApp:
    def __init__(self):
        st.set_page_config(layout="wide")
        self.engine = None
        self.data_fetcher = None
        self.validation_results = []
        self.date_filter = None

    def initialize_database(self):
        """Initialize database connection"""
        try:
            self.engine = get_db_engine()
            if self.engine is None:
                raise ConnectionError("Failed to establish database connection")
            self.data_fetcher = DataFetcher(self.engine)
            return True
        except Exception as e:
            logger.error(f"Database initialization error: {str(e)}")
            st.error("Failed to connect to database. Please check your configuration.")
            return False

    def setup_sidebar(self) -> Tuple[datetime, Dict[str, bool], int, int]:
        """Setup sidebar controls and return configuration"""
        with st.sidebar:
            st.title('Validation Settings')
            
            # Time range selection
            time_range = st.selectbox(
                "Validation Time Range",
                ("Last 1 Month", "Last 3 Months", "Last 6 Months"),
                key="time_range_select"
            )
            
            date_ranges = {
                "Last 1 Month": 30,
                "Last 3 Months": 90,
                "Last 6 Months": 180
            }
            self.date_filter = datetime.now() - timedelta(days=date_ranges[time_range])

            # Validation checkboxes
            st.write("Validation Criteria:")
            validation_checks = {
                'me_consumption': st.checkbox("ME Consumption", value=True, key="me_consumption_check"),
                'ae_consumption': st.checkbox("AE Consumption", value=True, key="ae_consumption_check"),
                'boiler_consumption': st.checkbox("Boiler Consumption", value=True, key="boiler_consumption_check"),
                'observed_distance': st.checkbox("Observed Distance", value=True, key="observed_distance_check"),
                'speed': st.checkbox("Speed", value=True, key="speed_check"),
                'fuel_rob': st.checkbox("Fuel ROB", value=True, key="fuel_rob_check"),
                'slip': st.checkbox("Slip Validation", value=True, key="slip_check"),
                'advanced': st.checkbox("Run Advanced Validations", value=False, key="advanced_validation_check")
            }

            # Process control parameters
            max_vessels = st.number_input(
                "Maximum number of vessels to process (0 for all)",
                min_value=0, value=0, key="max_vessels"
            )
            batch_size = st.number_input(
                "Batch size for distance validation",
                min_value=100, max_value=10000, value=1000, step=100,
                key="batch_size"
            )

            return validation_checks, max_vessels, batch_size

    def process_vessel_data(self, vessel_name: str, vessel_data: pd.DataFrame,
                          all_data: Dict[str, pd.DataFrame],
                          validation_checks: Dict[str, bool],
                          vessel_type_cache: Dict[str, str]) -> List[dict]:
        """Process data for a single vessel"""
        results = []
        
        try:
            # Extract relevant data for the vessel
            coefficients = all_data['coefficients']
            hull_performance_df = all_data['hull_performance']
            mcr_df = all_data['mcr']
            
            vessel_type = vessel_data['vessel_type'].iloc[0]
            vessel_coefficients = (
                coefficients[coefficients['vessel_name'] == vessel_name].iloc[0]
                if not coefficients[coefficients['vessel_name'] == vessel_name].empty else None
            )
            
            hull_performance = (
                hull_performance_df[hull_performance_df['vessel_name'] == vessel_name]['hull_rough_power_loss_pct_ed'].iloc[0]
                if not hull_performance_df[hull_performance_df['vessel_name'] == vessel_name].empty else 0
            )
            hull_performance_factor = 1 + (hull_performance / 100)
            
            mcr_value = (
                mcr_df[mcr_df['Vessel_Name'] == vessel_name]['ME_1_MCR_kW'].iloc[0]
                if not mcr_df[mcr_df['Vessel_Name'] == vessel_name].empty else None
            )
            mcr_value = float(mcr_value) if pd.notna(mcr_value) else None

            # Process each row
            for _, row in vessel_data.iterrows():
                failure_reasons = []
                
                try:
                    # Run enabled validations
                    if validation_checks['me_consumption']:
                        me_results = validate_me_consumption(
                            row, vessel_data, vessel_type,
                            vessel_coefficients, hull_performance_factor
                        )
                        failure_reasons.extend(me_results or [])
                        
                    if validation_checks['ae_consumption']:
                        ae_results = validate_ae_consumption(
                            row, vessel_data, self.date_filter
                        )
                        failure_reasons.extend(ae_results or [])
                        
                    if validation_checks['boiler_consumption']:
                        boiler_results = validate_boiler_consumption(
                            row, mcr_value
                        )
                        failure_reasons.extend(boiler_results or [])
                        
                    if validation_checks['speed']:
                        speed_results = validate_speed(row, vessel_type_cache)
                        failure_reasons.extend(speed_results or [])
                        
                    if validation_checks['fuel_rob']:
                        fuel_rob_failures = validate_fuel_rob_for_vessel(
                            all_data['sf_consumption'], vessel_name
                        )
                        if fuel_rob_failures:
                            failure_reasons.extend([
                                failure['Remarks'] for failure in fuel_rob_failures
                                if failure['Report Date'] == row['reportdate']
                            ])
                        
                    if validation_checks['slip']:
                        slip_results = validate_slip_percentage(row)
                        failure_reasons.extend(slip_results or [])

                    if failure_reasons:
                        results.append({
                            'Vessel Name': vessel_name,
                            'Report Date': row['reportdate'],
                            'Remarks': ", ".join(failure_reasons)
                        })
                        
                except Exception as e:
                    logger.error(f"Error processing row for vessel {vessel_name}: {str(e)}")
                    st.warning(f"Skipped some validations for vessel {vessel_name} due to errors")
                    continue

        except Exception as e:
            logger.error(f"Error processing vessel {vessel_name}: {str(e)}")
            st.warning(f"Error processing vessel {vessel_name}")

        return results

    def run_validation(self, validation_checks: Dict[str, bool],
                      max_vessels: int, batch_size: int):
        """Run the validation process"""
        try:
            if not self.initialize_database():
                return
                
            st.success("Database connection established successfully.")
            
            # Fetch all required data
            with st.spinner('Fetching required data...'):
                all_data = self.data_fetcher.fetch_all_data(self.date_filter)
                
            if not all_data:
                st.warning("No data found for the selected time period.")
                return
            
            df = all_data['vessel_performance']
            
            # Process vessels
            vessel_groups = list(df.groupby('vessel_name'))
            if max_vessels > 0:
                vessel_groups = vessel_groups[:max_vessels]

            # Setup progress tracking
            progress_bar = st.progress(0)
            progress_text = st.empty()
            vessel_type_cache = {}

            # Process each vessel
            for i, (vessel_name, vessel_data) in enumerate(vessel_groups):
                try:
                    # Process regular validations
                    results = self.process_vessel_data(
                        vessel_name, vessel_data, all_data,
                        validation_checks, vessel_type_cache
                    )
                    self.validation_results.extend(results or [])

                    # Run advanced validation if enabled
                    if validation_checks['advanced']:
                        self.run_advanced_validation_for_vessel(vessel_name)

                except Exception as e:
                    logger.error(f"Error processing vessel {vessel_name}: {str(e)}")
                    st.warning(f"Error processing vessel {vessel_name}")
                    continue
                finally:
                    # Update progress
                    progress = (i + 1) / len(vessel_groups)
                    progress_bar.progress(progress)
                    progress_text.text(f"Validating: {progress:.0%}")

            # Clear progress indicators
            progress_bar.empty()
            progress_text.empty()

            # Run distance validation if enabled
            if validation_checks['observed_distance']:
                with st.spinner('Performing distance validation...'):
                    try:
                        distance_results = validate_distance_data(
                            self.engine, self.date_filter, batch_size
                        )
                        if not distance_results.empty:
                            self.validation_results.extend(distance_results.to_dict('records'))
                    except Exception as e:
                        logger.error(f"Error in distance validation: {str(e)}")
                        st.warning("Error during distance validation")

            # Display results
            self.display_results()

        except Exception as e:
            logger.error(f"Validation error: {str(e)}")
            st.error(f"An error occurred during validation: {str(e)}")


            # Clear progress indicators
            progress_bar.empty()
            progress_text.empty()

            # Run distance validation if enabled
            if validation_checks['observed_distance']:
                with st.spinner('Performing distance validation...'):
                    distance_results = validate_distance_data(
                        self.engine, self.date_filter, batch_size
                    )
                    self.validation_results.extend(distance_results.to_dict('records'))

            # Display results
            self.display_results()

        except Exception as e:
            logger.error(f"Validation error: {str(e)}")
            st.error(f"An error occurred during validation: {str(e)}")

    def display_results(self):
        """Display validation results"""
        if not self.validation_results:
            st.success("All data passed the validation checks!")
            return
            
        results_df = pd.DataFrame(self.validation_results)
        st.write("Validation Results:")
        st.dataframe(results_df)
        
        # Add download button
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="Download validation report as CSV",
            data=csv,
            file_name='validation_report.csv',
            mime='text/csv',
            key="download_button"
        )

    def run(self):
        """Main application entry point"""
        st.title('Vessel Data Validation')
        
        # Setup sidebar and get configuration
        validation_checks, max_vessels, batch_size = self.setup_sidebar()
        
        # Display information sidebar
        self.display_right_sidebar()
        
        # Main content area
        with st.container():
            st.write(
                "This application validates vessel performance data based on "
                "selected criteria. Use the checkboxes in the sidebar to select "
                "which validations to run, then click the 'Validate Data' button "
                "to start the validation process."
            )
            
            if st.button('Validate Data', key="validate_button"):
                self.run_validation(validation_checks, max_vessels, batch_size)

def main():
    try:
        app = ValidationApp()
        app.run()
    except Exception as e:
        logger.error(f"Application error: {str(e)}", exc_info=True)
        st.error("An unexpected error occurred. Please check the logs for details.")

if __name__ == "__main__":
    main()
