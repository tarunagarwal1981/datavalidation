import pandas as pd

# Configuration
COLUMN_NAMES = {
    'ME_CONSUMPTION': 'actual_me_consumption',
    'ME_POWER': 'actual_me_power',
    'ME_RPM': 'me_rpm',
    'RUN_HOURS': 'steaming_time_hrs',
    'LOAD_TYPE': 'load_type',
    'CURRENT_SPEED': 'observed_speed',
    'DISPLACEMENT': 'displacement',
    'STREAMING_HOURS': 'steaming_time_hrs',
    'REPORT_DATE': 'reportdate',
    'VESSEL_NAME': 'vessel_name'
}

VALIDATION_THRESHOLDS = {
    'min': 0,
    'max': 50,
    'power_factor': 250,
    'container_max': 300,
    'non_container_max': 50,
    'historical_lower': 0.8,
    'historical_upper': 1.2,
    'expected_lower': 0.8,
    'expected_upper': 1.2
}

# Utility functions
def is_value_in_range(value, min_val, max_val):
    return min_val <= value <= max_val if pd.notna(value) else False

def calculate_avg_consumption(vessel_df, load_type):
    relevant_data = vessel_df[vessel_df[COLUMN_NAMES['LOAD_TYPE']] == load_type].dropna(subset=[COLUMN_NAMES['ME_CONSUMPTION']])
    relevant_data = relevant_data.sort_values(by=COLUMN_NAMES['REPORT_DATE']).tail(30)
    if len(relevant_data) >= 10:  # Only calculate if we have at least 10 data points
        total_consumption = relevant_data[COLUMN_NAMES['ME_CONSUMPTION']].sum()
        total_steaming_time = relevant_data[COLUMN_NAMES['RUN_HOURS']].sum()
        if total_steaming_time > 0:
            return total_consumption / total_steaming_time
    return None

def calculate_expected_consumption(coefficients, speed, displacement, hull_performance_factor):
    base_consumption = (coefficients['consp_speed1'] * speed +
                        coefficients['consp_disp1'] * displacement +
                        coefficients['consp_speed2'] * speed**2 +
                        coefficients['consp_disp2'] * displacement**2 +
                        coefficients['consp_intercept'])
    return base_consumption * hull_performance_factor

def is_value_within_percentage(value, reference, lower_percentage, upper_percentage):
    if pd.isna(value) or pd.isna(reference):
        return False
    lower_bound = reference * (1 - lower_percentage)
    upper_bound = reference * (1 + upper_percentage)
    return lower_bound <= value <= upper_bound

def calculate_power_based_consumption(power, run_hours, factor):
    return (factor / power) * run_hours / 10**6 if pd.notna(power) and pd.notna(run_hours) and power > 0 else None

# Main validation function
def validate_me_consumption(row, vessel_data, vessel_type, vessel_coefficients, hull_performance_factor):
    failure_reasons = []
    me_consumption = row[COLUMN_NAMES['ME_CONSUMPTION']]
    me_power = row[COLUMN_NAMES['ME_POWER']]
    me_rpm = row[COLUMN_NAMES['ME_RPM']]
    run_hours = row[COLUMN_NAMES['RUN_HOURS']]
    load_type = row[COLUMN_NAMES['LOAD_TYPE']]
    current_speed = row[COLUMN_NAMES['CURRENT_SPEED']]
    displacement = row[COLUMN_NAMES['DISPLACEMENT']]
    streaming_hours = row[COLUMN_NAMES['STREAMING_HOURS']]

    if pd.notna(me_consumption):
        # Check range
        if not is_value_in_range(me_consumption, VALIDATION_THRESHOLDS['min'], VALIDATION_THRESHOLDS['max']):
            failure_reasons.append("ME Consumption out of range")
        
        # Check against power-based calculation
        max_allowed_consumption = calculate_power_based_consumption(me_power, run_hours, VALIDATION_THRESHOLDS['power_factor'])
        if max_allowed_consumption and me_consumption > max_allowed_consumption:
            failure_reasons.append("ME Consumption too high for the Reported power")
        
        # Check if zero when underway
        if pd.notna(me_rpm) and me_rpm > 0 and me_consumption == 0:
            failure_reasons.append("ME Consumption cannot be zero when underway")
        
        # Check against vessel type limits
        max_limit = VALIDATION_THRESHOLDS['container_max'] if vessel_type == "CONTAINER" else VALIDATION_THRESHOLDS['non_container_max']
        if me_consumption > max_limit:
            failure_reasons.append(f"ME Consumption too high for {vessel_type} vessel")

        # Historical comparison
        avg_consumption = calculate_avg_consumption(vessel_data, load_type)
        if avg_consumption is not None:
            if not is_value_within_percentage(me_consumption, avg_consumption, 
                                              VALIDATION_THRESHOLDS['historical_lower'], 
                                              VALIDATION_THRESHOLDS['historical_upper']):
                failure_reasons.append(f"ME Consumption outside typical range of {load_type} condition")

        # Expected consumption validation with hull performance
        if isinstance(vessel_coefficients, dict) and pd.notna(streaming_hours) and streaming_hours > 0:
            try:
                expected_consumption = calculate_expected_consumption(
                    vessel_coefficients, 
                    current_speed, 
                    displacement, 
                    hull_performance_factor
                )
                if not is_value_within_percentage(me_consumption, expected_consumption,
                                                  VALIDATION_THRESHOLDS['expected_lower'],
                                                  VALIDATION_THRESHOLDS['expected_upper']):
                    failure_reasons.append("ME Consumption not aligned with speed consumption table (including hull performance)")
            except KeyError as e:
                failure_reasons.append(f"Missing coefficient for expected consumption calculation: {str(e)}")
            except TypeError as e:
                failure_reasons.append(f"Error in expected consumption calculation: {str(e)}")

    else:
        failure_reasons.append("ME Consumption data is missing")

    # Check for negative values
    if pd.notna(me_consumption) and me_consumption < 0:
        failure_reasons.append("ME Consumption cannot be negative")

    return failure_reasons
