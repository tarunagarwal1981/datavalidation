import pandas as pd
from config import COLUMN_NAMES

def is_value_in_range(value, min_val, max_val):
    """Check if a value is within a specified range."""
    return min_val <= value <= max_val if pd.notna(value) else False

def calculate_historical_average(df, consumption_column, days=30):
    """
    Calculate the historical average consumption for each vessel.
    
    :param df: DataFrame containing the vessel data
    :param consumption_column: Column name for the consumption data (ME or AE)
    :param days: Number of days to consider for historical data
    :return: Dictionary with vessel names as keys and average consumption as values
    """
    def calculate_avg_consumption(group):
        relevant_data = group.sort_values(COLUMN_NAMES['REPORT_DATE']).tail(days)
        if len(relevant_data) >= 10:  # Only calculate if we have at least 10 data points
            total_consumption = relevant_data[consumption_column].sum()
            total_steaming_time = relevant_data[COLUMN_NAMES['RUN_HOURS']].sum()
            if total_steaming_time > 0:
                return total_consumption / total_steaming_time
        return None

    return df.groupby(COLUMN_NAMES['VESSEL_NAME']).apply(calculate_avg_consumption).to_dict()

def is_value_within_percentage(value, reference, lower_percentage, upper_percentage):
    """Check if a value is within a specified percentage range of a reference value."""
    if pd.isna(value) or pd.isna(reference):
        return False
    lower_bound = reference * (1 - lower_percentage)
    upper_bound = reference * (1 + upper_percentage)
    return lower_bound <= value <= upper_bound

def calculate_power_based_consumption(power, run_hours, factor):
    """Calculate the maximum allowed consumption based on power and run hours."""
    return (factor / power) * run_hours / 10**6 if pd.notna(power) and pd.notna(run_hours) and power > 0 else None

def add_failure_reason(failure_reasons, reason):
    """Add a failure reason to the list if it's not already present."""
    if reason not in failure_reasons:
        failure_reasons.append(reason)

def validate_non_negative(value, field_name):
    """Validate that a value is non-negative."""
    if pd.notna(value) and value < 0:
        return f"{field_name} cannot be negative"
    return None

def calculate_expected_consumption(coefficients, speed, displacement, hull_performance_factor):
    """Calculate the expected consumption based on vessel coefficients and current conditions."""
    base_consumption = (coefficients['consp_speed1'] * speed +
                        coefficients['consp_disp1'] * displacement +
                        coefficients['consp_speed2'] * speed**2 +
                        coefficients['consp_disp2'] * displacement**2 +
                        coefficients['consp_intercept'])
    return base_consumption * hull_performance_factor
