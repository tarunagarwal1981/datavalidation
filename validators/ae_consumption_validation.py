import pandas as pd

# Configuration
COLUMN_NAMES = {
    'AE_CONSUMPTION': 'aux_engine_consumption',
    'AVG_AE_POWER': 'avg_ae_power',
    'AE_RUN_HOURS': 'total_ae_running_hours',
    'REPORT_DATE': 'reportdate'
}import pandas as pd

# Configuration
COLUMN_NAMES = {
    'AE_CONSUMPTION': 'aux_engine_consumption',
    'AVG_AE_POWER': 'avg_ae_power',
    'AE_RUN_HOURS': 'total_ae_running_hours',
    'REPORT_DATE': 'reportdate'
}

VALIDATION_THRESHOLDS = {
    'min': 0,
    'max': 50,
    'power_factor': 300,
    'historical_lower': 0.8,
    'historical_upper': 1.2
}

# Utility functions
def is_value_in_range(value, min_val, max_val):
    return min_val <= value <= max_val if pd.notna(value) else False

def calculate_historical_average(df, days=30):
    def calculate_avg_consumption(group):
        relevant_data = group.sort_values(COLUMN_NAMES['REPORT_DATE']).tail(days)
        if len(relevant_data) >= 10:
            total_consumption = relevant_data[COLUMN_NAMES['AE_CONSUMPTION']].sum()
            total_running_time = relevant_data[COLUMN_NAMES['AE_RUN_HOURS']].sum()
            if total_running_time > 0:
                return total_consumption / total_running_time
        return None
    return df.groupby('vessel_name').apply(calculate_avg_consumption).to_dict()

def is_value_within_percentage(value, reference, lower_percentage, upper_percentage):
    if pd.isna(value) or pd.isna(reference):
        return False
    lower_bound = reference * (lower_percentage)
    upper_bound = reference * (upper_percentage)
    return lower_bound <= value <= upper_bound

def calculate_power_based_consumption(power, run_hours, factor):
    return (factor * power) * run_hours / 10**6 if pd.notna(power) and pd.notna(run_hours) and power > 0 else None

# Main validation function
def validate_ae_consumption(row, vessel_data):
    failure_reasons = []
    ae_consumption = row[COLUMN_NAMES['AE_CONSUMPTION']]
    avg_ae_power = row[COLUMN_NAMES['AVG_AE_POWER']]
    ae_run_hours = row[COLUMN_NAMES['AE_RUN_HOURS']]

    if pd.notna(ae_consumption):
        # Check range
        if not is_value_in_range(ae_consumption, VALIDATION_THRESHOLDS['min'], VALIDATION_THRESHOLDS['max']):
            failure_reasons.append("AE Consumption out of range")

        # Check against power-based calculation
        max_allowed_consumption = calculate_power_based_consumption(avg_ae_power, ae_run_hours, VALIDATION_THRESHOLDS['power_factor'])
        if max_allowed_consumption and ae_consumption > max_allowed_consumption:
            failure_reasons.append("AE Consumption too high for the Reported power")

        # Check if zero when generating power
        if pd.notna(avg_ae_power) and avg_ae_power > 0 and ae_consumption == 0:
            failure_reasons.append("AE Consumption cannot be zero when generating power")

        # Historical comparison
        historical_data = calculate_historical_average(vessel_data)
        if historical_data:
            avg_consumption = historical_data.get(row['vessel_name'])
            if avg_consumption is not None:
                if not is_value_within_percentage(ae_consumption, avg_consumption, 
                                                  VALIDATION_THRESHOLDS['historical_lower'], 
                                                  VALIDATION_THRESHOLDS['historical_upper']):
                    failure_reasons.append("AE Consumption outside typical range")

        # Check if total AE Consumption is zero (assuming no shaft generator)
        if ae_consumption == 0:
            failure_reasons.append("Total AE Consumption cannot be zero without Shaft Generator")
    else:
        failure_reasons.append("AE Consumption data is missing")

    # Check for negative values
    if pd.notna(ae_consumption) and ae_consumption < 0:
        failure_reasons.append("AE Consumption cannot be negative")

    return failure_reasons

VALIDATION_THRESHOLDS = {
    'min': 0,
    'max': 50,
    'power_factor': 300,
    'historical_lower': 0.8,
    'historical_upper': 1.2
}

# Utility functions
def is_value_in_range(value, min_val, max_val):
    return min_val <= value <= max_val if pd.notna(value) else False

def calculate_historical_average(df, days=30):
    def calculate_avg_consumption(group):
        relevant_data = group.sort_values(COLUMN_NAMES['REPORT_DATE']).tail(days)
        if len(relevant_data) >= 10:
            total_consumption = relevant_data[COLUMN_NAMES['AE_CONSUMPTION']].sum()
            total_running_time = relevant_data[COLUMN_NAMES['AE_RUN_HOURS']].sum()
            if total_running_time > 0:
                return total_consumption / total_running_time
        return None
    return df.groupby('vessel_name').apply(calculate_avg_consumption).to_dict()

def is_value_within_percentage(value, reference, lower_percentage, upper_percentage):
    if pd.isna(value) or pd.isna(reference):
        return False
    lower_bound = reference * (lower_percentage)
    upper_bound = reference * (upper_percentage)
    return lower_bound <= value <= upper_bound

def calculate_power_based_consumption(power, run_hours, factor):
    return (factor * power) * run_hours / 10**6 if pd.notna(power) and pd.notna(run_hours) and power > 0 else None

# Main validation function
def validate_ae_consumption(row, vessel_data):
    failure_reasons = []
    ae_consumption = row[COLUMN_NAMES['AE_CONSUMPTION']]
    avg_ae_power = row[COLUMN_NAMES['AVG_AE_POWER']]
    ae_run_hours = row[COLUMN_NAMES['AE_RUN_HOURS']]

    if pd.notna(ae_consumption):
        # Check range
        if not is_value_in_range(ae_consumption, VALIDATION_THRESHOLDS['min'], VALIDATION_THRESHOLDS['max']):
            failure_reasons.append("AE Consumption out of range")

        # Check against power-based calculation
        max_allowed_consumption = calculate_power_based_consumption(avg_ae_power, ae_run_hours, VALIDATION_THRESHOLDS['power_factor'])
        if max_allowed_consumption and ae_consumption > max_allowed_consumption:
            failure_reasons.append("AE Consumption too high for the Reported power")

        # Check if zero when generating power
        if pd.notna(avg_ae_power) and avg_ae_power > 0 and ae_consumption == 0:
            failure_reasons.append("AE Consumption cannot be zero when generating power")

        # Historical comparison
        historical_data = calculate_historical_average(vessel_data)
        if historical_data:
            avg_consumption = historical_data.get(row['vessel_name'])
            if avg_consumption is not None:
                if not is_value_within_percentage(ae_consumption, avg_consumption, 
                                                  VALIDATION_THRESHOLDS['historical_lower'], 
                                                  VALIDATION_THRESHOLDS['historical_upper']):
                    failure_reasons.append("AE Consumption outside typical range")

        # Check if total AE Consumption is zero (assuming no shaft generator)
        if ae_consumption == 0:
            failure_reasons.append("Total AE Consumption cannot be zero without Shaft Generator")
    else:
        failure_reasons.append("AE Consumption data is missing")

    # Check for negative values
    if pd.notna(ae_consumption) and ae_consumption < 0:
        failure_reasons.append("AE Consumption cannot be negative")

    return failure_reasons
