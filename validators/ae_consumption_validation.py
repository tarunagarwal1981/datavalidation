import pandas as pd
from config import VALIDATION_THRESHOLDS, COLUMN_NAMES
from utils.validation_utils import (
    is_value_in_range, 
    calculate_power_based_consumption, 
    is_value_within_percentage,
    add_failure_reason,
    validate_non_negative
)

def validate_ae_consumption(row, historical_data):
    failure_reasons = []
    ae_consumption = row[COLUMN_NAMES['AE_CONSUMPTION']]
    avg_ae_power = row[COLUMN_NAMES['AVG_AE_POWER']]
    ae_run_hours = row[COLUMN_NAMES['AE_RUN_HOURS']]

    if pd.notna(ae_consumption):
        # Check range
        if not is_value_in_range(ae_consumption, VALIDATION_THRESHOLDS['ae_consumption']['min'], VALIDATION_THRESHOLDS['ae_consumption']['max']):
            add_failure_reason(failure_reasons, "AE Consumption out of range")

        # Check against power-based calculation
        max_allowed_consumption = calculate_power_based_consumption(avg_ae_power, ae_run_hours, VALIDATION_THRESHOLDS['ae_consumption']['power_factor'])
        if max_allowed_consumption and ae_consumption > max_allowed_consumption:
            add_failure_reason(failure_reasons, "AE Consumption too high for the Reported power")

        # Check if zero when generating power
        if pd.notna(avg_ae_power) and avg_ae_power > 0 and ae_consumption == 0:
            add_failure_reason(failure_reasons, "AE Consumption cannot be zero when generating power")

        # Historical comparison
        if historical_data and 'avg_ae_consumption' in historical_data:
            if not is_value_within_percentage(ae_consumption, historical_data['avg_ae_consumption'], 
                                              VALIDATION_THRESHOLDS['ae_consumption']['historical_lower'], 
                                              VALIDATION_THRESHOLDS['ae_consumption']['historical_upper']):
                add_failure_reason(failure_reasons, "AE Consumption outside typical range")

        # Check if total AE Consumption is zero (assuming no shaft generator)
        if ae_consumption == 0:
            add_failure_reason(failure_reasons, "Total AE Consumption cannot be zero without Shaft Generator")
    else:
        add_failure_reason(failure_reasons, "AE Consumption data is missing")

    # Check for negative values
    negative_check = validate_non_negative(ae_consumption, "AE Consumption")
    if negative_check:
        add_failure_reason(failure_reasons, negative_check)

    return failure_reasons
