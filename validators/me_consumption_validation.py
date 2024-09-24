import pandas as pd
from config import VALIDATION_THRESHOLDS, COLUMN_NAMES
from utils.validation_utils import (
    is_value_in_range, 
    calculate_power_based_consumption, 
    is_value_within_percentage,
    add_failure_reason,
    validate_non_negative
)

def validate_me_consumption(row, vessel_type, historical_data):
    failure_reasons = []
    me_consumption = row[COLUMN_NAMES['ME_CONSUMPTION']]
    me_power = row[COLUMN_NAMES['ME_POWER']]
    me_rpm = row[COLUMN_NAMES['ME_RPM']]
    run_hours = row[COLUMN_NAMES['RUN_HOURS']]

    if pd.notna(me_consumption):
        # Check range
        if not is_value_in_range(me_consumption, VALIDATION_THRESHOLDS['me_consumption']['min'], VALIDATION_THRESHOLDS['me_consumption']['max']):
            add_failure_reason(failure_reasons, "ME Consumption out of range")
        
        # Check against power-based calculation
        max_allowed_consumption = calculate_power_based_consumption(me_power, run_hours, VALIDATION_THRESHOLDS['me_consumption']['power_factor'])
        if max_allowed_consumption and me_consumption > max_allowed_consumption:
            add_failure_reason(failure_reasons, "ME Consumption too high for the Reported power")
        
        # Check if zero when underway
        if pd.notna(me_rpm) and me_rpm > 0 and me_consumption == 0:
            add_failure_reason(failure_reasons, "ME Consumption cannot be zero when underway")
        
        # Check against vessel type limits
        max_limit = VALIDATION_THRESHOLDS['me_consumption']['container_max'] if vessel_type == "CONTAINER" else VALIDATION_THRESHOLDS['me_consumption']['non_container_max']
        if me_consumption > max_limit:
            add_failure_reason(failure_reasons, f"ME Consumption too high for {vessel_type.lower()} vessel")

        # Historical comparison
        if historical_data and 'avg_me_consumption' in historical_data:
            if not is_value_within_percentage(me_consumption, historical_data['avg_me_consumption'], 
                                              VALIDATION_THRESHOLDS['me_consumption']['historical_lower'], 
                                              VALIDATION_THRESHOLDS['me_consumption']['historical_upper']):
                add_failure_reason(failure_reasons, "ME Consumption outside typical range")

    else:
        add_failure_reason(failure_reasons, "ME Consumption data is missing")

    # Check for negative values
    negative_check = validate_non_negative(me_consumption, "ME Consumption")
    if negative_check:
        add_failure_reason(failure_reasons, negative_check)

    return failure_reasons
