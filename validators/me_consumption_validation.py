import pandas as pd
from app.config import VALIDATION_THRESHOLDS, COLUMN_NAMES

def validate_me_consumption(row, vessel_type):
    failure_reasons = []
    me_consumption = row[COLUMN_NAMES['ME_CONSUMPTION']]
    me_power = row[COLUMN_NAMES['ME_POWER']]
    me_rpm = row[COLUMN_NAMES['ME_RPM']]
    run_hours = row[COLUMN_NAMES['RUN_HOURS']]

    if pd.notna(me_consumption):
        if me_consumption < VALIDATION_THRESHOLDS['me_consumption']['min'] or me_consumption > VALIDATION_THRESHOLDS['me_consumption']['max']:
            failure_reasons.append("ME Consumption out of range")
        
        if pd.notna(me_power) and pd.notna(run_hours):
            if me_consumption >= (VALIDATION_THRESHOLDS['me_consumption']['power_factor'] * me_power * run_hours / 10**6):
                failure_reasons.append("ME Consumption too high for the Reported power")
        
        if pd.notna(me_rpm) and me_rpm > 0 and me_consumption == 0:
            failure_reasons.append("ME Consumption cannot be zero when underway")
        
        if vessel_type == "CONTAINER" and me_consumption > VALIDATION_THRESHOLDS['me_consumption']['container_max']:
            failure_reasons.append("ME Consumption too high for container vessel")
        elif vessel_type != "CONTAINER" and me_consumption > VALIDATION_THRESHOLDS['me_consumption']['non_container_max']:
            failure_reasons.append("ME Consumption too high for non-container vessel")
    else:
        failure_reasons.append("ME Consumption data is missing")

    return failure_reasons
