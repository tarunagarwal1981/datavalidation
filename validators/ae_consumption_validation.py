import pandas as pd
from app.config import VALIDATION_THRESHOLDS, COLUMN_NAMES

def validate_ae_consumption(row, historical_data):
    failure_reasons = []
    ae_consumption = row[COLUMN_NAMES['AE_CONSUMPTION']]
    avg_ae_power = row[COLUMN_NAMES['AVG_AE_POWER']]
    ae_run_hours = row[COLUMN_NAMES['AE_RUN_HOURS']]

    if pd.notna(ae_consumption):
        if ae_consumption < VALIDATION_THRESHOLDS['ae_consumption']['min'] or ae_consumption > VALIDATION_THRESHOLDS['ae_consumption']['max']:
            failure_reasons.append("AE Consumption out of range")

        if pd.notna(avg_ae_power) and pd.notna(ae_run_hours) and avg_ae_power > 0:
            max_allowed_consumption = (VALIDATION_THRESHOLDS['ae_consumption']['power_factor'] / avg_ae_power) * ae_run_hours / 10**6
            if ae_consumption > max_allowed_consumption:
                failure_reasons.append("AE Consumption too high for the Reported power")

        if pd.notna(avg_ae_power) and avg_ae_power > 0 and ae_consumption == 0:
            failure_reasons.append("AE Consumption cannot be zero when generating power")

        if historical_data is not None and 'avg_ae_consumption' in historical_data:
            avg_consumption = historical_data['avg_ae_consumption']
            if pd.notna(avg_consumption):
                if not (VALIDATION_THRESHOLDS['ae_consumption']['historical_lower'] * avg_consumption <= ae_consumption <= VALIDATION_THRESHOLDS['ae_consumption']['historical_upper'] * avg_consumption):
                    failure_reasons.append("AE Consumption outside typical range")

        if ae_consumption == 0:
            failure_reasons.append("Total AE Consumption cannot be zero without Shaft Generator")
    else:
        failure_reasons.append("AE Consumption data is missing")

    return failure_reasons
