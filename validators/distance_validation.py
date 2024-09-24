import pandas as pd

# Configuration
COLUMN_NAMES = {
    'OBSERVED_DISTANCE': 'observed_distance',
    'STEAMING_HOURS': 'steaming_time_hrs',
    'VESSEL_TYPE': 'vessel_type'
}

VALIDATION_THRESHOLDS = {
    'container_max_distance': 600,
    'non_container_max_distance': 500,
    'distance_tolerance_lower': 0.9,
    'distance_tolerance_upper': 1.1
}

def validate_distance(row, vessel_type):
    failure_reasons = []
    observed_distance = row[COLUMN_NAMES['OBSERVED_DISTANCE']]
    steaming_hours = row[COLUMN_NAMES['STEAMING_HOURS']]

    if pd.isna(observed_distance):
        failure_reasons.append("Observed Distance data is missing")
        return failure_reasons

    # Check for negative distance
    if observed_distance < 0:
        failure_reasons.append("Observed Distance cannot be negative")

    # Check for maximum distance based on vessel type
    if vessel_type == "CONTAINER" and observed_distance > VALIDATION_THRESHOLDS['container_max_distance']:
        failure_reasons.append("Observed Distance too high for container vessel")
    elif vessel_type != "CONTAINER" and observed_distance > VALIDATION_THRESHOLDS['non_container_max_distance']:
        failure_reasons.append("Observed Distance too high for non-container vessel")

    # Check alignment with steaming hours
    if steaming_hours > 0:
        if observed_distance == 0:
            failure_reasons.append("Observed Distance cannot be zero when streaming")

    return failure_reasons
