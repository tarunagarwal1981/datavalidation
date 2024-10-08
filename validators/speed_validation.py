import pandas as pd
from database import get_db_engine

COLUMN_NAMES = {
    'ME_CONSUMPTION': 'actual_me_consumption',
    'ME_RPM': 'me_rpm',
    'STEAMING_HOURS': 'steaming_time_hrs',
    'ME_RUN_HOURS': 'steaming_time_hrs',
    'SPEED': 'observed_speed',
    'DISPLACEMENT': 'displacement',
    
    'REPORT_DATE': 'reportdate',
    'VESSEL_NAME': 'vessel_name',
    'OBSERVED_DISTANCE': 'observed_distance',
    'EVENT': 'event'
}

# # Configuration
# COLUMN_NAMES = {
#     'SPEED': 'SPEED',
#     'VESSEL_NAME': 'VESSEL_NAME',
#     'EVENT': 'EVENT',
#     'STEAMING_HOURS': 'STEAMING_TIME_HRS',
#     'OBSERVED_DISTANCE': 'OBSERVERD_DISTANCE',
#     'ME_RPM': 'MERPM',
#     'ME_RUN_HOURS': 'ME_RUNNING_HOURS_DAILY',
#     'ME_CONSUMPTION': 'ME_CONSUMPTION',
#     'REPORT_DATE': 'REPORT_DATE'
# }

VALIDATION_THRESHOLDS = {
    'min_speed': 0,
    'low_speed_at_sea': 5,
    'min_maneuvering_speed': 2,
    'max_maneuvering_speed': 5,
    'max_container_speed': 35,
    'max_non_container_speed': 20,
    'speed_distance_time_lower': 0.9,
    'speed_distance_time_upper': 1.1
}

VESSEL_STATUSES = {
    'AT_SEA': 'NOON AT SEA',
    'MANEUVERING': 'END OF SEA PASSAGE',
    'IN_PORT': 'NOON AT PORT'
}

# Utility functions
def get_vessel_type(vessel_name):
    engine = get_db_engine()
    query = "SELECT vessel_type FROM vessel_particulars WHERE vessel_name = %s"
    result = pd.read_sql_query(query, engine, params=(vessel_name,))
    return result['vessel_type'].iloc[0] if not result.empty else 'unknown'

def is_value_in_range(value, min_val, max_val):
    return min_val <= value <= max_val if pd.notna(value) else False

def validate_speed(row, vessel_type_cache={}):
    failure_reasons = []
    
    vessel_status = row[COLUMN_NAMES['EVENT']]
    speed = row[COLUMN_NAMES['SPEED']]

    # Check if the event is not 'NOON AT SEA' and speed is missing or null
    if vessel_status != VESSEL_STATUSES['AT_SEA'] and pd.isna(speed):
        return []  # Ignore this row and return an empty list

    vessel_name = row[COLUMN_NAMES['VESSEL_NAME']]
    steaming_hours = row[COLUMN_NAMES['STEAMING_HOURS']]
    observed_distance = row[COLUMN_NAMES['OBSERVED_DISTANCE']]
    me_rpm = row[COLUMN_NAMES['ME_RPM']]
    me_run_hours = row[COLUMN_NAMES['ME_RUN_HOURS']]
    me_consumption = row[COLUMN_NAMES['ME_CONSUMPTION']]

    # Get vessel type (with caching)
    if vessel_name not in vessel_type_cache:
        vessel_type_cache[vessel_name] = get_vessel_type(vessel_name)
    vessel_type = vessel_type_cache[vessel_name]

    if pd.notna(speed):
        # Negative speed check
        if speed < VALIDATION_THRESHOLDS['min_speed']:
            failure_reasons.append("Observed Speed cannot be negative")
        
        # Status-specific checks
        if vessel_status == VESSEL_STATUSES['AT_SEA'] and speed <= VALIDATION_THRESHOLDS['low_speed_at_sea']:
            failure_reasons.append("Unusually low speed for sea passage")
        elif vessel_status == VESSEL_STATUSES['MANEUVERING'] and not is_value_in_range(speed, VALIDATION_THRESHOLDS['min_maneuvering_speed'], VALIDATION_THRESHOLDS['max_maneuvering_speed']):
            failure_reasons.append("Unusual speed for maneuvering")
        elif vessel_status == VESSEL_STATUSES['IN_PORT'] and speed != 0:
            failure_reasons.append("Speed should be zero when in port")
        
        # Vessel type specific speed checks
        if vessel_type == 'container' and speed > VALIDATION_THRESHOLDS['max_container_speed']:
            failure_reasons.append("Speed too high for container vessel")
        elif vessel_type != 'container' and speed > VALIDATION_THRESHOLDS['max_non_container_speed']:
            failure_reasons.append("Speed too high for non-container vessel")
        
        # Expected speed calculation and comparison
        if pd.notna(steaming_hours) and pd.notna(observed_distance):
            if steaming_hours == 0:
                if observed_distance != 0:
                    failure_reasons.append("Observed distance is non-zero but steaming hours is zero")
                expected_speed = 0
            else:
                expected_speed = observed_distance / steaming_hours
            
            if expected_speed != 0:  # Avoid division by zero
                speed_ratio = speed / expected_speed
                if not is_value_in_range(speed_ratio, VALIDATION_THRESHOLDS['speed_distance_time_lower'], VALIDATION_THRESHOLDS['speed_distance_time_upper']):
                    failure_reasons.append("Observed Speed not aligned with distance and time")
        
        # Consistency check
        if speed > 0:
            if me_rpm == 0 or me_run_hours == 0 or me_consumption == 0:
                failure_reasons.append("Inconsistent data: Speed > 0 but engine parameters indicate no movement")
    else:
        # Only add this failure reason if it's a 'NOON AT SEA' event
        if vessel_status == VESSEL_STATUSES['AT_SEA']:
            failure_reasons.append("Speed data is missing for NOON AT SEA event")

    return failure_reasons

def fetch_speed_data(date_filter):
    engine = get_db_engine()
    query = """
    SELECT *
    FROM vessel_performance_summary
    WHERE reportdate >= %s;
    """
    data = pd.read_sql_query(query, engine, params=(date_filter,))
    print("Fetched Data: \n", data.head())  # Print first few rows of data
    print("Columns: ", data.columns)  # Print the columns to check
    return data
