import pandas as pd
import math

# Configuration
COLUMN_NAMES = {
    'OBSERVED_DISTANCE': 'observed_distance',
    'STEAMING_HOURS': 'steaming_time_hrs',
    'VESSEL_TYPE': 'vessel_type',
    'LATITUDE': 'LATITUDE',
    'LONGITUDE': 'LONGITUDE',
    'PREV_LATITUDE': 'prev_LATITUDE',
    'PREV_LONGITUDE': 'prev_LONGITUDE'
}

VALIDATION_THRESHOLDS = {
    'container_max_distance': 600,
    'non_container_max_distance': 500,
    'distance_tolerance_lower': 0.9,
    'distance_tolerance_upper': 1.1
}

def chord_distance(lat1, lon1, lat2, lon2, radius=6371):
    """
    Calculate the straight-line (chord) distance between two points on Earth's surface.
    """
    phi1, lambda1 = math.radians(lat1), math.radians(lon1)
    phi2, lambda2 = math.radians(lat2), math.radians(lon2)
    
    x1 = radius * math.cos(phi1) * math.cos(lambda1)
    y1 = radius * math.cos(phi1) * math.sin(lambda1)
    z1 = radius * math.sin(phi1)
    
    x2 = radius * math.cos(phi2) * math.cos(lambda2)
    y2 = radius * math.cos(phi2) * math.sin(lambda2)
    z2 = radius * math.sin(phi2)
    
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
    return distance

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

    # Calculate distance between current and previous positions
    if not pd.isna(row[COLUMN_NAMES['LATITUDE']]) and not pd.isna(row[COLUMN_NAMES['LONGITUDE']]) and \
       not pd.isna(row[COLUMN_NAMES['PREV_LATITUDE']]) and not pd.isna(row[COLUMN_NAMES['PREV_LONGITUDE']]):
        calculated_distance = chord_distance(
            row[COLUMN_NAMES['PREV_LATITUDE']], row[COLUMN_NAMES['PREV_LONGITUDE']],
            row[COLUMN_NAMES['LATITUDE']], row[COLUMN_NAMES['LONGITUDE']]
        )
        if observed_distance < calculated_distance:
            failure_reasons.append("Observed Distance less than calculated distance between positions")

    return failure_reasons
