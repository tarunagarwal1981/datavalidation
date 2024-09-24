import pandas as pd
import math
from database import get_db_engine

# Configuration
COLUMN_NAMES = {
    'OBSERVED_DISTANCE': 'observed_distance',
    'ME_RUNNING_HOURS': 'steaming_time_hrs',
    'STREAMING_HOURS': 'steaming_time_hrs',
    'VESSEL_NAME': 'vessel_name',
    'VESSEL_TYPE': 'vessel_type',
    'REPORT_DATE': 'reportdate',
    'LATITUDE': 'LATITUDE',
    'LONGITUDE': 'LONGITUDE'
}

VALIDATION_THRESHOLDS = {
    'container_max_distance': 600,
    'non_container_max_distance': 500,
    'distance_alignment_lower': 0.9,
    'distance_alignment_upper': 1.1
}

def fetch_vessel_performance_data():
    engine = get_db_engine()
    query = """
    SELECT vps.*, vp.vessel_type
    FROM vessel_performance_summary vps
    LEFT JOIN vessel_particulars vp ON vps.vessel_name = vp.vessel_name
    ORDER BY vps.vessel_name, vps.reportdate
    """
    return pd.read_sql_query(query, engine)

def fetch_position_data(vessel_name, report_date):
    engine = get_db_engine()
    query = f"""
    SELECT {COLUMN_NAMES['LATITUDE']}, {COLUMN_NAMES['LONGITUDE']}
    FROM sf_consumption_logs
    WHERE {COLUMN_NAMES['VESSEL_NAME']} = %s AND {COLUMN_NAMES['REPORT_DATE']} = %s
    ORDER BY {COLUMN_NAMES['REPORT_DATE']} DESC
    LIMIT 2
    """
    return pd.read_sql_query(query, engine, params=(vessel_name, report_date))

def chord_distance(lat1, lon1, lat2, lon2, radius=6371):
    phi1 = math.radians(lat1)
    lambda1 = math.radians(lon1)
    phi2 = math.radians(lat2)
    lambda2 = math.radians(lon2)

    x1 = radius * math.cos(phi1) * math.cos(lambda1)
    y1 = radius * math.cos(phi1) * math.sin(lambda1)
    z1 = radius * math.sin(phi1)

    x2 = radius * math.cos(phi2) * math.cos(lambda2)
    y2 = radius * math.cos(phi2) * math.sin(lambda2)
    z2 = radius * math.sin(phi2)

    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
    return distance

def validate_distance(row, prev_row):
    failure_reasons = []
    observed_distance = row[COLUMN_NAMES['OBSERVED_DISTANCE']]
    me_running_hours = row[COLUMN_NAMES['ME_RUNNING_HOURS']]
    streaming_hours = row[COLUMN_NAMES['STREAMING_HOURS']]
    vessel_type = row[COLUMN_NAMES['VESSEL_TYPE']]
    vessel_name = row[COLUMN_NAMES['VESSEL_NAME']]
    report_date = row[COLUMN_NAMES['REPORT_DATE']]

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

    # Check alignment with engine running hours
    if pd.notna(me_running_hours) and me_running_hours > 0:
        expected_distance = me_running_hours * 1  # Assuming average speed of 1 for now
        if not (VALIDATION_THRESHOLDS['distance_alignment_lower'] * expected_distance <= observed_distance <= VALIDATION_THRESHOLDS['distance_alignment_upper'] * expected_distance):
            failure_reasons.append("Observed Distance not aligned with engine running hours")

    # Check for streaming hours
    if pd.notna(streaming_hours) and streaming_hours > 0:
        if observed_distance == 0:
            failure_reasons.append("Observed Distance cannot be zero when streaming")
        else:
            expected_streaming_distance = streaming_hours * 1  # Assuming average speed of 1 for now
            if not (VALIDATION_THRESHOLDS['distance_alignment_lower'] * expected_streaming_distance <= observed_distance <= VALIDATION_THRESHOLDS['distance_alignment_upper'] * expected_streaming_distance):
                failure_reasons.append("Observed Distance not aligned with streaming hours")

    # Calculate distance between current and previous positions
    if prev_row is not None:
        position_data = fetch_position_data(vessel_name, report_date)
        if len(position_data) == 2:
            current_pos = position_data.iloc[0]
            prev_pos = position_data.iloc[1]
            calculated_distance = chord_distance(
                current_pos[COLUMN_NAMES['LATITUDE']], current_pos[COLUMN_NAMES['LONGITUDE']],
                prev_pos[COLUMN_NAMES['LATITUDE']], prev_pos[COLUMN_NAMES['LONGITUDE']]
            )
            if observed_distance < calculated_distance:
                failure_reasons.append("Observed Distance less than calculated distance between positions")

    return failure_reasons

def validate_distance_data():
    df = fetch_vessel_performance_data()
    validation_results = []

    for vessel_name, vessel_data in df.groupby(COLUMN_NAMES['VESSEL_NAME']):
        prev_row = None
        for _, row in vessel_data.iterrows():
            failure_reasons = validate_distance(row, prev_row)
            if failure_reasons:
                validation_results.append({
                    'Vessel Name': vessel_name,
                    'Report Date': row[COLUMN_NAMES['REPORT_DATE']],
                    'Remarks': ", ".join(failure_reasons)
                })
            prev_row = row

    return pd.DataFrame(validation_results)
