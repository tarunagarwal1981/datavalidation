import pandas as pd
import math
from database import get_db_engine
from sqlalchemy.exc import SQLAlchemyError

# Configuration
COLUMN_NAMES = {
    'VESSEL_NAME': 'VESSEL_NAME',
    'REPORT_DATE': 'REPORT_DATE',
    'LATITUDE': 'LATITUDE',
    'LONGITUDE': 'LONGITUDE',
    'OBSERVED_DISTANCE': 'OBSERVERD_DISTANCE',  # Note: This is the correct spelling from your schema
    'STEAMING_TIME_HRS': 'STEAMING_TIME_HRS'
}

VALIDATION_THRESHOLDS = {
    'container_max_distance': 600,
    'non_container_max_distance': 500,
    'distance_alignment_lower': 0.9,
    'distance_alignment_upper': 1.1
}

def fetch_validation_data():
    engine = get_db_engine()
    try:
        query = f"""
        SELECT *
        FROM sf_consumption_logs
        ORDER BY "VESSEL_NAME", "REPORT_DATE"
        """
        return pd.read_sql_query(query, engine)
    except SQLAlchemyError as e:
        print(f"Error fetching validation data: {str(e)}")
        return pd.DataFrame()

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
    
    # Check if required columns exist
    required_columns = [COLUMN_NAMES['OBSERVED_DISTANCE'], COLUMN_NAMES['STEAMING_TIME_HRS'], 
                        COLUMN_NAMES['LATITUDE'], COLUMN_NAMES['LONGITUDE']]
    missing_columns = [col for col in required_columns if col not in row.index]
    if missing_columns:
        return [f"Missing columns: {', '.join(missing_columns)}"]

    observed_distance = row[COLUMN_NAMES['OBSERVED_DISTANCE']]
    steaming_time_hrs = row[COLUMN_NAMES['STEAMING_TIME_HRS']]

    if pd.isna(observed_distance):
        failure_reasons.append("Observed Distance data is missing")
        return failure_reasons

    # Check for negative distance
    if observed_distance < 0:
        failure_reasons.append("Observed Distance cannot be negative")

    # Check for maximum distance (assuming all vessels have the same limit for simplicity)
    if observed_distance > VALIDATION_THRESHOLDS['non_container_max_distance']:
        failure_reasons.append("Observed Distance too high")

    # Check alignment with steaming hours
    if pd.notna(steaming_time_hrs) and steaming_time_hrs > 0:
        if observed_distance == 0:
            failure_reasons.append("Observed Distance cannot be zero when streaming")
        else:
            expected_distance = steaming_time_hrs * 1  # Assuming average speed of 1 for now
            if not (VALIDATION_THRESHOLDS['distance_alignment_lower'] * expected_distance <= observed_distance <= VALIDATION_THRESHOLDS['distance_alignment_upper'] * expected_distance):
                failure_reasons.append("Observed Distance not aligned with steaming hours")

    # Calculate distance between current and previous positions
    if prev_row is not None:
        try:
            calculated_distance = chord_distance(
                row[COLUMN_NAMES['LATITUDE']], row[COLUMN_NAMES['LONGITUDE']],
                prev_row[COLUMN_NAMES['LATITUDE']], prev_row[COLUMN_NAMES['LONGITUDE']]
            )
            if observed_distance < calculated_distance:
                failure_reasons.append("Observed Distance less than calculated distance between positions")
        except KeyError as e:
            failure_reasons.append(f"Error calculating distance: missing column {str(e)}")

    return failure_reasons

def validate_distance_data():
    df = fetch_validation_data()
    validation_results = []

    if df.empty:
        return pd.DataFrame(columns=['Vessel Name', 'Report Date', 'Remarks'])

    for vessel_name, vessel_data in df.groupby('VESSEL_NAME'):
        prev_row = None
        for _, row in vessel_data.iterrows():
            failure_reasons = validate_distance(row, prev_row)
            if failure_reasons:
                validation_results.append({
                    'Vessel Name': vessel_name,
                    'Report Date': row['REPORT_DATE'],
                    'Remarks': ", ".join(failure_reasons)
                })
            prev_row = row

    return pd.DataFrame(validation_results)

if __name__ == "__main__":
    results = validate_distance_data()
    print(results)
