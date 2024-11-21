import pandas as pd
from database import get_db_engine

# Configuration
COLUMN_NAMES = {
    'SLIP_PCT': 'slip_pct',
    'VESSEL_NAME': 'vessel_name',
    'EVENT': 'event',
    'OBSERVED_DISTANCE': 'observed_distance',
    'ENGINE_DISTANCE': 'engine_distance',
    'REPORT_DATE': 'reportdate'
}

VALIDATION_THRESHOLDS = {
    'slip_min': -50,
    'slip_max': 50,
    'slip_warning': 30
}

VESSEL_STATUSES = {
    'AT_SEA': 'NOON AT SEA',
    'MANEUVERING': 'END OF SEA PASSAGE',
    'IN_PORT': 'NOON AT PORT'
}

# Utility Functions
def get_db_engine():
    """
    Placeholder for database connection engine.
    Replace with actual implementation.
    """
    # Add your actual database connection logic here.
    pass

def is_value_in_range(value, min_val, max_val):
    """
    Checks if a value is within a specified range.

    Args:
        value (float): The value to check.
        min_val (float): Minimum acceptable value.
        max_val (float): Maximum acceptable value.

    Returns:
        bool: True if the value is within range, False otherwise.
    """
    return min_val <= value <= max_val if pd.notna(value) else False

def fetch_slip_data(date_filter):
    """
    Fetch data required for Slip % validation from the database.

    Args:
        date_filter (str): A date filter for the query.

    Returns:
        pd.DataFrame: A DataFrame containing the required data.
    """
    engine = get_db_engine()
    query = """
    SELECT 
        reportdate,
        vessel_name,
        slip_pct,
        event,
        observed_distance,
        engine_distance
    FROM vessel_performance_summary
    WHERE reportdate >= %s;
    """
    data = pd.read_sql_query(query, engine, params=(date_filter,))
    print("Fetched Slip Data: \n", data.head())  # Debugging output
    return data

def validate_slip_percentage(row):
    """
    Validates the slip_pct field for given conditions.

    Args:
        row (pd.Series): A row of data containing slip_pct and related fields.

    Returns:
        list: A list of validation warnings or errors.
    """
    failure_reasons = []
    slip_pct = row.get(COLUMN_NAMES['SLIP_PCT'])
    vessel_status = row.get(COLUMN_NAMES['EVENT'])
    observed_distance = row.get(COLUMN_NAMES['OBSERVED_DISTANCE'])
    engine_distance = row.get(COLUMN_NAMES['ENGINE_DISTANCE'])

    # Validate Slip %
    if pd.notna(slip_pct):
        # Check if Slip % is within valid range (-50 to 50)
        if not is_value_in_range(slip_pct, VALIDATION_THRESHOLDS['slip_min'], VALIDATION_THRESHOLDS['slip_max']):
            failure_reasons.append("ValidationError: Slip percentage out of typical range (-50 to 50)")

        # Ensure Slip % is only calculated during 'NOON AT SEA'
        if vessel_status != VESSEL_STATUSES['AT_SEA']:
            failure_reasons.append("ValidationError: Slip should only be calculated during sea passage")

        # Raise a warning if Slip % > 30
        if slip_pct > VALIDATION_THRESHOLDS['slip_warning']:
            failure_reasons.append("Warning: High slip percentage. Check Observed Distance and Engine Distance")
    else:
        failure_reasons.append("ValidationError: Slip percentage is missing")

    # Validate consistency between observed and engine distances
    if pd.notna(observed_distance) and pd.notna(engine_distance):
        if observed_distance <= 0 or engine_distance <= 0:
            failure_reasons.append("ValidationError: Observed or Engine Distance should be greater than zero")
        elif engine_distance < observed_distance:
            failure_reasons.append("ValidationError: Engine Distance is less than Observed Distance")

    return failure_reasons

def validate_slip_data(date_filter):
    """
    Fetches and validates Slip % data for all records after the specified date filter.

    Args:
        date_filter (str): A date filter for the query.

    Returns:
        list: A list of validation logs containing errors and warnings.
    """
    failure_logs = []
    data = fetch_slip_data(date_filter)

    for _, row in data.iterrows():
        slip_failures = validate_slip_percentage(row)
        if slip_failures:
            failure_logs.append({
                'report_date': row[COLUMN_NAMES['REPORT_DATE']],
                'vessel_name': row[COLUMN_NAMES['VESSEL_NAME']],
                'failures': slip_failures
            })

    return failure_logs

if __name__ == "__main__":
    # Example usage
    date_filter = "2024-01-01"  # Replace with the desired date filter
    validation_results = validate_slip_data(date_filter)

    for result in validation_results:
        print(f"Date: {result['report_date']}, Vessel: {result['vessel_name']}, Issues: {result['failures']}")
