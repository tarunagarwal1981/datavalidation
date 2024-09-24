import pandas as pd
from config import COLUMN_NAMES

def is_value_in_range(value, min_val, max_val):
    """Check if a value is within a specified range."""
    return min_val <= value <= max_val if pd.notna(value) else False

def calculate_historical_average(df, column_name, days=30):
    """Calculate the historical average for a specified column."""
    return df.groupby(COLUMN_NAMES['VESSEL_NAME']).apply(
        lambda x: x.sort_values(COLUMN_NAMES['REPORT_DATE']).tail(days)[column_name].mean()
    ).to_dict()

def is_value_within_percentage(value, reference, lower_percentage, upper_percentage):
    """Check if a value is within a specified percentage range of a reference value."""
    if pd.isna(value) or pd.isna(reference):
        return False
    lower_bound = reference * (1 - lower_percentage)
    upper_bound = reference * (1 + upper_percentage)
    return lower_bound <= value <= upper_bound

def calculate_power_based_consumption(power, run_hours, factor):
    """Calculate the maximum allowed consumption based on power and run hours."""
    return (factor / power) * run_hours / 10**6 if pd.notna(power) and pd.notna(run_hours) and power > 0 else None

def add_failure_reason(failure_reasons, reason):
    """Add a failure reason to the list if it's not already present."""
    if reason not in failure_reasons:
        failure_reasons.append(reason)

def validate_non_negative(value, field_name):
    """Validate that a value is non-negative."""
    if pd.notna(value) and value < 0:
        return f"{field_name} cannot be negative"
    return None
