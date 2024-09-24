import pandas as pd

# Configuration
COLUMN_NAMES = {
    'BOILER_CONSUMPTION': 'boiler_consumption',
    'ME_POWER': 'actual_me_power',
    'EVENT': 'event',
    'VESSEL_NAME': 'vessel_name'
}

VALIDATION_THRESHOLDS = {
    'boiler_consumption_min': 0,
    'boiler_consumption_max': 100,
    'me_load_threshold': 40
}

EVENT_AT_SEA = 'NOON AT SEA'

# Utility functions
def calculate_me_load(me_power, mcr):
    if pd.notna(me_power) and pd.notna(mcr) and mcr != 0:
        try:
            return (float(me_power) * 100) / float(mcr)
        except ValueError:
            return None
    return None

def is_value_in_range(value, min_val, max_val):
    return min_val <= value <= max_val if pd.notna(value) else False

# Main validation function
def validate_boiler_consumption(row, mcr_value):
    failure_reasons = []
    boiler_consumption = row[COLUMN_NAMES['BOILER_CONSUMPTION']]
    me_power = row[COLUMN_NAMES['ME_POWER']]
    event = row[COLUMN_NAMES['EVENT']]

    if pd.notna(boiler_consumption):
        # Check range
        if not is_value_in_range(boiler_consumption, VALIDATION_THRESHOLDS['boiler_consumption_min'], VALIDATION_THRESHOLDS['boiler_consumption_max']):
            failure_reasons.append("Boiler Consumption out of range")

        # For now, we're considering Cargo Heating as 0
        cargo_heating = 0
        if boiler_consumption < cargo_heating:
            failure_reasons.append("Boiler Consumption cannot be less than Cargo Heating Consumption")

        # Check if vessel is at sea and ME Load is high
        if event == EVENT_AT_SEA and pd.notna(mcr_value):
            me_load = calculate_me_load(me_power, mcr_value)
            if me_load is not None and me_load > VALIDATION_THRESHOLDS['me_load_threshold'] and boiler_consumption > 0:
                failure_reasons.append("Warning: Boiler Consumption expected to be zero at high ME Load during sea passage")

    else:
        failure_reasons.append("Boiler Consumption data is missing")

    return failure_reasons
