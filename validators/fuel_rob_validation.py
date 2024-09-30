import pandas as pd
import numpy as np

def validate_fuel_rob(df):
    """
    Validate Fuel Remaining On Board (ROB) for different fuel types.
    
    Args:
    df (pd.DataFrame): Dataframe containing sf_consumption_logs data for a single vessel,
                       sorted in ascending order by date.
    
    Returns:
    list: List of dictionaries containing validation results for each row.
    """
    fuel_types = ['HSFO', 'LSMGO', 'ULSFO', 'VLSFO', 'MDO', 'LNG']
    validation_results = []

    # Replace NaN with 0
    df = df.fillna(0)

    for index, row in df.iterrows():
        if index == 0:  # Skip the first row
            continue

        prev_row = df.iloc[index - 1]
        failure_reasons = []

        for fuel_type in fuel_types:
            if fuel_type == 'ULSFO':
                current_rob = row[f'ROB_{fuel_type}']
                prev_rob = prev_row[f'ROB_{fuel_type}']
                bunkered_qty = row[f'BUNKERED_QTY_VLSFO']
                total_consumption = row[f'TOTAL_CONSUMPTION_MDO']
            elif fuel_type == 'VLSFO':
                current_rob = row[f'ROB_{fuel_type}']
                prev_rob = prev_row[f'ROB_{fuel_type}']
                bunkered_qty = row[f'BUNKERED_QTY_ULSFO']
                total_consumption = row[f'TOTAL_CONSUMPTION_ULSFO']
            elif fuel_type == 'MDO':
                current_rob = row[f'ROB_{fuel_type}']
                prev_rob = prev_row[f'ROB_{fuel_type}']
                bunkered_qty = row[f'BUNKERED_QTY_MDO']
                total_consumption = row[f'TOTAL_CONSUMPTION_VLSFO']
            else:
                current_rob = row[f'ROB_{fuel_type}']
                prev_rob = prev_row[f'ROB_{fuel_type}']
                bunkered_qty = row[f'BUNKERED_QTY_{fuel_type}']
                total_consumption = row[f'TOTAL_CONSUMPTION_{fuel_type}']

            calculated_rob = prev_rob + bunkered_qty - total_consumption

            if not np.isclose(current_rob, calculated_rob):
                failure_reasons.append(
                    f"{fuel_type} ROB validation failed. "
                    f"Calculated: {calculated_rob:.2f}, "
                    f"Actual: {current_rob:.2f}, "
                    f"Difference: {abs(current_rob - calculated_rob):.2f}"
                )

        if failure_reasons:
            validation_results.append({
                'Report Date': row['reportdate'],
                'Remarks': "; ".join(failure_reasons)
            })

    return validation_results

def validate_fuel_rob_for_vessel(df, vessel_name):
    """
    Validate Fuel ROB for a specific vessel.
    
    Args:
    df (pd.DataFrame): Dataframe containing sf_consumption_logs data for all vessels.
    vessel_name (str): Name of the vessel to validate.
    
    Returns:
    list: List of dictionaries containing validation results for the specified vessel.
    """
    vessel_df = df[df['vessel_name'] == vessel_name].sort_values('reportdate')
    return validate_fuel_rob(vessel_df)
