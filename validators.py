from config import VALIDATION_THRESHOLDS, COLUMN_NAMES

def validate_me_consumption(row, vessel_type):
    failure_reasons = []
    me_consumption = row[COLUMN_NAMES['ME_CONSUMPTION']]
    me_power = row[COLUMN_NAMES['ME_POWER']]
    me_rpm = row[COLUMN_NAMES['ME_RPM']]
    run_hours = row[COLUMN_NAMES['RUN_HOURS']]

    if me_consumption < VALIDATION_THRESHOLDS['me_consumption']['min'] or me_consumption > VALIDATION_THRESHOLDS['me_consumption']['max']:
        failure_reasons.append("ME Consumption out of range")
    
    if me_consumption >= (VALIDATION_THRESHOLDS['me_consumption']['power_factor'] * me_power * run_hours / 10**6):
        failure_reasons.append("ME Consumption too high for the Reported power")
    
    if me_rpm > 0 and me_consumption == 0:
        failure_reasons.append("ME Consumption cannot be zero when underway")
    
    if vessel_type == "CONTAINER" and me_consumption > VALIDATION_THRESHOLDS['me_consumption']['container_max']:
        failure_reasons.append("ME Consumption too high for container vessel")
    elif vessel_type != "CONTAINER" and me_consumption > VALIDATION_THRESHOLDS['me_consumption']['non_container_max']:
        failure_reasons.append("ME Consumption too high for non-container vessel")

    return failure_reasons

def calculate_avg_consumption(vessel_df, load_type):
    relevant_data = vessel_df[vessel_df[COLUMN_NAMES['LOAD_TYPE']] == load_type].dropna(subset=[COLUMN_NAMES['ME_CONSUMPTION']])
    relevant_data = relevant_data.sort_values(by=COLUMN_NAMES['REPORT_DATE']).tail(30)

    if len(relevant_data) >= 10:  # Only calculate if we have at least 10 data points
        total_consumption = relevant_data[COLUMN_NAMES['ME_CONSUMPTION']].sum()
        total_steaming_time = relevant_data[COLUMN_NAMES['RUN_HOURS']].sum()
        if total_steaming_time > 0:
            return total_consumption / total_steaming_time
    return None

def calculate_expected_consumption(coefficients, speed, displacement, hull_performance_factor):
    base_consumption = (coefficients['consp_speed1'] * speed +
                        coefficients['consp_disp1'] * displacement +
                        coefficients['consp_speed2'] * speed**2 +
                        coefficients['consp_disp2'] * displacement**2 +
                        coefficients['consp_intercept'])
    return base_consumption * hull_performance_factor

def run_all_validations(df, coefficients_df, hull_performance_df):
    validation_results = []
    
    for vessel_name, vessel_data in df.groupby(COLUMN_NAMES['VESSEL_NAME']):
        vessel_type = vessel_data[COLUMN_NAMES['VESSEL_TYPE']].iloc[0]
        vessel_coefficients = coefficients_df[coefficients_df[COLUMN_NAMES['VESSEL_NAME']] == vessel_name].iloc[0] if not coefficients_df[coefficients_df[COLUMN_NAMES['VESSEL_NAME']] == vessel_name].empty else None
        
        hull_performance = hull_performance_df[hull_performance_df[COLUMN_NAMES['VESSEL_NAME']] == vessel_name][COLUMN_NAMES['HULL_PERFORMANCE']].iloc[0] if not hull_performance_df[hull_performance_df[COLUMN_NAMES['VESSEL_NAME']] == vessel_name].empty else 0
        hull_performance_factor = 1 + (hull_performance / 100)
        
        for _, row in vessel_data.iterrows():
            failure_reasons = validate_me_consumption(row, vessel_type)
            
            # Historical data comparison
            avg_consumption = calculate_avg_consumption(vessel_data, row[COLUMN_NAMES['LOAD_TYPE']])
            if avg_consumption is not None:
                if not (VALIDATION_THRESHOLDS['me_consumption']['historical_lower'] * avg_consumption <= row[COLUMN_NAMES['ME_CONSUMPTION']] <= VALIDATION_THRESHOLDS['me_consumption']['historical_upper'] * avg_consumption):
                    failure_reasons.append(f"ME Consumption outside typical range of {row[COLUMN_NAMES['LOAD_TYPE']]} condition")

            # Expected consumption validation with hull performance
            if vessel_coefficients is not None and row[COLUMN_NAMES['STREAMING_HOURS']] > 0:
                expected_consumption = calculate_expected_consumption(
                    vessel_coefficients, 
                    row[COLUMN_NAMES['CURRENT_SPEED']], 
                    row[COLUMN_NAMES['DISPLACEMENT']], 
                    hull_performance_factor
                )
                if not (VALIDATION_THRESHOLDS['me_consumption']['expected_lower'] * expected_consumption <= row[COLUMN_NAMES['ME_CONSUMPTION']] <= VALIDATION_THRESHOLDS['me_consumption']['expected_upper'] * expected_consumption):
                    failure_reasons.append("ME Consumption not aligned with speed consumption table (including hull performance)")

            if failure_reasons:
                validation_results.append({
                    'Vessel Name': vessel_name,
                    'Report Date': row[COLUMN_NAMES['REPORT_DATE']],
                    'Remarks': ", ".join(failure_reasons)
                })
    
    return validation_results
