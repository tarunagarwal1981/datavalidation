from .me_consumption_validation import validate_me_consumption
from .ae_consumption_validation import validate_ae_consumption
from config import COLUMN_NAMES

# Import other validation functions as you create them

def run_all_validations(df, coefficients_df, hull_performance_df):
    validation_results = []
    historical_ae_data = calculate_historical_ae_data(df)
    
    for vessel_name, vessel_data in df.groupby(COLUMN_NAMES['VESSEL_NAME']):
        vessel_type = vessel_data[COLUMN_NAMES['VESSEL_TYPE']].iloc[0]
        vessel_coefficients = coefficients_df[coefficients_df[COLUMN_NAMES['VESSEL_NAME']] == vessel_name].iloc[0] if not coefficients_df[coefficients_df[COLUMN_NAMES['VESSEL_NAME']] == vessel_name].empty else None
        
        hull_performance = hull_performance_df[hull_performance_df[COLUMN_NAMES['VESSEL_NAME']] == vessel_name][COLUMN_NAMES['HULL_PERFORMANCE']].iloc[0] if not hull_performance_df[hull_performance_df[COLUMN_NAMES['VESSEL_NAME']] == vessel_name].empty else 0
        hull_performance_factor = 1 + (hull_performance / 100)
        
        for _, row in vessel_data.iterrows():
            failure_reasons = []
            
            # ME consumption validation
            failure_reasons.extend(validate_me_consumption(row, vessel_type))
            
            # AE consumption validation
            historical_data = {'avg_ae_consumption': historical_ae_data.get(vessel_name)} if vessel_name in historical_ae_data else None
            failure_reasons.extend(validate_ae_consumption(row, historical_data))
            
            # Add more validations here as you create them...

            if failure_reasons:
                validation_results.append({
                    'Vessel Name': vessel_name,
                    'Report Date': row[COLUMN_NAMES['REPORT_DATE']],
                    'Remarks': ", ".join(failure_reasons)
                })
    
    return validation_results

def calculate_historical_ae_data(df):
    return df.groupby(COLUMN_NAMES['VESSEL_NAME']).apply(
        lambda x: x.sort_values(COLUMN_NAMES['REPORT_DATE']).tail(30)[COLUMN_NAMES['AE_CONSUMPTION']].mean()
    ).to_dict()
