import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import KBinsDiscretizer
from scipy.stats import ks_2samp
import ruptures as rpt
from database import get_db_engine

COLUMN_NAMES = {
    'VESSEL_NAME': 'VESSEL_NAME',
    'REPORT_DATE': 'REPORT_DATE',
    'ME_CONSUMPTION': 'ME_CONSUMPTION',
    'OBSERVERD_DISTANCE': 'OBSERVERD_DISTANCE',
    'SPEED': 'SPEED',
    'DISPLACEMENT': 'DISPLACEMENT',
    'STEAMING_TIME_HRS': 'STEAMING_TIME_HRS',
    'WINDFORCE': 'WINDFORCE',
    'VESSEL_ACTIVITY': 'VESSEL_ACTIVITY',
    'LOAD_TYPE': 'LOAD_TYPE'
}

def run_advanced_validation(engine, vessel_name, date_filter):
    # Fetch data for the vessel
    query = """
    SELECT * FROM sf_consumption_logs
    WHERE "{}" = %s AND "{}" >= %s;
    """.format(COLUMN_NAMES['VESSEL_NAME'], COLUMN_NAMES['REPORT_DATE'])
    df = pd.read_sql_query(query, engine, params=(vessel_name, date_filter))

    # Split data into training (first 6 months) and validation (last 6 months)
    df[COLUMN_NAMES['REPORT_DATE']] = pd.to_datetime(df[COLUMN_NAMES['REPORT_DATE']])
    df = df.sort_values(by=COLUMN_NAMES['REPORT_DATE'])
    mid_point = len(df) // 2
    train_df = df.iloc[:mid_point]
    test_df = df.iloc[mid_point:]

    # Preprocess training and validation data separately to avoid data leakage
    train_df = preprocess_data(train_df)
    test_df = preprocess_data(test_df)

    # If test_df is empty after preprocessing, return empty results
    if test_df.shape[0] == 0:
        return []

    # Anomaly Detection using Isolation Forest and LOF
    anomalies = detect_anomalies(test_df)

    # Drift Detection using KS Test
    drift = detect_drift(train_df, test_df)

    # Change Point Detection using Ruptures
    change_points = detect_change_points(test_df)

    # Feature Relationships using Mutual Information
    relationships = validate_relationships(train_df)

    # Format the results
    results = {
        'anomalies': anomalies,
        'drift': drift,
        'change_points': change_points,
        'relationships': relationships
    }
    
    formatted_results = format_validation_results(results, vessel_name)
    return formatted_results

def detect_anomalies(df, n_neighbors=20):
    # Handle missing values by dropping rows with NaN values
    df = df.dropna()
    if df.shape[0] == 0:
        return pd.DataFrame()  # Return an empty DataFrame if no rows are left after dropping NaNs

    features = [
        COLUMN_NAMES['ME_CONSUMPTION'], COLUMN_NAMES['OBSERVERD_DISTANCE'], COLUMN_NAMES['SPEED'],
        COLUMN_NAMES['DISPLACEMENT'], COLUMN_NAMES['STEAMING_TIME_HRS'], COLUMN_NAMES['WINDFORCE'],
        COLUMN_NAMES['VESSEL_ACTIVITY'], COLUMN_NAMES['LOAD_TYPE']
    ]

    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=0.1)
    lof_anomalies = lof.fit_predict(df[features]) if df[features].shape[0] > 0 else []

    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    iso_forest_anomalies = iso_forest.fit_predict(df[features]) if df[features].shape[0] > 0 else []

    anomalies = df[(lof_anomalies == -1) | (iso_forest_anomalies == -1)]
    return anomalies if not anomalies.empty else pd.DataFrame()

def detect_drift(train_df, test_df):
    features = [
        COLUMN_NAMES['ME_CONSUMPTION'], COLUMN_NAMES['OBSERVERD_DISTANCE'], COLUMN_NAMES['SPEED'],
        COLUMN_NAMES['DISPLACEMENT'], COLUMN_NAMES['STEAMING_TIME_HRS'], COLUMN_NAMES['WINDFORCE']
    ]

    drift_detected = {}
    for feature in features:
        ks_stat, p_value = ks_2samp(train_df[feature], test_df[feature])
        drift_detected[feature] = p_value < 0.05  # Drift if p-value is below threshold

    return drift_detected

def detect_change_points(df):
    features = [COLUMN_NAMES['ME_CONSUMPTION'], COLUMN_NAMES['OBSERVERD_DISTANCE'], COLUMN_NAMES['SPEED']]
    change_points = {}

    for feature in features:
        algo = rpt.Pelt(model="rbf").fit(df[feature].values)
        change_points[feature] = algo.predict(pen=1)

    return change_points

def validate_relationships(df):
    features = [
        COLUMN_NAMES['ME_CONSUMPTION'], COLUMN_NAMES['SPEED'], COLUMN_NAMES['DISPLACEMENT'],
        COLUMN_NAMES['STEAMING_TIME_HRS']
    ]

    # Check if the dataframe is empty before proceeding
    if df.shape[0] == 0:
        return {feature: 0.0 for feature in features[1:]}

    # Discretize the target feature if necessary to handle non-continuous data
    discretizer = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')
    try:
        target = discretizer.fit_transform(df[[COLUMN_NAMES['ME_CONSUMPTION']]]).ravel()
    except ValueError:
        return {feature: 0.0 for feature in features[1:]}  # Return default values if discretization fails

    mutual_info = mutual_info_regression(df[features[1:]], target)  # Use only predictor features for mutual information

    relationships = {}
    for i, feature in enumerate(features[1:]):  # Skip ME_CONSUMPTION itself
        relationships[feature] = mutual_info[i]

    return relationships

def preprocess_data(df):
    # Convert columns to appropriate data types
    df[COLUMN_NAMES['VESSEL_NAME']] = df[COLUMN_NAMES['VESSEL_NAME']].astype(str)
    df[COLUMN_NAMES['VESSEL_ACTIVITY']] = df[COLUMN_NAMES['VESSEL_ACTIVITY']].astype(str)
    df[COLUMN_NAMES['LOAD_TYPE']] = df[COLUMN_NAMES['LOAD_TYPE']].astype(str)
    df[COLUMN_NAMES['REPORT_DATE']] = pd.to_datetime(df[COLUMN_NAMES['REPORT_DATE']])
    
    # Handle missing values by imputing or dropping
    df = df.dropna(how='all')  # Drop rows where all values are NaN to avoid empty DataFrames
    df = df.fillna(df.select_dtypes(include=['number']).mean())  # Impute numeric columns only with column mean
    
    # Handle missing values by dropping rows with NaNs in critical columns
    df = df.dropna(subset=[
        COLUMN_NAMES['ME_CONSUMPTION'], COLUMN_NAMES['OBSERVERD_DISTANCE'], COLUMN_NAMES['SPEED'],
        COLUMN_NAMES['DISPLACEMENT'], COLUMN_NAMES['STEAMING_TIME_HRS'], COLUMN_NAMES['WINDFORCE'],
        COLUMN_NAMES['VESSEL_ACTIVITY'], COLUMN_NAMES['LOAD_TYPE']
    ])
    
    # If the dataframe is empty after dropping critical NaNs, return it as is
    if df.shape[0] == 0:
        return df

    # Convert categorical columns to numeric codes
    df[COLUMN_NAMES['VESSEL_ACTIVITY']] = pd.Categorical(df[COLUMN_NAMES['VESSEL_ACTIVITY']]).codes
    df[COLUMN_NAMES['LOAD_TYPE']] = pd.Categorical(df[COLUMN_NAMES['LOAD_TYPE']]).codes

    # Scale numeric columns
    numeric_columns = [
        COLUMN_NAMES['ME_CONSUMPTION'], COLUMN_NAMES['OBSERVERD_DISTANCE'], COLUMN_NAMES['SPEED'],
        COLUMN_NAMES['DISPLACEMENT'], COLUMN_NAMES['STEAMING_TIME_HRS'], COLUMN_NAMES['WINDFORCE']
    ]
    scaler = RobustScaler()
    if df[numeric_columns].shape[0] > 0:
        df.loc[:, numeric_columns] = scaler.fit_transform(df[numeric_columns])

    return df

def format_validation_results(results, vessel_name):
    formatted_results = []
    
    # Format anomalies
    for _, row in results['anomalies'].iterrows():
        anomalous_features = [k for k, v in row.items() if v == -1 and k != COLUMN_NAMES['REPORT_DATE']]
        if anomalous_features:
            formatted_results.append({
                'Vessel Name': vessel_name,
                'Date': row[COLUMN_NAMES['REPORT_DATE']].strftime('%Y-%m-%d'),
                'Issue': 'Unusual Data Detected',
                'Explanation': f"Unusual values were found in {', '.join(anomalous_features)}. This could indicate measurement errors or exceptional operating conditions."
            })
    
    # Format drift
    for feature, has_drift in results['drift'].items():
        if has_drift:
            formatted_results.append({
                'Vessel Name': vessel_name,
                'Date': 'Multiple Dates',
                'Issue': 'Data Trend Change Detected',
                'Explanation': f"The pattern of {feature} has changed significantly over time. This might indicate changes in operating conditions or equipment performance."
            })
    
    # Format change points
    for feature, points in results['change_points'].items():
        if points:
            formatted_results.append({
                'Vessel Name': vessel_name,
                'Date': 'Specific Dates',
                'Issue': 'Sudden Changes Detected',
                'Explanation': f"Sudden changes were detected in {feature}. This could indicate equipment changes, maintenance events, or changes in operating procedures."
            })
    
    # Format relationships
    weak_relationships = [f for f, v in results['relationships'].items() if v < 0.3]
    if weak_relationships:
        formatted_results.append({
            'Vessel Name': vessel_name,
            'Date': 'Overall Analysis',
            'Issue': 'Unexpected Data Relationships',
            'Explanation': f"The following factors show weaker than expected influence on fuel consumption: {', '.join(weak_relationships)}. This might indicate data quality issues or unusual operating conditions."
        })
    
    return formatted_results

# Example usage in streamlit_app.py
# if st.button('Validate Data'):
#     engine = get_db_engine()
#     try:
#         # ... (keep existing code for fetching data)
#         
#         validation_results = []
#         for vessel_name in df['vessel_name'].unique():
#             vessel_results = run_advanced_validation(engine, vessel_name, three_months_ago)
#             validation_results.extend(vessel_results)
#         
#         if validation_results:
#             result_df = pd.DataFrame(validation_results)
#             st.write("Validation Results:")
#             st.dataframe(result_df)
#             
#             csv = result_df.to_csv(index=False)
#             st.download_button(label="Download validation report as CSV", data=csv, file_name='validation_report.csv', mime='text/csv')
#         else:
#             st.write("All data passed the validation checks!")
#     
#     except Exception as e:
#         st.error(f"An error occurred: {str(e)}")
