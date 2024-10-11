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
    validation_results = []
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
        return {
            'validation_results': validation_results,
            'anomalies': pd.DataFrame(),
            'drift': {},
            'change_points': {},
            'relationships': {}
        }

    # Anomaly Detection using Isolation Forest and LOF
    anomalies = detect_anomalies(test_df)

    # Drift Detection using KS Test
    drift = detect_drift(train_df, test_df)

    # Change Point Detection using Ruptures
    change_points = detect_change_points(test_df)

    # Feature Relationships using Mutual Information
    relationships = validate_relationships(train_df)

    # Ensure that the return value is a dictionary with all expected keys
    results = {
        'validation_results': validation_results,
        'anomalies': anomalies if not anomalies.empty else pd.DataFrame(),
        'drift': drift if drift else {},
        'change_points': change_points if change_points else {},
        'relationships': relationships if relationships else {}
    }

    for index, row in anomalies.iterrows():
        validation_results.append({
            'Vessel Name': str(vessel_name),
            'Anomaly Name': 'Anomaly Detected',
            'Feature': {k: str(v) for k, v in row.to_dict().items()}  # Convert all values to strings to avoid type issues
        })
    for feature, has_drift in drift.items():
        if has_drift:
            validation_results.append({
                'Vessel Name': str(vessel_name),
                'Anomaly Name': 'Drift Detected',
                'Feature': str(feature)  # Convert feature to string
            })
    for feature, points in change_points.items():
        if points:
            validation_results.append({
                'Vessel Name': str(vessel_name),
                'Anomaly Name': 'Change Point Detected',
                'Feature': str(feature),
                'Value': str(points)  # Convert points to string to avoid type issues
            })

    return results

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
