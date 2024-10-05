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

    # Anomaly Detection using Isolation Forest and LOF
    anomalies = detect_anomalies(test_df)

    # Drift Detection using KS Test
    drift = detect_drift(train_df, test_df)

    # Change Point Detection using Ruptures
    change_points = detect_change_points(test_df)

    # Feature Relationships using Mutual Information
    relationships = validate_relationships(train_df)

    for index, row in anomalies.iterrows():
        validation_results.append({
            'Vessel Name': vessel_name,
            'Anomaly Name': 'Anomaly Detected',
            'Feature': row.to_dict()
        })
    for feature, has_drift in drift.items():
        if has_drift:
            validation_results.append({
                'Vessel Name': vessel_name,
                'Anomaly Name': 'Drift Detected',
                'Feature': feature
            })
    for feature, points in change_points.items():
        if points:
            validation_results.append({
                'Vessel Name': vessel_name,
                'Anomaly Name': 'Change Point Detected',
                'Feature': feature,
                'Value': points
            })
    return pd.DataFrame(validation_results)

def detect_anomalies(df, n_neighbors=20):
    # Handle missing values by dropping rows with NaN values
    df = df.dropna()
    if df.shape[0] == 0:
        return pd.DataFrame()  # Return an empty DataFrame if no rows are left after dropping NaNs
    # Handle missing values by dropping rows with NaN values
    df = df.dropna()
    features = [
        COLUMN_NAMES['ME_CONSUMPTION'], COLUMN_NAMES['OBSERVERD_DISTANCE'], COLUMN_NAMES['SPEED'],
        COLUMN_NAMES['DISPLACEMENT'], COLUMN_NAMES['STEAMING_TIME_HRS'], COLUMN_NAMES['WINDFORCE'],
        COLUMN_NAMES['VESSEL_ACTIVITY'], COLUMN_NAMES['LOAD_TYPE']
    ]

    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=0.1)
    if df[features].shape[0] > 0:
        lof_anomalies = lof.fit_predict(df[features])
    else:
        lof_anomalies = []

    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    if df[features].shape[0] > 0:
        iso_forest_anomalies = iso_forest.fit_predict(df[features])
    else:
        iso_forest_anomalies = []

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

    # Discretize the target feature if necessary to handle non-continuous data
    discretizer = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')
    target = discretizer.fit_transform(df[[COLUMN_NAMES['ME_CONSUMPTION']]]).ravel()

    mutual_info = mutual_info_regression(df[features[1:]], target)  # Use only predictor features for mutual information

    relationships = {}
    for i, feature in enumerate(features[1:]):  # Skip ME_CONSUMPTION itself
        relationships[feature] = mutual_info[i]

    return relationships

def preprocess_data(df):
    # Handle missing values by imputing or dropping
    df = df.dropna(how='all')  # Drop rows where all values are NaN to avoid empty DataFrames  # Return an empty DataFrame if any column is completely NaN
    df = df.fillna(df.mean())  # Impute missing values with column mean to avoid errors
    # Handle missing values by imputing or dropping
    df = df.dropna(subset=[
        COLUMN_NAMES['ME_CONSUMPTION'], COLUMN_NAMES['OBSERVERD_DISTANCE'], COLUMN_NAMES['SPEED'],
        COLUMN_NAMES['DISPLACEMENT'], COLUMN_NAMES['STEAMING_TIME_HRS'], COLUMN_NAMES['WINDFORCE'],
        COLUMN_NAMES['VESSEL_ACTIVITY'], COLUMN_NAMES['LOAD_TYPE']
    ])
    df[COLUMN_NAMES['VESSEL_ACTIVITY']] = pd.Categorical(df[COLUMN_NAMES['VESSEL_ACTIVITY']]).codes
    df[COLUMN_NAMES['LOAD_TYPE']] = pd.Categorical(df[COLUMN_NAMES['LOAD_TYPE']]).codes

    numeric_columns = [
        COLUMN_NAMES['ME_CONSUMPTION'], COLUMN_NAMES['OBSERVERD_DISTANCE'], COLUMN_NAMES['SPEED'],
        COLUMN_NAMES['DISPLACEMENT'], COLUMN_NAMES['STEAMING_TIME_HRS'], COLUMN_NAMES['WINDFORCE']
    ]
    scaler = RobustScaler()
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

    return df
