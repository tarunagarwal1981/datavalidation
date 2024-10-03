import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler
from scipy.stats import ks_2samp, chisquare
import ruptures as rpt
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif

def load_data(engine, vessel_name):
    query = f"""
    SELECT * FROM sf_consumption_log
    WHERE VESSEL_NAME = '{vessel_name}'
    AND REPORT_DATE >= DATEADD(month, -6, CURRENT_DATE())
    ORDER BY REPORT_DATE
    """
    df = pd.read_sql(query, engine)
    df['REPORT_DATE'] = pd.to_datetime(df['REPORT_DATE'])
    return df

def preprocess_data(df):
    # Handling missing values
    df.fillna(df.median(), inplace=True)
    
    # Encoding categorical variables
    df['VESSEL_ACTIVITY'] = pd.Categorical(df['VESSEL_ACTIVITY']).codes
    df['LOAD_TYPE'] = pd.Categorical(df['LOAD_TYPE']).codes
    
    # Normalizing numeric columns with RobustScaler
    numeric_columns = ['ME_CONSUMPTION', 'OBSERVERD_DISTANCE', 'SPEED', 'DISPLACEMENT', 
                       'STEAMING_TIME_HRS', 'WINDFORCE']
    scaler = RobustScaler()
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    
    return df

def detect_anomalies(df):
    features = ['ME_CONSUMPTION', 'OBSERVERD_DISTANCE', 'SPEED', 'DISPLACEMENT', 
                'STEAMING_TIME_HRS', 'WINDFORCE', 'VESSEL_ACTIVITY', 'LOAD_TYPE']
    
    # Shared scaling/preprocessing step
    scaled_features = RobustScaler().fit_transform(df[features])
    
    # Dynamic contamination based on historical anomaly rates (example)
    historical_anomaly_rate = 0.05  # This should be calculated based on historical data
    contamination = max(0.01, min(historical_anomaly_rate, 0.1))
    
    lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    
    lof_anomalies = lof.fit_predict(scaled_features)
    iso_forest_anomalies = iso_forest.fit_predict(scaled_features)
    
    # Ensemble voting mechanism
    combined_anomalies = (lof_anomalies == -1).astype(int) + (iso_forest_anomalies == -1).astype(int)
    anomalies = df[combined_anomalies > 1]  # Consider it an anomaly if flagged by both models
    
    return anomalies

def detect_drift(train_df, test_df):
    continuous_features = ['ME_CONSUMPTION', 'OBSERVERD_DISTANCE', 'SPEED', 'DISPLACEMENT', 
                           'STEAMING_TIME_HRS', 'WINDFORCE']
    categorical_features = ['VESSEL_ACTIVITY', 'LOAD_TYPE']
    
    drift_detected = {}
    
    # KS test for continuous features
    for feature in continuous_features:
        ks_stat, p_value = ks_2samp(train_df[feature], test_df[feature])
        drift_detected[feature] = p_value < 0.05  # Drift if p-value is below threshold

    # Chi-Square test for categorical features
    for feature in categorical_features:
        chi_stat, p_value = chisquare(train_df[feature].value_counts(), test_df[feature].value_counts())
        drift_detected[feature] = p_value < 0.05  # Detect drift in categorical features
    
    return drift_detected

def detect_change_points(df):
    features = ['ME_CONSUMPTION', 'OBSERVERD_DISTANCE', 'SPEED']
    change_points = {}
    
    for feature in features:
        algo = rpt.Pelt(model="rbf").fit(df[feature].values)
        penalty = np.std(df[feature].values)  # Adaptive penalty based on data variance
        change_points[feature] = algo.predict(pen=penalty)
    
    return change_points

def validate_relationships(df):
    # Mutual information for continuous variables
    continuous_features = ['ME_CONSUMPTION', 'SPEED', 'DISPLACEMENT', 'STEAMING_TIME_HRS']
    mutual_info = mutual_info_regression(df[continuous_features], df['ME_CONSUMPTION'])
    
    relationships = {}
    for i, feature in enumerate(continuous_features[1:]):
        relationships[feature] = mutual_info[i]
    
    # Mutual information for categorical variables
    categorical_features = ['VESSEL_ACTIVITY', 'LOAD_TYPE']
    cat_mutual_info = mutual_info_classif(df[categorical_features], df['ME_CONSUMPTION'])
    
    for i, feature in enumerate(categorical_features):
        relationships[feature] = cat_mutual_info[i]
    
    return relationships

def run_advanced_validation(engine, vessel_name):
    df = load_data(engine, vessel_name)
    df_processed = preprocess_data(df)
    
    # Split data into train and test
    train_df = df_processed[df_processed['REPORT_DATE'] < df_processed['REPORT_DATE'].max() - pd.Timedelta(days=90)]
    test_df = df_processed[df_processed['REPORT_DATE'] >= df_processed['REPORT_DATE'].max() - pd.Timedelta(days=90)]
    
    results = {
        'anomalies': detect_anomalies(test_df),
        'drift': detect_drift(train_df, test_df),
        'change_points': detect_change_points(test_df),
        'relationships': validate_relationships(test_df)
    }
    
    return results
