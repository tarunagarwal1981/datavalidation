import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler
from sqlalchemy.exc import SQLAlchemyError
from scipy.stats import ks_2samp, chisquare
import ruptures as rpt
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from database import get_db_engine
import streamlit as st

# Configuration for column names
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

# Fetching vessel data from sf_consumption_logs
@st.cache_data
def load_data(_vessel_name, _date_filter):
    engine = get_db_engine()  # Initialize the engine inside the function (not cached)
    try:
        query = f"""
        SELECT {', '.join(f'"{col}"' for col in COLUMN_NAMES.values())}
        FROM sf_consumption_logs
        WHERE "{COLUMN_NAMES['VESSEL_NAME']}" = %s
        AND "{COLUMN_NAMES['REPORT_DATE']}" >= %s
        ORDER BY "{COLUMN_NAMES['REPORT_DATE']}"
        """
        df = pd.read_sql_query(query, engine, params=(_vessel_name, _date_filter))
        df[COLUMN_NAMES['REPORT_DATE']] = pd.to_datetime(df[COLUMN_NAMES['REPORT_DATE']])
        return df
    except SQLAlchemyError as e:
        st.error(f"Error fetching data for {_vessel_name}: {str(e)}")
        return pd.DataFrame()

# Preprocess data: handle missing values and scale numeric columns
def preprocess_data(df):
    # Ensure only numeric columns are scaled, and non-numeric columns are ignored or encoded
    numeric_columns = [
        COLUMN_NAMES['ME_CONSUMPTION'], COLUMN_NAMES['OBSERVERD_DISTANCE'], 
        COLUMN_NAMES['SPEED'], COLUMN_NAMES['DISPLACEMENT'], 
        COLUMN_NAMES['STEAMING_TIME_HRS'], COLUMN_NAMES['WINDFORCE']
    ]
    
    # Ensure columns are numeric and handle invalid entries
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert invalid numbers to NaN

    # Fill missing numeric values with column medians
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
    
    # Encoding categorical variables
    df[COLUMN_NAMES['VESSEL_ACTIVITY']] = pd.Categorical(df[COLUMN_NAMES['VESSEL_ACTIVITY']]).codes
    df[COLUMN_NAMES['LOAD_TYPE']] = pd.Categorical(df[COLUMN_NAMES['LOAD_TYPE']]).codes

    # Scale the numeric columns
    scaler = RobustScaler()
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    
    return df

# Detect anomalies using IsolationForest and LocalOutlierFactor
def detect_anomalies(df):
    features = [
        COLUMN_NAMES['ME_CONSUMPTION'], COLUMN_NAMES['OBSERVERD_DISTANCE'], COLUMN_NAMES['SPEED'], 
        COLUMN_NAMES['DISPLACEMENT'], COLUMN_NAMES['STEAMING_TIME_HRS'], COLUMN_NAMES['WINDFORCE'], 
        COLUMN_NAMES['VESSEL_ACTIVITY'], COLUMN_NAMES['LOAD_TYPE']
    ]
    
    # Ensure only numeric features are used
    df[features] = df[features].apply(pd.to_numeric, errors='coerce')
    df = df.dropna()  # Drop any rows that still have NaN values after conversion
    
    scaled_features = df[features]  # Ensure only numeric data goes to the scaler
    
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    
    lof_anomalies = lof.fit_predict(scaled_features)
    iso_forest_anomalies = iso_forest.fit_predict(scaled_features)
    
    combined_anomalies = (lof_anomalies == -1).astype(int) + (iso_forest_anomalies == -1).astype(int)
    anomalies = df[combined_anomalies > 1]
    
    return anomalies

# Detect drift in continuous and categorical features using statistical tests
def detect_drift(train_df, test_df):
    continuous_features = [
        COLUMN_NAMES['ME_CONSUMPTION'], COLUMN_NAMES['OBSERVERD_DISTANCE'], COLUMN_NAMES['SPEED'], 
        COLUMN_NAMES['DISPLACEMENT'], COLUMN_NAMES['STEAMING_TIME_HRS'], COLUMN_NAMES['WINDFORCE']
    ]
    categorical_features = [COLUMN_NAMES['VESSEL_ACTIVITY'], COLUMN_NAMES['LOAD_TYPE']]
    
    drift_detected = {}
    
    # KS test for continuous features
    for feature in continuous_features:
        ks_stat, p_value = ks_2samp(train_df[feature], test_df[feature])
        drift_detected[feature] = p_value < 0.05  # Drift if p-value is below threshold
    
    # Chi-Square test for categorical features
    for feature in categorical_features:
        chi_stat, p_value = chisquare(train_df[feature].value_counts(), test_df[feature].value_counts())
        drift_detected[feature] = p_value < 0.05
    
    return drift_detected

# Detect change points in time series data using the Pelt algorithm
def detect_change_points(df):
    features = [COLUMN_NAMES['ME_CONSUMPTION'], COLUMN_NAMES['OBSERVERD_DISTANCE'], COLUMN_NAMES['SPEED']]
    change_points = {}
    
    for feature in features:
        algo = rpt.Pelt(model="rbf").fit(df[feature].values)
        penalty = np.std(df[feature].values)
        change_points[feature] = algo.predict(pen=penalty)
    
    return change_points

# Validate relationships using mutual information for continuous and categorical variables
def validate_relationships(df):
    continuous_features = [COLUMN_NAMES['ME_CONSUMPTION'], COLUMN_NAMES['SPEED'], COLUMN_NAMES['DISPLACEMENT'], COLUMN_NAMES['STEAMING_TIME_HRS']]
    mutual_info = mutual_info_regression(df[continuous_features], df[COLUMN_NAMES['ME_CONSUMPTION']])
    
    relationships = {}
    for i, feature in enumerate(continuous_features[1:]):
        relationships[feature] = mutual_info[i]
    
    # Mutual information for categorical features
    categorical_features = [COLUMN_NAMES['VESSEL_ACTIVITY'], COLUMN_NAMES['LOAD_TYPE']]
    cat_mutual_info = mutual_info_classif(df[categorical_features], df[COLUMN_NAMES['ME_CONSUMPTION']])
    
    for i, feature in enumerate(categorical_features):
        relationships[feature] = cat_mutual_info[i]
    
    return relationships

# Main function to run advanced validations on the vessel data
def run_advanced_validation(engine, vessel_name, date_filter):
    df = load_data(vessel_name, date_filter)
    
    if df.empty:
        raise ValueError(f"No data available for vessel {vessel_name}")
    
    df_processed = preprocess_data(df)
    
    # Split data into train and test sets
    train_df = df_processed[df_processed[COLUMN_NAMES['REPORT_DATE']] < df_processed[COLUMN_NAMES['REPORT_DATE']].max() - pd.Timedelta(days=90)]
    test_df = df_processed[df_processed[COLUMN_NAMES['REPORT_DATE']] >= df_processed[COLUMN_NAMES['REPORT_DATE']].max() - pd.Timedelta(days=90)]
    
    results = {
        'anomalies': detect_anomalies(test_df),
        'drift': detect_drift(train_df, test_df),
        'change_points': detect_change_points(test_df),
        'relationships': validate_relationships(test_df)
    }
    
    return results


# Example of how to use the advanced validation functionality
if __name__ == "__main__":
    from datetime import datetime, timedelta
    from database import get_db_engine  # Import the function to get the engine

    engine = get_db_engine()  # Get the database engine
    date_filter = datetime.now() - timedelta(days=180)  # Last 6 months
    vessel_name = "Example_Vessel"  # Replace with actual vessel name

    try:
        results = run_advanced_validation(engine, vessel_name, date_filter)  # Pass engine as first argument
        print(pd.DataFrame(results))
    except ValueError as e:
        print(str(e))
