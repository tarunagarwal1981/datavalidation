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
    # List of numeric columns for scaling
    numeric_columns = [
        COLUMN_NAMES['ME_CONSUMPTION'], COLUMN_NAMES['OBSERVERD_DISTANCE'], 
        COLUMN_NAMES['SPEED'], COLUMN_NAMES['DISPLACEMENT'], 
        COLUMN_NAMES['STEAMING_TIME_HRS'], COLUMN_NAMES['WINDFORCE']
    ]
    
    # Ensure columns are numeric and handle invalid entries
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert invalid numbers to NaN
    
    # Display missing values in a user-friendly format
    missing_values_info = df[numeric_columns].isna().sum().to_dict()
    st.write(f"Missing values per column: {missing_values_info}")
    
    # Fill missing numeric values with column medians
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
    
    # Encoding categorical variables
    df[COLUMN_NAMES['VESSEL_ACTIVITY']] = pd.Categorical(df[COLUMN_NAMES['VESSEL_ACTIVITY']]).codes
    df[COLUMN_NAMES['LOAD_TYPE']] = pd.Categorical(df[COLUMN_NAMES['LOAD_TYPE']]).codes

    # Scale the numeric columns
    scaler = RobustScaler()
    
    # Provide the shape of the numeric columns before scaling
    st.write(f"Shape of numeric columns before scaling: {[df[col].shape for col in numeric_columns]}")
    
    # Scaling the numeric features
    scaled_numeric_features = scaler.fit_transform(df[numeric_columns])
    
    # Combine numeric and categorical features (if needed)
    df_scaled = pd.DataFrame(scaled_numeric_features, columns=numeric_columns)
    
    # Include the encoded categorical variables
    df_scaled[COLUMN_NAMES['VESSEL_ACTIVITY']] = df[COLUMN_NAMES['VESSEL_ACTIVITY']]
    df_scaled[COLUMN_NAMES['LOAD_TYPE']] = df[COLUMN_NAMES['LOAD_TYPE']]
    
    # Retain the original 'REPORT_DATE' and 'VESSEL_NAME' for reporting purposes
    df_scaled[COLUMN_NAMES['REPORT_DATE']] = df[COLUMN_NAMES['REPORT_DATE']]
    df_scaled[COLUMN_NAMES['VESSEL_NAME']] = df[COLUMN_NAMES['VESSEL_NAME']]
    
    # Provide the shape of the scaled numeric and encoded categorical features
    st.write(f"Shape of features after scaling: {df_scaled.shape}")
    
    return df_scaled

# Detect anomalies using IsolationForest and LocalOutlierFactor
def detect_anomalies(df):
    # Combine numeric and encoded categorical features for anomaly detection
    features = [
        COLUMN_NAMES['ME_CONSUMPTION'], COLUMN_NAMES['OBSERVERD_DISTANCE'], COLUMN_NAMES['SPEED'], 
        COLUMN_NAMES['DISPLACEMENT'], COLUMN_NAMES['STEAMING_TIME_HRS'], COLUMN_NAMES['WINDFORCE'], 
        COLUMN_NAMES['VESSEL_ACTIVITY'], COLUMN_NAMES['LOAD_TYPE']
    ]
    
    # Ensure all features are numeric
    df[features] = df[features].apply(pd.to_numeric, errors='coerce')
    df = df.dropna(subset=features)  # Drop rows with missing values after scaling
    
    # Check the shape of the features to be used in anomaly detection
    st.write(f"Shape of features for anomaly detection: {df[features].shape}")
    
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    
    lof_anomalies = lof.fit_predict(df[features])
    iso_forest_anomalies = iso_forest.fit_predict(df[features])
    
    combined_anomalies = (lof_anomalies == -1).astype(int) + (iso_forest_anomalies == -1).astype(int)
    anomalies = df[combined_anomalies > 1]  # Keep only anomalies

    # Prepare the results with vessel name, report date, and the detected discrepancy
    anomaly_results = anomalies[[COLUMN_NAMES['VESSEL_NAME'], COLUMN_NAMES['REPORT_DATE']]].copy()
    anomaly_results['Discrepancy'] = 'Anomaly detected by both IsolationForest and LocalOutlierFactor'
    
    return anomaly_results

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
        'anomalies': detect_anomalies(test_df),  # Return user-friendly format
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
    vessel_name = "ACE ETERNITY"  # Example vessel name

    try:
        results = run_advanced_validation(engine, vessel_name, date_filter)  # Pass engine as first argument
        print(pd.DataFrame(results['anomalies']))  # Display anomalies in DataFrame format
    except ValueError as e:
        print(str(e))
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
