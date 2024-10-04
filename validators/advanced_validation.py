import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler
from scipy.stats import ks_2samp, chi2_contingency
from sqlalchemy.exc import SQLAlchemyError
import streamlit as st
from database import get_db_engine
from datetime import datetime, timedelta

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
def load_data(_engine, vessel_name, date_filter):
    try:
        query = f"""
        SELECT {', '.join(f'"{col}"' for col in COLUMN_NAMES.values())}
        FROM sf_consumption_logs
        WHERE "{COLUMN_NAMES['VESSEL_NAME']}" = %s
        AND "{COLUMN_NAMES['REPORT_DATE']}" >= %s
        ORDER BY "{COLUMN_NAMES['REPORT_DATE']}"
        """
        df = pd.read_sql_query(query, _engine, params=(vessel_name, date_filter))
        df[COLUMN_NAMES['REPORT_DATE']] = pd.to_datetime(df[COLUMN_NAMES['REPORT_DATE']])
        return df
    except SQLAlchemyError as e:
        st.error(f"Error fetching data for {vessel_name}: {str(e)}")
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

    # Display missing values in numeric columns
    missing_values_info = df[numeric_columns].isna().sum().to_dict()
    st.write(f"Missing values in numeric columns: {missing_values_info}")

    # Fill missing numeric values with column medians
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())

    # Handle missing values in categorical columns
    categorical_columns = [COLUMN_NAMES['VESSEL_ACTIVITY'], COLUMN_NAMES['LOAD_TYPE']]
    missing_categorical = df[categorical_columns].isna().sum().to_dict()
    st.write(f"Missing values in categorical columns: {missing_categorical}")

    # Fill missing categorical values with the mode (most frequent value)
    for col in categorical_columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    # Encoding categorical variables
    df[COLUMN_NAMES['VESSEL_ACTIVITY']] = pd.Categorical(df[COLUMN_NAMES['VESSEL_ACTIVITY']]).codes
    df[COLUMN_NAMES['LOAD_TYPE']] = pd.Categorical(df[COLUMN_NAMES['LOAD_TYPE']]).codes

    # Scale the numeric columns
    scaler = RobustScaler()

    # Provide the shape of the numeric columns before scaling
    st.write(f"Shape of numeric columns before scaling: {[df[col].shape for col in numeric_columns]}")

    # Scaling the numeric features
    scaled_numeric_features = scaler.fit_transform(df[numeric_columns])

    # Combine scaled numeric features with categorical ones
    df_scaled = pd.DataFrame(scaled_numeric_features, columns=numeric_columns)

    # Add categorical features back to the dataframe
    df_scaled[COLUMN_NAMES['VESSEL_ACTIVITY']] = df[COLUMN_NAMES['VESSEL_ACTIVITY']].values
    df_scaled[COLUMN_NAMES['LOAD_TYPE']] = df[COLUMN_NAMES['LOAD_TYPE']].values

    # Add REPORT_DATE and VESSEL_NAME back to the dataframe
    df_scaled[COLUMN_NAMES['REPORT_DATE']] = df[COLUMN_NAMES['REPORT_DATE']].values
    df_scaled[COLUMN_NAMES['VESSEL_NAME']] = df[COLUMN_NAMES['VESSEL_NAME']].values

    # Ensure all features are included
    st.write(f"Shape of features after scaling: {df_scaled.shape}")

    return df_scaled

# Detect anomalies using IsolationForest and LocalOutlierFactor
def detect_anomalies(df):
    # Select features for anomaly detection (both numeric and categorical)
    features = [
        COLUMN_NAMES['ME_CONSUMPTION'], COLUMN_NAMES['OBSERVERD_DISTANCE'],
        COLUMN_NAMES['SPEED'], COLUMN_NAMES['DISPLACEMENT'],
        COLUMN_NAMES['STEAMING_TIME_HRS'], COLUMN_NAMES['WINDFORCE'],
        COLUMN_NAMES['VESSEL_ACTIVITY'], COLUMN_NAMES['LOAD_TYPE']
    ]

    # Check the shape of the features to be used in anomaly detection
    st.write(f"Shape of features for anomaly detection: {df[features].shape}")

    # Check if there is enough data
    if df[features].shape[0] < 5:
        st.write("Not enough data for anomaly detection.")
        return pd.DataFrame(columns=[COLUMN_NAMES['VESSEL_NAME'], COLUMN_NAMES['REPORT_DATE'], 'Discrepancy'])

    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
    iso_forest = IsolationForest(contamination=0.1, random_state=42)

    lof_anomalies = lof.fit_predict(df[features])
    iso_forest_anomalies = iso_forest.fit_predict(df[features])

    combined_anomalies = (lof_anomalies == -1).astype(int) + (iso_forest_anomalies == -1).astype(int)
    anomalies = df[combined_anomalies > 1]  # Keep only anomalies detected by both methods

    # Check if anomalies are found
    st.write(f"Anomalies detected: {len(anomalies)}")

    if anomalies.empty:
        return pd.DataFrame(columns=[COLUMN_NAMES['VESSEL_NAME'], COLUMN_NAMES['REPORT_DATE'], 'Discrepancy'])

    # Prepare the results with vessel name, report date, and the detected discrepancy
    anomaly_results = anomalies[[COLUMN_NAMES['VESSEL_NAME'], COLUMN_NAMES['REPORT_DATE']]].copy()
    anomaly_results['Discrepancy'] = 'Anomaly detected'

    return anomaly_results

# Detect data drift between train and test sets
def detect_data_drift(train_df, test_df):
    drift_results = []
    numeric_columns = [
        COLUMN_NAMES['ME_CONSUMPTION'], COLUMN_NAMES['OBSERVERD_DISTANCE'],
        COLUMN_NAMES['SPEED'], COLUMN_NAMES['DISPLACEMENT'],
        COLUMN_NAMES['STEAMING_TIME_HRS'], COLUMN_NAMES['WINDFORCE']
    ]
    categorical_columns = [COLUMN_NAMES['VESSEL_ACTIVITY'], COLUMN_NAMES['LOAD_TYPE']]

    for col in numeric_columns:
        # Perform Kolmogorov-Smirnov test
        stat, p_value = ks_2samp(train_df[col], test_df[col])
        drift_detected = p_value < 0.05  # Significance level of 0.05
        if drift_detected:
            drift_results.append({
                'VESSEL_NAME': test_df[COLUMN_NAMES['VESSEL_NAME']].iloc[0],
                'REPORT_DATE': test_df[COLUMN_NAMES['REPORT_DATE']].max(),
                'Discrepancy': f'Data drift detected in {col}'
            })

    for col in categorical_columns:
        # Create contingency table
        contingency_table = pd.crosstab(train_df[col], test_df[col])
        if contingency_table.size == 0:
            st.write(f"Skipping chi-squared test for {col} due to empty contingency table.")
            continue  # Skip to next column
        elif contingency_table.shape[0] == 1 or contingency_table.shape[1] == 1:
            st.write(f"Skipping chi-squared test for {col} due to insufficient categories.")
            continue  # Skip to next column
        else:
            # Perform Chi-squared test
            try:
                chi2, p_value, dof, ex = chi2_contingency(contingency_table)
                drift_detected = p_value < 0.05
                if drift_detected:
                    drift_results.append({
                        'VESSEL_NAME': test_df[COLUMN_NAMES['VESSEL_NAME']].iloc[0],
                        'REPORT_DATE': test_df[COLUMN_NAMES['REPORT_DATE']].max(),
                        'Discrepancy': f'Data drift detected in {col}'
                    })
            except ValueError as e:
                st.write(f"Skipping chi-squared test for {col} due to error: {e}")
                continue  # Skip to next column

    if drift_results:
        drift_df = pd.DataFrame(drift_results)
    else:
        drift_df = pd.DataFrame(columns=['VESSEL_NAME', 'REPORT_DATE', 'Discrepancy'])

    return drift_df

# Main function to run advanced validations on the vessel data
def run_advanced_validation(engine, vessel_name, date_filter):
    df = load_data(engine, vessel_name, date_filter)

    if df.empty:
        raise ValueError(f"No data available for vessel {vessel_name}")

    df_processed = preprocess_data(df)

    # Check the date range of the data
    st.write(f"Data Date Range: {df_processed[COLUMN_NAMES['REPORT_DATE']].min()} to {df_processed[COLUMN_NAMES['REPORT_DATE']].max()}")

    # Adjust the cutoff date to ensure sufficient data
    max_date = df_processed[COLUMN_NAMES['REPORT_DATE']].max()
    cutoff_date = max_date - pd.Timedelta(days=30)  # Adjusted from 90 to 30 days
    st.write(f"Cutoff date for train-test split: {cutoff_date}")

    train_df = df_processed[df_processed[COLUMN_NAMES['REPORT_DATE']] < cutoff_date]
    test_df = df_processed[df_processed[COLUMN_NAMES['REPORT_DATE']] >= cutoff_date]

    # Log the number of records in each set
    st.write(f"Number of records in training set: {len(train_df)}")
    st.write(f"Number of records in testing set: {len(test_df)}")

    if len(train_df) < 5 or len(test_df) < 5:
        raise ValueError("Not enough data in training or testing set to perform data drift detection.")

    # Perform data drift detection
    drift_df = detect_data_drift(train_df, test_df)

    # Detect anomalies in test set
    anomalies_df = detect_anomalies(test_df)

    # Combine the results into a single DataFrame
    combined_results = pd.concat([anomalies_df, drift_df], ignore_index=True)

    if combined_results.empty:
        st.write("No discrepancies detected.")
        # Return an empty DataFrame with the required columns
        final_results = pd.DataFrame(columns=['VESSEL_NAME', 'REPORT_DATE', 'Discrepancy'])
    else:
        # Prepare the results to have only the required columns
        final_results = combined_results[['VESSEL_NAME', 'REPORT_DATE', 'Discrepancy']]

    return final_results

# Example of how to use the advanced validation functionality
if __name__ == "__main__":
    from database import get_db_engine  # Import the function to get the engine

    engine = get_db_engine()  # Get the database engine
    date_filter = datetime.now() - timedelta(days=180)  # Last 6 months
    vessel_name = "ACE ETERNITY"  # Replace with actual vessel name

    try:
        final_results = run_advanced_validation(engine, vessel_name, date_filter)
        st.write(f"Advanced Validation Results for {vessel_name}:")
        if final_results.empty:
            st.write("No discrepancies detected.")
        else:
            st.write(final_results)  # Display the final results table
    except ValueError as e:
        st.error(f"An error occurred: {str(e)}")
