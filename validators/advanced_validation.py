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
    numeric_columns = [
        COLUMN_NAMES['ME_CONSUMPTION'], COLUMN_NAMES['OBSERVERD_DISTANCE'],
        COLUMN_NAMES['SPEED'], COLUMN_NAMES['DISPLACEMENT'],
        COLUMN_NAMES['STEAMING_TIME_HRS'], COLUMN_NAMES['WINDFORCE']
    ]

    # Convert to numeric and handle invalid entries
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

    # Fill missing values for numeric columns
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())

    # Fill missing categorical values with mode
    categorical_columns = [COLUMN_NAMES['VESSEL_ACTIVITY'], COLUMN_NAMES['LOAD_TYPE']]
    df[categorical_columns] = df[categorical_columns].apply(lambda col: col.fillna(col.mode()[0]))

    # Encoding categorical variables
    df[COLUMN_NAMES['VESSEL_ACTIVITY']] = pd.Categorical(df[COLUMN_NAMES['VESSEL_ACTIVITY']]).codes
    df[COLUMN_NAMES['LOAD_TYPE']] = pd.Categorical(df[COLUMN_NAMES['LOAD_TYPE']]).codes

    # Scaling numeric columns
    scaler = RobustScaler()
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

    return df

# Detect anomalies using Isolation Forest and Local Outlier Factor
def detect_anomalies(df):
    features = [
        COLUMN_NAMES['ME_CONSUMPTION'], COLUMN_NAMES['OBSERVERD_DISTANCE'],
        COLUMN_NAMES['SPEED'], COLUMN_NAMES['DISPLACEMENT'],
        COLUMN_NAMES['STEAMING_TIME_HRS'], COLUMN_NAMES['WINDFORCE'],
        COLUMN_NAMES['VESSEL_ACTIVITY'], COLUMN_NAMES['LOAD_TYPE']
    ]

    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
    iso_forest = IsolationForest(contamination=0.1, random_state=42)

    lof_anomalies = lof.fit_predict(df[features])
    iso_forest_anomalies = iso_forest.fit_predict(df[features])

    combined_anomalies = (lof_anomalies == -1).astype(int) + (iso_forest_anomalies == -1).astype(int)
    anomalies = df[combined_anomalies > 1]  # Detected by both models

    if anomalies.empty:
        return pd.DataFrame(columns=[COLUMN_NAMES['VESSEL_NAME'], COLUMN_NAMES['REPORT_DATE'], 'Discrepancy'])

    anomaly_results = anomalies[[COLUMN_NAMES['VESSEL_NAME'], COLUMN_NAMES['REPORT_DATE']]].copy()
    anomaly_results['Discrepancy'] = 'Anomaly detected'

    return anomaly_results

# Detect data drift using KS Test for numeric columns and chi-squared test for categorical columns
def detect_data_drift(train_df, test_df):
    drift_results = []
    numeric_columns = [
        COLUMN_NAMES['ME_CONSUMPTION'], COLUMN_NAMES['OBSERVERD_DISTANCE'],
        COLUMN_NAMES['SPEED'], COLUMN_NAMES['DISPLACEMENT'],
        COLUMN_NAMES['STEAMING_TIME_HRS'], COLUMN_NAMES['WINDFORCE']
    ]
    categorical_columns = [COLUMN_NAMES['VESSEL_ACTIVITY'], COLUMN_NAMES['LOAD_TYPE']]

    for col in numeric_columns:
        stat, p_value = ks_2samp(train_df[col], test_df[col])
        if p_value < 0.05:
            drift_results.append({
                'VESSEL_NAME': test_df[COLUMN_NAMES['VESSEL_NAME']].iloc[0],
                'REPORT_DATE': test_df[COLUMN_NAMES['REPORT_DATE']].max(),
                'Discrepancy': f'Data drift detected in {col}'
            })

    for col in categorical_columns:
        contingency_table = pd.crosstab(train_df[col], test_df[col])
        if contingency_table.size > 1:
            chi2, p_value, _, _ = chi2_contingency(contingency_table)
            if p_value < 0.05:
                drift_results.append({
                    'VESSEL_NAME': test_df[COLUMN_NAMES['VESSEL_NAME']].iloc[0],
                    'REPORT_DATE': test_df[COLUMN_NAMES['REPORT_DATE']].max(),
                    'Discrepancy': f'Data drift detected in {col}'
                })

    if drift_results:
        return pd.DataFrame(drift_results)
    else:
        return pd.DataFrame(columns=['VESSEL_NAME', 'REPORT_DATE', 'Discrepancy'])

# Main function to run advanced validation on vessel data
def run_advanced_validation(engine, vessel_name, date_filter):
    df = load_data(engine, vessel_name, date_filter)

    if df.empty:
        raise ValueError(f"No data available for vessel {vessel_name}")

    df_processed = preprocess_data(df)

    max_date = df_processed[COLUMN_NAMES['REPORT_DATE']].max()
    cutoff_date = max_date - pd.Timedelta(days=30)

    train_df = df_processed[df_processed[COLUMN_NAMES['REPORT_DATE']] < cutoff_date]
    test_df = df_processed[df_processed[COLUMN_NAMES['REPORT_DATE']] >= cutoff_date]

    if len(train_df) < 5 or len(test_df) < 5:
        raise ValueError("Not enough data in training or testing set for drift detection.")

    drift_df = detect_data_drift(train_df, test_df)
    anomalies_df = detect_anomalies(test_df)

    combined_results = pd.concat([anomalies_df, drift_df], ignore_index=True)

    if combined_results.empty:
        st.write("No discrepancies detected.")
        final_results = pd.DataFrame(columns=['VESSEL_NAME', 'REPORT_DATE', 'Discrepancy'])
    else:
        final_results = combined_results[['VESSEL_NAME', 'REPORT_DATE', 'Discrepancy']]

    return final_results

# Example of how to use the advanced validation functionality
if __name__ == "__main__":
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
