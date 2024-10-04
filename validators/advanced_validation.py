import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler
from sqlalchemy.exc import SQLAlchemyError
import streamlit as st
from database import get_db_engine
from datetime import datetime, timedelta
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

@st.cache_data
def load_data(vessel_name, date_filter):
    engine = get_db_engine()
    try:
        query = f"""
        SELECT {', '.join(f'"{col}"' for col in COLUMN_NAMES.values())}
        FROM sf_consumption_logs
        WHERE "{COLUMN_NAMES['VESSEL_NAME']}" = %s
        AND "{COLUMN_NAMES['REPORT_DATE']}" >= %s
        ORDER BY "{COLUMN_NAMES['REPORT_DATE']}"
        """
        df = pd.read_sql_query(query, engine, params=(vessel_name, date_filter))
        df[COLUMN_NAMES['REPORT_DATE']] = pd.to_datetime(df[COLUMN_NAMES['REPORT_DATE']])
        logging.info(f"Data loaded for {vessel_name}. Shape: {df.shape}")
        return df
    except SQLAlchemyError as e:
        logging.error(f"Error fetching data for {vessel_name}: {str(e)}")
        st.error(f"Error fetching data for {vessel_name}: {str(e)}")
        return pd.DataFrame()

def preprocess_data(df):
    numeric_columns = [
        COLUMN_NAMES['ME_CONSUMPTION'], COLUMN_NAMES['OBSERVERD_DISTANCE'], 
        COLUMN_NAMES['SPEED'], COLUMN_NAMES['DISPLACEMENT'], 
        COLUMN_NAMES['STEAMING_TIME_HRS'], COLUMN_NAMES['WINDFORCE']
    ]
    
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    missing_values_info = df[numeric_columns].isna().sum().to_dict()
    logging.info(f"Missing values per column: {missing_values_info}")
    st.write(f"Missing values per column: {missing_values_info}")
    
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
    
    df[COLUMN_NAMES['VESSEL_ACTIVITY']] = pd.Categorical(df[COLUMN_NAMES['VESSEL_ACTIVITY']]).codes
    df[COLUMN_NAMES['LOAD_TYPE']] = pd.Categorical(df[COLUMN_NAMES['LOAD_TYPE']]).codes

    scaler = RobustScaler()
    
    logging.info(f"Shape of numeric columns before scaling: {[df[col].shape for col in numeric_columns]}")
    st.write(f"Shape of numeric columns before scaling: {[df[col].shape for col in numeric_columns]}")
    
    scaled_numeric_features = scaler.fit_transform(df[numeric_columns])
    
    df_scaled = pd.DataFrame(scaled_numeric_features, columns=numeric_columns, index=df.index)

    df_scaled[COLUMN_NAMES['VESSEL_ACTIVITY']] = df[COLUMN_NAMES['VESSEL_ACTIVITY']]
    df_scaled[COLUMN_NAMES['LOAD_TYPE']] = df[COLUMN_NAMES['LOAD_TYPE']]
    df_scaled[COLUMN_NAMES['REPORT_DATE']] = df[COLUMN_NAMES['REPORT_DATE']]
    df_scaled[COLUMN_NAMES['VESSEL_NAME']] = df[COLUMN_NAMES['VESSEL_NAME']]

    logging.info(f"Shape of features after scaling: {df_scaled.shape}")
    st.write(f"Shape of features after scaling: {df_scaled.shape}")
    
    return df_scaled

def detect_anomalies(df):
    features = [
        COLUMN_NAMES['ME_CONSUMPTION'], COLUMN_NAMES['OBSERVERD_DISTANCE'], 
        COLUMN_NAMES['SPEED'], COLUMN_NAMES['DISPLACEMENT'], 
        COLUMN_NAMES['STEAMING_TIME_HRS'], COLUMN_NAMES['WINDFORCE'], 
        COLUMN_NAMES['VESSEL_ACTIVITY'], COLUMN_NAMES['LOAD_TYPE']
    ]
    
    logging.info(f"Shape of features for anomaly detection: {df[features].shape}")
    st.write(f"Shape of features for anomaly detection: {df[features].shape}")
    
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    
    lof_anomalies = lof.fit_predict(df[features])
    iso_forest_anomalies = iso_forest.fit_predict(df[features])
    
    combined_anomalies = (lof_anomalies == -1).astype(int) + (iso_forest_anomalies == -1).astype(int)
    anomalies = df[combined_anomalies > 1]

    anomaly_results = anomalies[[COLUMN_NAMES['VESSEL_NAME'], COLUMN_NAMES['REPORT_DATE']]].copy()
    anomaly_results['Discrepancy'] = 'Anomaly detected by both IsolationForest and LocalOutlierFactor'
    
    # Add feature values for anomalies
    for feature in features:
        anomaly_results[feature] = anomalies[feature]
    
    return anomaly_results

def run_advanced_validation(engine, vessel_name, date_filter):
    df = load_data(vessel_name, date_filter)
    
    if df.empty:
        raise ValueError(f"No data available for vessel {vessel_name}")
    
    df_processed = preprocess_data(df)
    
    train_df = df_processed[df_processed[COLUMN_NAMES['REPORT_DATE']] < df_processed[COLUMN_NAMES['REPORT_DATE']].max() - pd.Timedelta(days=90)]
    test_df = df_processed[df_processed[COLUMN_NAMES['REPORT_DATE']] >= df_processed[COLUMN_NAMES['REPORT_DATE']].max() - pd.Timedelta(days=90)]
    
    anomalies = detect_anomalies(test_df)
    
    logging.info(f"Advanced Validation Results for {vessel_name}:")
    logging.info(f"Anomalies detected: {len(anomalies)}")
    st.write(f"Advanced Validation Results for {vessel_name}:")
    st.write(f"Anomalies detected: {len(anomalies)}")
    
    if not anomalies.empty:
        logging.info("Anomaly Details:")
        logging.info(anomalies.to_string())
        st.write("Anomaly Details:")
        st.dataframe(anomalies)
    else:
        logging.info("No anomalies detected.")
        st.write("No anomalies detected.")
    
    return {"anomalies": anomalies}

def main():
    st.title('Vessel Data Validation')

    # Sidebar information
    st.sidebar.write("Data validation happens for the last 6 months.")

    # Get list of vessels (you might need to implement this function)
    # vessels = get_vessel_list()
    vessels = ["ACE ETERNITY", "VESSEL 2", "VESSEL 3"]  # Example list

    # Vessel selection
    selected_vessel = st.selectbox("Select a vessel", vessels)

    if st.button('Run Advanced Validation'):
        engine = get_db_engine()
        date_filter = datetime.now() - timedelta(days=180)

        try:
            results = run_advanced_validation(engine, selected_vessel, date_filter)
            
            # The results are already displayed in the run_advanced_validation function
            # You can add additional processing or display of results here if needed
            
            # Example: Download results as CSV
            if not results['anomalies'].empty:
                csv = results['anomalies'].to_csv(index=False)
                st.download_button(
                    label="Download anomalies as CSV",
                    data=csv,
                    file_name=f'{selected_vessel}_anomalies.csv',
                    mime='text/csv',
                )
            
        except Exception as e:
            logging.error(f"An error occurred: {str(e)}", exc_info=True)
            st.error(f"An error occurred: {str(e)}")
            st.write("Error details:")
            st.write(e)

if __name__ == "__main__":
    main()
