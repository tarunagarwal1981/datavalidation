import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import KBinsDiscretizer
from scipy.stats import ks_2samp
import ruptures as rpt
from database import get_db_engine, fetch_vessel_performance_data

COLUMN_NAMES = {
    'VESSEL_NAME': 'vessel_name',
    'REPORT_DATE': 'reportdate',
    'ME_CONSUMPTION': 'me_consumption',
    'OBSERVERD_DISTANCE': 'observed_distance',
    'SPEED': 'speed',
    'DISPLACEMENT': 'displacement',
    'STEAMING_TIME_HRS': 'steaming_time_hrs',
    'WINDFORCE': 'windforce',
    'VESSEL_ACTIVITY': 'vessel_activity',
    'LOAD_TYPE': 'load_type'
}

def preprocess_data(df):
    df[COLUMN_NAMES['VESSEL_NAME']] = df[COLUMN_NAMES['VESSEL_NAME']].astype(str)
    df[COLUMN_NAMES['VESSEL_ACTIVITY']] = df[COLUMN_NAMES['VESSEL_ACTIVITY']].astype(str)
    df[COLUMN_NAMES['LOAD_TYPE']] = df[COLUMN_NAMES['LOAD_TYPE']].astype(str)
    df[COLUMN_NAMES['REPORT_DATE']] = pd.to_datetime(df[COLUMN_NAMES['REPORT_DATE']])
    
    df = df.dropna(how='all')
    df = df.fillna(df.select_dtypes(include=['number']).mean())
    
    critical_columns = [
        COLUMN_NAMES['ME_CONSUMPTION'], COLUMN_NAMES['OBSERVERD_DISTANCE'], COLUMN_NAMES['SPEED'],
        COLUMN_NAMES['DISPLACEMENT'], COLUMN_NAMES['STEAMING_TIME_HRS'], COLUMN_NAMES['WINDFORCE'],
        COLUMN_NAMES['VESSEL_ACTIVITY'], COLUMN_NAMES['LOAD_TYPE']
    ]
    df = df.dropna(subset=critical_columns)
    
    if df.shape[0] == 0:
        return df

    df[COLUMN_NAMES['VESSEL_ACTIVITY']] = pd.Categorical(df[COLUMN_NAMES['VESSEL_ACTIVITY']]).codes
    df[COLUMN_NAMES['LOAD_TYPE']] = pd.Categorical(df[COLUMN_NAMES['LOAD_TYPE']]).codes

    numeric_columns = [
        COLUMN_NAMES['ME_CONSUMPTION'], COLUMN_NAMES['OBSERVERD_DISTANCE'], COLUMN_NAMES['SPEED'],
        COLUMN_NAMES['DISPLACEMENT'], COLUMN_NAMES['STEAMING_TIME_HRS'], COLUMN_NAMES['WINDFORCE']
    ]
    scaler = RobustScaler()
    if df[numeric_columns].shape[0] > 0:
        df.loc[:, numeric_columns] = scaler.fit_transform(df[numeric_columns])

    return df

def detect_anomalies(df, n_neighbors=20):
    df = df.dropna()
    if df.shape[0] == 0:
        return pd.DataFrame()

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
        drift_detected[feature] = p_value < 0.05

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

    if df.shape[0] == 0:
        return {feature: 0.0 for feature in features[1:]}

    discretizer = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')
    try:
        target = discretizer.fit_transform(df[[COLUMN_NAMES['ME_CONSUMPTION']]]).ravel()
    except ValueError:
        return {feature: 0.0 for feature in features[1:]}

    mutual_info = mutual_info_regression(df[features[1:]], target)

    relationships = {}
    for i, feature in enumerate(features[1:]):
        relationships[feature] = mutual_info[i]

    return relationships

def format_validation_results(results):
    formatted_results = []
    
    for _, row in results['anomalies'].iterrows():
        formatted_results.append({
            'Vessel Name': row[COLUMN_NAMES['VESSEL_NAME']],
            'Date': row[COLUMN_NAMES['REPORT_DATE']].strftime('%Y-%m-%d'),
            'Issue': 'Unusual Data Detected',
            'Explanation': f"Unusual values were found in {', '.join([k for k, v in row.items() if v == -1 and k != COLUMN_NAMES['REPORT_DATE']])}. This could indicate measurement errors or exceptional operating conditions."
        })
    
    for feature, has_drift in results['drift'].items():
        if has_drift:
            formatted_results.append({
                'Vessel Name': results['anomalies'][COLUMN_NAMES['VESSEL_NAME']].iloc[0] if not results['anomalies'].empty else "N/A",
                'Date': 'Multiple Dates',
                'Issue': 'Data Trend Change Detected',
                'Explanation': f"The pattern of {feature} has changed significantly over time. This might indicate changes in operating conditions or equipment performance."
            })
    
    for feature, points in results['change_points'].items():
        if points:
            formatted_results.append({
                'Vessel Name': results['anomalies'][COLUMN_NAMES['VESSEL_NAME']].iloc[0] if not results['anomalies'].empty else "N/A",
                'Date': 'Specific Dates',
                'Issue': 'Sudden Changes Detected',
                'Explanation': f"Sudden changes were detected in {feature}. This could indicate equipment changes, maintenance events, or changes in operating procedures."
            })
    
    weak_relationships = [f for f, v in results['relationships'].items() if v < 0.3]
    if weak_relationships:
        formatted_results.append({
            'Vessel Name': results['anomalies'][COLUMN_NAMES['VESSEL_NAME']].iloc[0] if not results['anomalies'].empty else "N/A",
            'Date': 'Overall Analysis',
            'Issue': 'Unexpected Data Relationships',
            'Explanation': f"The following factors show weaker than expected influence on fuel consumption: {', '.join(weak_relationships)}. This might indicate data quality issues or unusual operating conditions."
        })
    
    return formatted_results

def run_advanced_validation(df, vessel_name):
    vessel_df = df[df[COLUMN_NAMES['VESSEL_NAME']] == vessel_name]
    
    # Split data into training (first 6 months) and validation (last 6 months)
    vessel_df = vessel_df.sort_values(by=COLUMN_NAMES['REPORT_DATE'])
    mid_point = len(vessel_df) // 2
    train_df = vessel_df.iloc[:mid_point]
    test_df = vessel_df.iloc[mid_point:]

    train_df = preprocess_data(train_df)
    test_df = preprocess_data(test_df)

    if test_df.shape[0] == 0:
        return []

    anomalies = detect_anomalies(test_df)
    drift = detect_drift(train_df, test_df)
    change_points = detect_change_points(test_df)
    relationships = validate_relationships(train_df)

    results = {
        'anomalies': anomalies,
        'drift': drift,
        'change_points': change_points,
        'relationships': relationships
    }
    
    return format_validation_results(results)

def main():
    st.title('Vessel Data Validation')

    st.sidebar.write("Data validation happened for the last 3 months.")

    if st.button('Validate Data'):
        engine = get_db_engine()
        try:
            three_months_ago = datetime.now() - timedelta(days=90)
            df = fetch_vessel_performance_data(engine, three_months_ago)
            
            if not df.empty:
                validation_results = []
                for vessel_name in df[COLUMN_NAMES['VESSEL_NAME']].unique():
                    vessel_results = run_advanced_validation(df, vessel_name)
                    validation_results.extend(vessel_results)
                
                if validation_results:
                    result_df = pd.DataFrame(validation_results)
                    st.write("Validation Results:")
                    st.dataframe(result_df)
                    
                    csv = result_df.to_csv(index=False)
                    st.download_button(label="Download validation report as CSV", data=csv, file_name='validation_report.csv', mime='text/csv')
                else:
                    st.write("All data passed the validation checks!")
            else:
                st.write("No data found for the last 3 months.")
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

    st.sidebar.write("Validation Criteria:")
    st.sidebar.write("- Anomaly Detection")
    st.sidebar.write("- Data Drift Detection")
    st.sidebar.write("- Change Point Detection")
    st.sidebar.write("- Feature Relationship Analysis")

    st.write("This application validates vessel performance data based on multiple advanced criteria.")
    st.write("Click the 'Validate Data' button to start the validation process.")

if __name__ == "__main__":
    main()
