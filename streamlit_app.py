import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler
from scipy.stats import ks_2samp, chisquare
import ruptures as rpt
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif

from validators.me_consumption_validation import validate_me_consumption, fetch_vessel_performance_data, fetch_vessel_coefficients, fetch_hull_performance_data
from validators.ae_consumption_validation import validate_ae_consumption
from validators.boiler_consumption_validation import validate_boiler_consumption, fetch_mcr_data
from validators.distance_validation import validate_distance_data
from validators.speed_validation import validate_speed, fetch_speed_data
from validators.fuel_rob_validation import validate_fuel_rob_for_vessel, fetch_sf_consumption_logs

st.set_page_config(layout="wide")

# Advanced Validation Functions
def preprocess_data(df):
    df.fillna(df.median(), inplace=True)
    
    df['VESSEL_ACTIVITY'] = pd.Categorical(df['VESSEL_ACTIVITY']).codes
    df['LOAD_TYPE'] = pd.Categorical(df['LOAD_TYPE']).codes
    
    numeric_columns = ['ME_CONSUMPTION', 'OBSERVERD_DISTANCE', 'SPEED', 'DISPLACEMENT', 
                       'STEAMING_TIME_HRS', 'WINDFORCE']
    scaler = RobustScaler()
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    
    return df

def detect_anomalies(df):
    features = ['ME_CONSUMPTION', 'OBSERVERD_DISTANCE', 'SPEED', 'DISPLACEMENT', 
                'STEAMING_TIME_HRS', 'WINDFORCE', 'VESSEL_ACTIVITY', 'LOAD_TYPE']
    
    scaled_features = RobustScaler().fit_transform(df[features])
    
    historical_anomaly_rate = 0.05
    contamination = max(0.01, min(historical_anomaly_rate, 0.1))
    
    lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    
    lof_anomalies = lof.fit_predict(scaled_features)
    iso_forest_anomalies = iso_forest.fit_predict(scaled_features)
    
    combined_anomalies = (lof_anomalies == -1).astype(int) + (iso_forest_anomalies == -1).astype(int)
    anomalies = df[combined_anomalies > 1]
    
    return anomalies

def detect_drift(train_df, test_df):
    continuous_features = ['ME_CONSUMPTION', 'OBSERVERD_DISTANCE', 'SPEED', 'DISPLACEMENT', 
                           'STEAMING_TIME_HRS', 'WINDFORCE']
    categorical_features = ['VESSEL_ACTIVITY', 'LOAD_TYPE']
    
    drift_detected = {}
    
    for feature in continuous_features:
        ks_stat, p_value = ks_2samp(train_df[feature], test_df[feature])
        drift_detected[feature] = p_value < 0.05

    for feature in categorical_features:
        chi_stat, p_value = chisquare(train_df[feature].value_counts(), test_df[feature].value_counts())
        drift_detected[feature] = p_value < 0.05
    
    return drift_detected

def detect_change_points(df):
    features = ['ME_CONSUMPTION', 'OBSERVERD_DISTANCE', 'SPEED']
    change_points = {}
    
    for feature in features:
        algo = rpt.Pelt(model="rbf").fit(df[feature].values)
        penalty = np.std(df[feature].values)
        change_points[feature] = algo.predict(pen=penalty)
    
    return change_points

def validate_relationships(df):
    continuous_features = ['ME_CONSUMPTION', 'SPEED', 'DISPLACEMENT', 'STEAMING_TIME_HRS']
    mutual_info = mutual_info_regression(df[continuous_features], df['ME_CONSUMPTION'])
    
    relationships = {}
    for i, feature in enumerate(continuous_features[1:]):
        relationships[feature] = mutual_info[i]
    
    categorical_features = ['VESSEL_ACTIVITY', 'LOAD_TYPE']
    cat_mutual_info = mutual_info_classif(df[categorical_features], df['ME_CONSUMPTION'])
    
    for i, feature in enumerate(categorical_features):
        relationships[feature] = cat_mutual_info[i]
    
    return relationships

def run_advanced_validation(df, vessel_name):
    df_vessel = df[df['vessel_name'] == vessel_name]
    df_processed = preprocess_data(df_vessel)
    
    train_df = df_processed[df_processed['REPORT_DATE'] < df_processed['REPORT_DATE'].max() - pd.Timedelta(days=90)]
    test_df = df_processed[df_processed['REPORT_DATE'] >= df_processed['REPORT_DATE'].max() - pd.Timedelta(days=90)]
    
    results = {
        'anomalies': detect_anomalies(test_df),
        'drift': detect_drift(train_df, test_df),
        'change_points': detect_change_points(test_df),
        'relationships': validate_relationships(test_df)
    }
    
    return results

def main():
    main_content, right_sidebar = st.columns([3, 1])

    with st.sidebar:
        st.sidebar.title('Validation Settings')
        time_range = st.sidebar.selectbox(
            "Validation Time Range",
            ("Last 1 Month", "Last 3 Months", "Last 6 Months"),
            key="time_range_select"
        )

        if time_range == "Last 1 Month":
            date_filter = datetime.now() - timedelta(days=30)
        elif time_range == "Last 3 Months":
            date_filter = datetime.now() - timedelta(days=90)
        else:  # Last 6 Months
            date_filter = datetime.now() - timedelta(days=180)

        st.sidebar.write("Validation Criteria:")
        me_consumption_check = st.sidebar.checkbox("ME Consumption", value=True, key="me_consumption_check")
        ae_consumption_check = st.sidebar.checkbox("AE Consumption", value=True, key="ae_consumption_check")
        boiler_consumption_check = st.sidebar.checkbox("Boiler Consumption", value=True, key="boiler_consumption_check")
        observed_distance_check = st.sidebar.checkbox("Observed Distance", value=True, key="observed_distance_check")
        speed_check = st.sidebar.checkbox("Speed", value=True, key="speed_check")
        fuel_rob_check = st.sidebar.checkbox("Fuel ROB", value=True, key="fuel_rob_check")

        advanced_validation_check = st.sidebar.checkbox("Run Advanced Validations", value=False, key="advanced_validation_check")

        max_vessels = st.sidebar.number_input("Maximum number of vessels to process (0 for all)", min_value=0, value=0, key="max_vessels")
        batch_size = st.sidebar.number_input("Batch size for distance validation", min_value=100, max_value=10000, value=1000, step=100, key="batch_size")

    with main_content:
        st.title('Vessel Data Validation')

        if st.button('Validate Data', key="validate_button"):
            try:
                validation_results = []

                if me_consumption_check or ae_consumption_check or boiler_consumption_check or speed_check or fuel_rob_check or advanced_validation_check:
                    df = fetch_vessel_performance_data(date_filter)
                    coefficients_df = fetch_vessel_coefficients()
                    hull_performance_df = fetch_hull_performance_data()
                    mcr_df = fetch_mcr_data(date_filter)
                    sf_consumption_logs = fetch_sf_consumption_logs(date_filter)
                    
                    if not df.empty:
                        vessel_groups = list(df.groupby('vessel_name'))
                        if max_vessels > 0:
                            vessel_groups = vessel_groups[:max_vessels]
                        
                        progress_bar = st.progress(0)
                        progress_text = st.empty()
                        
                        vessel_type_cache = {}
                        
                        for i, (vessel_name, vessel_data) in enumerate(vessel_groups):
                            vessel_type = vessel_data['vessel_type'].iloc[0]
                            vessel_coefficients = coefficients_df[coefficients_df['vessel_name'] == vessel_name].iloc[0] if not coefficients_df[coefficients_df['vessel_name'] == vessel_name].empty else None
                            
                            hull_performance = hull_performance_df[hull_performance_df['vessel_name'] == vessel_name]['hull_rough_power_loss_pct_ed'].iloc[0] if not hull_performance_df[hull_performance_df['vessel_name'] == vessel_name].empty else 0
                            hull_performance_factor = 1 + (hull_performance / 100)
                            
                            mcr_value = mcr_df[mcr_df['Vessel_Name'] == vessel_name]['ME_1_MCR_kW'].iloc[0] if not mcr_df[mcr_df['Vessel_Name'] == vessel_name].empty else None
                            mcr_value = float(mcr_value) if pd.notna(mcr_value) else None
                            
                            for _, row in vessel_data.iterrows():
                                failure_reasons = []
                                
                                if me_consumption_check:
                                    me_failure_reasons = validate_me_consumption(row, vessel_data, vessel_type, vessel_coefficients, hull_performance_factor)
                                    failure_reasons.extend(me_failure_reasons)
                                
                                if ae_consumption_check:
                                    ae_failure_reasons = validate_ae_consumption(row, vessel_data, date_filter)
                                    failure_reasons.extend(ae_failure_reasons)
                                
                                if boiler_consumption_check:
                                    boiler_failure_reasons = validate_boiler_consumption(row, mcr_value)
                                    failure_reasons.extend(boiler_failure_reasons)
                                
                                if speed_check:
                                    speed_failure_reasons = validate_speed(row, vessel_type_cache)
                                    failure_reasons.extend(speed_failure_reasons)
                                
                                if fuel_rob_check:
                                    fuel_rob_failures = validate_fuel_rob_for_vessel(sf_consumption_logs, vessel_name)
                                    failure_reasons.extend([failure['Remarks'] for failure in fuel_rob_failures if failure['Report Date'] == row['reportdate']])
                                
                                if failure_reasons:
                                    validation_results.append({
                                        'Vessel Name': vessel_name,
                                        'Report Date': row['reportdate'],
                                        'Remarks': ", ".join(failure_reasons)
                                    })
                            
                            if advanced_validation_check:
                                advanced_results = run_advanced_validation(df, vessel_name)
                                st.write(f"Advanced Validation Results for {vessel_name}:")
                                st.write(f"Anomalies detected: {len(advanced_results['anomalies'])}")
                                st.write("Drift detected in features:", ", ".join([f for f, d in advanced_results['drift'].items() if d]))
                                st.write("Change points detected:")
                                for feature, points in advanced_results['change_points'].items():
                                    st.write(f"  {feature}: {points}")
                                st.write("Feature relationships (Mutual Information):")
                                for feature, mi in advanced_results['relationships'].items():
                                    st.write(f"  {feature}: {mi:.4f}")
                                st.write("---")
                            
                            progress = (i + 1) / len(vessel_groups)
                            progress_bar.progress(progress)
                            progress_text.text(f"Validating: {progress:.0%}")
                        
                        progress_bar.empty()
                        progress_text.empty()
                
                if observed_distance_check:
                    with st.spinner('Performing distance validation...'):
                        distance_validation_results = validate_distance_data(date_filter, batch_size)
                        validation_results.extend(distance_validation_results.to_dict('records'))
                
                all_results = pd.DataFrame(validation_results)
                
                if not all_results.empty:
                    st.write("Validation Results:")
                    st.dataframe(all_results)
                    
                    csv = all_results.to_csv(index=False)
                    st.download_button(label="Download validation report as CSV", data=csv, file_name='validation_report.csv', mime='text/csv', key="download_button")
                else:
                    st.write("All data passed the validation checks!")

                if advanced_validation_check and st.button('Retrain Models'):
                    st.write("Retraining models... (implement retraining logic here)")
            
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

        st.write("This application validates vessel performance data based on selected criteria.")
        st.write("Use the checkboxes in the sidebar to select which validations to run, then click the 'Validate Data' button to start the validation process.")

    with right_sidebar:
        st.markdown("<h2 style='font-size: 18px;'>Validation Checks</h2>", unsafe_allow_html=True)
        
        st.markdown("<h3 style='font-size: 14px;'>ME Consumption Validations</h3>", unsafe_allow_html=True)
        st.markdown("...")  # (previous content)

        st.markdown("<h3 style='font-size: 14px;'>AE Consumption Validations</h3>", unsafe_allow_html=True)
        st.markdown("...")  # (previous content)

        st.markdown("<h3 style='font-size: 14px;'>Boiler Consumption Validations</h3>", unsafe_allow_html=True)
        st.markdown("...")  # (previous content)

        st.markdown("<h3 style='font-size: 14px;'>Observed Distance Validations</h3>", unsafe_allow_html=True)
        st.markdown("...")  # (previous content)

        st.markdown("<h3 style='font-size: 14px;'>Speed Validations</h3>", unsafe_allow_html=True)
        st.markdown("""
        <div style='font-size: 10px;'>
        1. Negative speed: Flags if observed speed is negative.<br>
        2. Low speed at sea: Checks if speed is unusually low during sea passage.<br>
        3. Unusual maneuvering speed: Flags if speed is outside expected range during maneuvering.<br>
        4. Non-zero port speed: Checks if speed is non-zero when in port.<br>
        5. High speed for vessel type: Flags if speed exceeds maximum for container or non-container vessels.<br>
        6. Speed-distance-time alignment: Compares observed speed with calculated speed based on distance and time.<br>
        7. Consistency check: Flags if speed is positive but engine parameters indicate no movement.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<h3 style='font-size: 14px;'>Fuel ROB Validations</h3>", unsafe_allow_html=True)
        st.markdown("""
        <div style='font-size: 10px;'>
        1. HSFO ROB: Validates High-Sulfur Fuel Oil Remaining On Board.<br>
        2. LSMGO ROB: Validates Low-Sulfur Marine Gas Oil Remaining On Board.<br>
        3. ULSFO ROB: Validates Ultra-Low-Sulfur Fuel Oil Remaining On Board.<br>
        4. VLSFO ROB: Validates Very Low-Sulfur Fuel Oil Remaining On Board.<br>
        5. MDO ROB: Validates Marine Diesel Oil Remaining On Board.<br>
        6. LNG ROB: Validates Liquefied Natural Gas Remaining On Board.<br>
        Each validation checks if the current ROB matches the calculated value based on previous ROB, bunkered quantity, and total consumption.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<h3 style='font-size: 14px;'>Advanced Validations</h3>", unsafe_allow_html=True)
        st.markdown("""
        <div style='font-size: 10px;'>
        1. Anomaly Detection: Uses ensemble of Isolation Forest and Local Outlier Factor to detect unusual data points.<br>
        2. Drift Detection: Identifies significant changes in data distribution over time.<br>
        3. Change Point Detection: Detects abrupt changes in time series data.<br>
        4. Feature Relationship Analysis: Measures mutual information between features to understand complex dependencies.
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
