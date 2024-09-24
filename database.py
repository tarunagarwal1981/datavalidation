from sqlalchemy import create_engine
import pandas as pd
from datetime import datetime, timedelta
import urllib.parse

DB_CONFIG = {
    'host': 'aws-0-ap-south-1.pooler.supabase.com',
    'database': 'postgres',
    'user': 'postgres.conrxbcvuogbzfysomov',
    'password': 'wXAryCC8@iwNvj#',
    'port': '6543'
}

def get_db_engine():
    encoded_password = urllib.parse.quote(DB_CONFIG['password'])
    db_url = f"postgresql+psycopg2://{DB_CONFIG['user']}:{encoded_password}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    engine = create_engine(db_url)
    return engine

def fetch_vessel_performance_data(engine):
    query = """
    SELECT vps.*, vp.vessel_type
    FROM vessel_performance_summary vps
    LEFT JOIN vessel_particulars vp ON vps.vessel_name = vp.vessel_name
    WHERE vps.reportdate >= %s
    ORDER BY vps.vessel_name, vps.reportdate;
    """
    three_months_ago = datetime.now() - timedelta(days=90)
    df = pd.read_sql_query(query, engine, params=(three_months_ago,))
    return df

def fetch_sf_consumption_logs(engine):
    query = """
    SELECT *
    FROM sf_consumption_logs
    WHERE reportdate >= %s
    ORDER BY reportdate;
    """
    three_months_ago = datetime.now() - timedelta(days=90)
    df = pd.read_sql_query(query, engine, params=(three_months_ago,))
    return df

def fetch_vessel_coefficients(engine):
    query = """
    SELECT *
    FROM vessel_performance_coefficients;
    """
    return pd.read_sql_query(query, engine)

def fetch_hull_performance_data(engine):
    query = """
    SELECT vessel_name, hull_rough_power_loss_pct_ed
    FROM hull_performance_six_months;
    """
    return pd.read_sql_query(query, engine)

def fetch_mcr_data(engine):
    query = """
    SELECT "Vessel_Name", 
           CAST(NULLIF("ME_1_MCR_kW", '') AS FLOAT) AS "ME_1_MCR_kW"
    FROM machinery_particulars;
    """
    return pd.read_sql_query(query, engine)

def merge_vessel_and_consumption_data(vessel_df, consumption_df):
    # Print column names for debugging
    print("Vessel DataFrame columns:", vessel_df.columns)
    print("Consumption DataFrame columns:", consumption_df.columns)
    
    # Assuming there's a common column to join on, like 'vessel_name' and 'reportdate'
    # Adjust the column names as necessary
    merged_df = pd.merge(vessel_df, consumption_df, 
                         on=['vessel_name', 'reportdate'], 
                         how='left')
    
    # Check if LATITUDE and LONGITUDE columns exist
    if 'LATITUDE' in merged_df.columns and 'LONGITUDE' in merged_df.columns:
        # Add previous latitude and longitude
        merged_df['prev_LATITUDE'] = merged_df.groupby('vessel_name')['LATITUDE'].shift(1)
        merged_df['prev_LONGITUDE'] = merged_df.groupby('vessel_name')['LONGITUDE'].shift(1)
    else:
        print("Warning: LATITUDE and/or LONGITUDE columns not found in merged dataframe")
    
    return merged_df
