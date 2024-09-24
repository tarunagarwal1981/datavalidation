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
    print("Columns in vessel_performance_data:", df.columns.tolist())
    return df

def fetch_sf_consumption_logs(engine):
    query = """
    SELECT *
    FROM sf_consumption_logs
    WHERE "REPORT_DATE" >= %s
    ORDER BY "REPORT_DATE";
    """
    three_months_ago = datetime.now() - timedelta(days=90)
    df = pd.read_sql_query(query, engine, params=(three_months_ago,))
    print("Columns in sf_consumption_logs:", df.columns.tolist())
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
    # Rename columns in consumption_df to match vessel_df
    consumption_df = consumption_df.rename(columns={
        'VESSEL_NAME': 'vessel_name',
        'REPORT_DATE': 'reportdate'
    })
    
    # Merge the dataframes
    merged_df = pd.merge(vessel_df, consumption_df, 
                         on=['vessel_name', 'reportdate'], 
                         how='left')
    
    # Add previous latitude and longitude if they exist
    if 'LATITUDE' in merged_df.columns and 'LONGITUDE' in merged_df.columns:
        merged_df['prev_LATITUDE'] = merged_df.groupby('vessel_name')['LATITUDE'].shift(1)
        merged_df['prev_LONGITUDE'] = merged_df.groupby('vessel_name')['LONGITUDE'].shift(1)
    
    print("Columns in merged_df:", merged_df.columns.tolist())
    return merged_df
