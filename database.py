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
    WHERE "REPORT_DATE" >= %s
    ORDER BY "REPORT_DATE";
    """
    three_months_ago = datetime.now() - timedelta(days=90)
    df = pd.read_sql_query(query, engine, params=(three_months_ago,))
    
    # Print column names for debugging
    print("Columns in sf_consumption_logs:", df.columns.tolist())
    
    # Rename columns to match expected names (case-insensitive)
    column_mapping = {
        'vessel_name': 'vessel_name',
        'report_date': 'reportdate',
        'latitude': 'latitude',
        'longitude': 'longitude'
    }
    df.columns = [col.lower() for col in df.columns]
    df = df.rename(columns=column_mapping)
    
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
    print("Columns in vessel_df:", vessel_df.columns.tolist())
    print("Columns in consumption_df:", consumption_df.columns.tolist())
    
    # Merge the dataframes
    merged_df = pd.merge(vessel_df, consumption_df, 
                         on=['vessel_name', 'reportdate'], 
                         how='left')
    
    # Add previous latitude and longitude
    if 'latitude' in merged_df.columns and 'longitude' in merged_df.columns:
        merged_df['prev_latitude'] = merged_df.groupby('vessel_name')['latitude'].shift(1)
        merged_df['prev_longitude'] = merged_df.groupby('vessel_name')['longitude'].shift(1)
    else:
        print("Warning: 'latitude' and/or 'longitude' columns not found in merged dataframe")
    
    return merged_df
