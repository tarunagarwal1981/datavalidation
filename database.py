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
    WITH ranked_logs AS (
        SELECT 
            vps.*, 
            vp.vessel_type,
            sl.LATITUDE,
            sl.LONGITUDE,
            ROW_NUMBER() OVER (PARTITION BY vps.vessel_name ORDER BY vps.reportdate) as row_num
        FROM vessel_performance_summary vps
        LEFT JOIN vessel_particulars vp ON vps.vessel_name = vp.vessel_name
        LEFT JOIN sf_consumption_logs sl ON vps.vessel_name = sl.VESSEL_NAME AND vps.reportdate = sl.reportdate
        WHERE vps.reportdate >= %s
    )
    SELECT 
        r1.*,
        r2.LATITUDE as prev_LATITUDE,
        r2.LONGITUDE as prev_LONGITUDE
    FROM ranked_logs r1
    LEFT JOIN ranked_logs r2 ON r1.vessel_name = r2.vessel_name AND r1.row_num = r2.row_num + 1
    ORDER BY r1.vessel_name, r1.reportdate;
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

def fetch_sf_consumption_logs(engine):
    query = """
    SELECT 
        VESSEL_NAME,
        reportdate,
        LATITUDE,
        LONGITUDE
    FROM sf_consumption_logs
    WHERE reportdate >= %s
    ORDER BY VESSEL_NAME, reportdate;
    """
    three_months_ago = datetime.now() - timedelta(days=90)
    df = pd.read_sql_query(query, engine, params=(three_months_ago,))
    return df
