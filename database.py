from sqlalchemy import create_engine
import pandas as pd
from datetime import datetime, timedelta
import urllib.parse
from app.config import DB_CONFIG, COLUMN_NAMES

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
    WHERE vps.reportdate >= %s;
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
    query = f"""
    SELECT vessel_name, {COLUMN_NAMES['HULL_PERFORMANCE']}
    FROM hull_performance_six_months;
    """
    return pd.read_sql_query(query, engine)
