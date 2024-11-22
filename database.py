from sqlalchemy import create_engine
import urllib.parse
from sqlalchemy.exc import SQLAlchemyError
import logging
from contextlib import contextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseConnection:
    DB_CONFIG = {
        'host': 'aws-0-ap-south-1.pooler.supabase.com',
        'database': 'postgres',
        'user': 'postgres.conrxbcvuogbzfysomov',
        'password': 'wXAryCC8@iwNvj#',
        'port': '6543'
    }

    _engine = None

    @classmethod
    def get_db_engine(cls):
        """
        Creates and returns a SQLAlchemy engine instance.
        Returns None if connection fails.
        """
        if cls._engine is None:
            try:
                encoded_password = urllib.parse.quote(cls.DB_CONFIG['password'])
                db_url = f"postgresql+psycopg2://{cls.DB_CONFIG['user']}:{encoded_password}@{cls.DB_CONFIG['host']}:{cls.DB_CONFIG['port']}/{cls.DB_CONFIG['database']}"
                cls._engine = create_engine(db_url, pool_pre_ping=True)
                # Test the connection
                with cls._engine.connect() as conn:
                    conn.execute("SELECT 1")
                logger.info("Database connection established successfully")
            except SQLAlchemyError as e:
                logger.error(f"Failed to create database engine: {str(e)}")
                cls._engine = None
                raise
        return cls._engine

def get_db_engine():
    """
    Wrapper function for backward compatibility with existing validators
    """
    return DatabaseConnection.get_db_engine()
