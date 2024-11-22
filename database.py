from sqlalchemy import create_engine, text
import urllib.parse
from sqlalchemy.exc import SQLAlchemyError
import logging
from contextlib import contextmanager

# Configure logging
logging.basicConfig(level=logging.DEBUG)
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
        """
        if cls._engine is None:
            try:
                logger.debug("Creating new database engine")
                encoded_password = urllib.parse.quote(cls.DB_CONFIG['password'])
                db_url = f"postgresql+psycopg2://{cls.DB_CONFIG['user']}:{encoded_password}@{cls.DB_CONFIG['host']}:{cls.DB_CONFIG['port']}/{cls.DB_CONFIG['database']}"
                
                cls._engine = create_engine(
                    db_url,
                    pool_pre_ping=True,
                    pool_size=5,
                    max_overflow=10,
                    pool_timeout=30,
                    connect_args={
                        "keepalives": 1,
                        "keepalives_idle": 30,
                        "keepalives_interval": 10,
                        "keepalives_count": 5
                    }
                )
                
                # Test the connection
                with cls._engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                    conn.commit()
                logger.info("Database connection established successfully")
            except Exception as e:
                logger.error(f"Failed to create database engine: {str(e)}")
                cls._engine = None
                raise
        return cls._engine

    @classmethod
    @contextmanager
    def get_connection(cls):
        """Context manager for database connections"""
        engine = cls.get_db_engine()
        if engine is None:
            raise RuntimeError("Could not establish database connection")
        
        try:
            with engine.connect() as conn:
                yield conn
                conn.commit()
        except Exception as e:
            logger.error(f"Database operation error: {str(e)}")
            raise

def get_db_engine():
    """Wrapper function for backward compatibility"""
    try:
        return DatabaseConnection.get_db_engine()
    except Exception as e:
        logger.error(f"Error in get_db_engine wrapper: {str(e)}")
        return None
