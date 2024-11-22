from sqlalchemy import create_engine, text
import urllib.parse
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.pool import QueuePool
import logging
from contextlib import contextmanager

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
        if cls._engine is None:
            try:
                logger.debug("Creating new database engine")
                encoded_password = urllib.parse.quote(cls.DB_CONFIG['password'])
                db_url = f"postgresql+psycopg2://{cls.DB_CONFIG['user']}:{encoded_password}@{cls.DB_CONFIG['host']}:{cls.DB_CONFIG['port']}/{cls.DB_CONFIG['database']}"
                
                # Configure engine with optimized pool settings
                cls._engine = create_engine(
                    db_url,
                    poolclass=QueuePool,
                    pool_size=5,
                    max_overflow=10,
                    pool_timeout=60,  # Increased timeout
                    pool_pre_ping=True,
                    connect_args={
                        "connect_timeout": 60,  # Connection timeout in seconds
                        "keepalives": 1,
                        "keepalives_idle": 30,
                        "keepalives_interval": 10,
                        "keepalives_count": 5
                    }
                )
                
                # Test the connection
                with cls._engine.connect() as conn:
                    conn.execute(text("SELECT 1")).fetchone()
                    conn.commit()
                logger.info("Database connection established successfully")
                
            except Exception as e:
                logger.error(f"Database connection error: {str(e)}", exc_info=True)
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
        
        conn = None
        try:
            conn = engine.connect()
            yield conn
            conn.commit()
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database operation error: {str(e)}", exc_info=True)
            raise
        finally:
            if conn:
                conn.close()

def execute_query(query, params=None):
    """Helper function to execute queries with proper connection handling"""
    try:
        with DatabaseConnection.get_connection() as conn:
            if isinstance(query, str):
                query = text(query)
            result = conn.execute(query, params or {})
            return result
    except Exception as e:
        logger.error(f"Query execution error: {str(e)}", exc_info=True)
        raise

def get_db_engine():
    """Wrapper function for backward compatibility"""
    try:
        return DatabaseConnection.get_db_engine()
    except Exception as e:
        logger.error(f"Error in get_db_engine wrapper: {str(e)}", exc_info=True)
        return None
