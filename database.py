from sqlalchemy import create_engine, text
import urllib.parse
from sqlalchemy.exc import SQLAlchemyError, OperationalError
import logging
from contextlib import contextmanager
import time
from tenacity import retry, stop_after_attempt, wait_exponential

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
    def create_engine(cls):
        """Create a new database engine with retry logic"""
        try:
            encoded_password = urllib.parse.quote(cls.DB_CONFIG['password'])
            db_url = f"postgresql+psycopg2://{cls.DB_CONFIG['user']}:{encoded_password}@{cls.DB_CONFIG['host']}:{cls.DB_CONFIG['port']}/{cls.DB_CONFIG['database']}"
            
            return create_engine(
                db_url,
                pool_size=5,
                max_overflow=10,
                pool_timeout=30,
                pool_pre_ping=True,
                pool_recycle=3600,
                connect_args={
                    "keepalives": 1,
                    "keepalives_idle": 30,
                    "keepalives_interval": 10,
                    "keepalives_count": 5,
                    "connect_timeout": 10
                }
            )
        except Exception as e:
            logger.error(f"Error creating engine: {str(e)}")
            raise

    @classmethod
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def get_db_engine(cls):
        """Get database engine with retry logic"""
        if cls._engine is None or not cls.test_connection(cls._engine):
            logger.debug("Creating new database engine")
            cls._engine = cls.create_engine()
            if not cls.test_connection(cls._engine):
                cls._engine = None
                raise OperationalError("Failed to establish database connection")
        return cls._engine

    @staticmethod
    def test_connection(engine):
        """Test if the database connection is working"""
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1")).fetchone()
                return True
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            return False

    @classmethod
    @contextmanager
    def get_connection(cls):
        """Get a database connection with automatic retry"""
        for attempt in range(3):
            try:
                engine = cls.get_db_engine()
                conn = engine.connect()
                try:
                    yield conn
                    conn.commit()
                except Exception as e:
                    conn.rollback()
                    raise
                finally:
                    conn.close()
                break
            except Exception as e:
                logger.error(f"Database connection error (attempt {attempt + 1}): {str(e)}")
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)

def get_db_engine():
    """Wrapper function for backward compatibility"""
    try:
        engine = DatabaseConnection.get_db_engine()
        # Test the connection immediately
        if DatabaseConnection.test_connection(engine):
            logger.info("Database connection established successfully")
            return engine
        logger.error("Database connection test failed")
        return None
    except Exception as e:
        logger.error(f"Error in get_db_engine wrapper: {str(e)}")
        return None

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def execute_query(query, params=None):
    """Execute a database query with retry logic"""
    try:
        with DatabaseConnection.get_connection() as conn:
            if isinstance(query, str):
                query = text(query)
            result = conn.execute(query, params or {})
            return result
    except Exception as e:
        logger.error(f"Query execution error: {str(e)}")
        raise
