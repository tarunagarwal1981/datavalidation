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
        Returns None if connection fails.
        """
        if cls._engine is None:
            try:
                logger.debug("Creating new database engine")
                encoded_password = urllib.parse.quote(cls.DB_CONFIG['password'])
                db_url = f"postgresql+psycopg2://{cls.DB_CONFIG['user']}:{encoded_password}@{cls.DB_CONFIG['host']}:{cls.DB_CONFIG['port']}/{cls.DB_CONFIG['database']}"
                
                # Create engine with echo for debugging
                cls._engine = create_engine(
                    db_url,
                    pool_pre_ping=True,
                    pool_size=5,
                    max_overflow=10,
                    pool_timeout=30,
                    echo=True
                )
                
                # Test the connection
                logger.debug("Testing database connection")
                with cls._engine.connect() as conn:
                    result = conn.execute(text("SELECT 1"))
                    result.fetchone()
                    conn.commit()
                logger.info("Database connection established successfully")
                
            except SQLAlchemyError as e:
                logger.error(f"Failed to create database engine: {str(e)}", exc_info=True)
                cls._engine = None
                raise
            except Exception as e:
                logger.error(f"Unexpected error in get_db_engine: {str(e)}", exc_info=True)
                cls._engine = None
                raise
                
        return cls._engine

def get_db_engine():
    """
    Wrapper function for backward compatibility with existing validators
    """
    try:
        return DatabaseConnection.get_db_engine()
    except Exception as e:
        logger.error(f"Error in get_db_engine wrapper: {str(e)}", exc_info=True)
        return None
