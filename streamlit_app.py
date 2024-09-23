from pyspark.sql import DataFrame
from pyspark.sql.types import DecimalType
from sqlalchemy import create_engine
import pandas as pd

# Supabase connection details
supabase_password = "wXAryCC8%40iwNvj%23"  # URL-encoded password
supabase_url = f"postgresql://postgres.conrxbcvuogbzfysomov:{supabase_password}@aws-0-ap-south-1.pooler.supabase.com:6543/postgres"

# Create a SQLAlchemy engine to connect to Supabase
engine = create_engine(supabase_url)

def convert_decimal_columns_to_float(spark_df: DataFrame) -> DataFrame:
    """
    Converts all DecimalType columns in a Spark DataFrame to FloatType to optimize conversion to Pandas DataFrame.
    
    Parameters:
    spark_df (DataFrame): The input Spark DataFrame
    
    Returns:
    DataFrame: The Spark DataFrame with all DecimalType columns converted to FloatType
    """
    for col_name, dtype in spark_df.dtypes:
        if dtype.startswith('decimal'):
            print(f"Converting column {col_name} from DecimalType to FloatType")
            spark_df = spark_df.withColumn(col_name, spark_df[col_name].cast("float"))
    
    return spark_df

def push_dataframe_to_supabase(dataframe: DataFrame, table_name: str):
    """
    Converts a Spark DataFrame to Pandas and pushes it to a Supabase table.
    
    Parameters:
    dataframe (DataFrame): The Spark DataFrame
    table_name (str): The destination table name in Supabase
    """
    # Convert DecimalType columns to Float
    dataframe = convert_decimal_columns_to_float(dataframe)
    
    # Convert Spark DataFrame to Pandas DataFrame
    pandas_df = dataframe.toPandas()

    # Push Pandas DataFrame to Supabase
    pandas_df.to_sql(table_name, engine, if_exists="replace", index=False)
    print(f"Table '{table_name}' pushed to Supabase successfully.")

# Retry mechanism in case of Spark driver restarts
def process_with_retry(sql_query: str, table_name: str, retries=3):
    """
    Execute a Spark SQL query, retrying in case of Spark driver restarts.
    
    Parameters:
    sql_query (str): The SQL query to execute
    table_name (str): The table name for Supabase push
    retries (int): Number of retries before giving up
    """
    attempt = 0
    while attempt < retries:
        try:
            # Execute the SQL query and load the DataFrame
            spark_df = spark.sql(sql_query)
            
            # Push the DataFrame to Supabase
            push_dataframe_to_supabase(spark_df, table_name)
            
            # Exit the loop if successful
            break
        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {str(e)}")
            attempt += 1
            if attempt < retries:
                print(f"Retrying... (Attempt {attempt + 1}/{retries})")
            else:
                print("Max retries reached. Exiting.")

# Queries and tables
queries_and_tables = [
    ("SELECT * FROM reporting_layer.digital_desk.hull_performance", "hull_performance"),
    ("SELECT * FROM reporting_layer.digital_desk.hull_performance_six_months", "hull_performance_six_months"),
    ("SELECT * FROM reporting_layer.digital_desk.vessel_performance_coefficients", "vessel_performance_coefficients"),
    ("SELECT * FROM reporting_layer.digital_desk.vessel_performance_summary", "vessel_performance_summary"),
    ("SELECT * FROM reporting_layer.vessel_datahub.vessel_particulars", "vessel_particulars")
]

# Function to add vessel_name to vessel_performance_summary
def add_vessel_name(spark_df: DataFrame, vessel_particulars_df: DataFrame) -> DataFrame:
    """
    Adds the vessel_name column to the vessel_performance_summary table by joining with vessel_particulars.
    
    Parameters:
    spark_df (DataFrame): The Spark DataFrame for vessel_performance_summary
    vessel_particulars_df (DataFrame): The Spark DataFrame for vessel_particulars
    
    Returns:
    DataFrame: The updated DataFrame with the vessel_name column
    """
    # Join the vessel_performance_summary with vessel_particulars on vessel_imo
    updated_df = spark_df.join(vessel_particulars_df, spark_df["vessel_imo"] == vessel_particulars_df["vessel_imo"], "left_outer") \
                         .select(spark_df["*"], vessel_particulars_df["vessel_name"])
    
    return updated_df

# Modify the process_with_retry function to handle vessel_name addition
def process_with_vessel_name(sql_query: str, table_name: str, vessel_particulars_df: DataFrame, retries=3):
    """
    Execute a Spark SQL query, add vessel_name by joining with vessel_particulars, and push the result to Supabase.
    
    Parameters:
    sql_query (str): The SQL query to execute
    table_name (str): The table name for Supabase push
    vessel_particulars_df (DataFrame): The Spark DataFrame for vessel_particulars to add vessel_name
    retries (int): Number of retries before giving up
    """
    attempt = 0
    while attempt < retries:
        try:
            # Execute the SQL query and load the DataFrame
            spark_df = spark.sql(sql_query)
            
            # If the table is vessel_performance_summary, add the vessel_name
            if table_name == "vessel_performance_summary":
                spark_df = add_vessel_name(spark_df, vessel_particulars_df)
            
            # Push the updated DataFrame to Supabase
            push_dataframe_to_supabase(spark_df, table_name)
            
            # Exit the loop if successful
            break
        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {str(e)}")
            attempt += 1
            if attempt < retries:
                print(f"Retrying... (Attempt {attempt + 1}/{retries})")
            else:
                print("Max retries reached. Exiting.")

# Load the vessel_particulars table once and reuse it
vessel_particulars_df = spark.sql("SELECT * FROM reporting_layer.vessel_datahub.vessel_particulars")

# Process each table, passing the vessel_particulars_df for the vessel_performance_summary table
for query, table in queries_and_tables:
    process_with_vessel_name(query, table, vessel_particulars_df)
