import os
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()

def get_db_connection_string():
    driver = os.getenv("DB_DRIVER")
    server = os.getenv("DB_SERVER")
    database = os.getenv("DB_DATABASE")
    trusted_connection = os.getenv("DB_TRUSTED_CONNECTION")

    params = (
        f"DRIVER={driver};"
        f"SERVER={server};"
        f"DATABASE={database};"
        f"Trusted_Connection={trusted_connection};"
    )
    return f"mssql+pyodbc:///?odbc_connect={params.replace(';', '%3B').replace('=', '%3D')}"

# Create the engine globally to be shared
try:
    engine = create_engine(get_db_connection_string())
except Exception as e:
    print(f"Warning: Failed to initialize database engine: {e}")
    engine = None
