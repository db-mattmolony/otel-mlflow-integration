"""
Configuration for OpenTelemetry MLflow Integration Demo
Copy this file and set your actual credentials.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Databricks Configuration
DATABRICKS_HOST = os.getenv("DATABRICKS_HOST", "https://your-workspace.cloud.databricks.com")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN", "")


# MLflow Configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", DATABRICKS_HOST)

# Separate experiment names for basic scripts vs FastAPI+OTel example
MLFLOW_EXPERIMENT_NAME_BASIC = os.getenv(
    "MLFLOW_EXPERIMENT_NAME_BASIC",
    "/Users/your.email@company.com/otel-tracing-basic",
)
MLFLOW_EXPERIMENT_NAME_OTEL = os.getenv(
    "MLFLOW_EXPERIMENT_NAME_OTEL",
    "/Workspace/Users/your.email@databricks.com/otel-tracing-otel",
)


MLFLOW_TRACING_SQL_WAREHOUSE_ID = os.getenv(
    "MLFLOW_TRACING_SQL_WAREHOUSE_ID",
    "tttttttt336",
)
UC_CATALOG_NAME = os.getenv(
    "UC_CATALOG_NAME",
    "my_catalog",
)
UC_SCHEMA_NAME = os.getenv(
    "UC_SCHEMA_NAME",
    "my_schema",
)


# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Validate required configuration
def validate_config():
    """Validate that required configuration is set"""
    errors = []
    
    if not DATABRICKS_TOKEN:
        errors.append("DATABRICKS_TOKEN is not set")
    
    if "your-workspace" in DATABRICKS_HOST:
        errors.append("DATABRICKS_HOST needs to be set to your actual workspace")
    
    if not OPENAI_API_KEY:
        errors.append("OPENAI_API_KEY is not set (only needed for OpenAI examples)")
    
    return errors



