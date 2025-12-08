"""
Setup Unity Catalog Trace Table

This script configures MLflow tracing to store traces in a Unity Catalog table.
This enables you to query and analyze your traces using SQL.

Prerequisites:
1. A Databricks workspace with Unity Catalog enabled
2. A catalog and schema where you have CREATE TABLE permissions
3. A SQL warehouse you have access to

Usage:
    python 3_setup_uc_trace_table.py
"""
import os
from dotenv import load_dotenv

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.entities import UCSchemaLocation
from mlflow.tracing.enablement import set_experiment_trace_location

# Load environment variables from .env file
load_dotenv()

# ============================================================================
# Configuration - Update these values or set them in your .env file
# ============================================================================

# SQL Warehouse ID - Required for UC trace table queries
# Find this in Databricks: SQL Warehouses > Your Warehouse > Connection Details
SQL_WAREHOUSE_ID = os.getenv("MLFLOW_TRACING_SQL_WAREHOUSE_ID", "")

# Unity Catalog configuration
UC_CATALOG_NAME = os.getenv("UC_CATALOG_NAME", "")
UC_SCHEMA_NAME = os.getenv("UC_SCHEMA_NAME", "")

# Experiment name for UC-backed traces
EXPERIMENT_NAME = os.getenv(
    "MLFLOW_EXPERIMENT_NAME_OTEL",
    "/Users/your.email@company.com/uc-trace-experiment"
)


def validate_config():
    """Validate that all required configuration is set."""
    errors = []
    
    if not SQL_WAREHOUSE_ID:
        errors.append("MLFLOW_TRACING_SQL_WAREHOUSE_ID is not set")
    
    if not UC_CATALOG_NAME:
        errors.append("UC_CATALOG_NAME is not set")
    
    if not UC_SCHEMA_NAME:
        errors.append("UC_SCHEMA_NAME is not set")
    
    if "your.email" in EXPERIMENT_NAME:
        errors.append("MLFLOW_EXPERIMENT_NAME_UC needs to be set to your actual experiment path")
    
    return errors


def setup_uc_trace_table():
    """
    Set up Unity Catalog as the trace storage location for an MLflow experiment.
    
    This creates or retrieves an experiment and links it to a UC schema where
    traces will be stored as tables that can be queried with SQL.
    """
    print("=" * 60)
    print("Setting up Unity Catalog Trace Table")
    print("=" * 60)
    
    # Validate configuration
    errors = validate_config()
    if errors:
        print("\n❌ Configuration errors:")
        for error in errors:
            print(f"   - {error}")
        print("\nPlease update your .env file with the required values.")
        print("See env_template.txt for reference.")
        return None
    
    # Set the SQL warehouse ID environment variable
    os.environ["MLFLOW_TRACING_SQL_WAREHOUSE_ID"] = SQL_WAREHOUSE_ID
    
    # Configure MLflow to use Databricks
    mlflow.set_tracking_uri("databricks")
    
    print(f"\n📋 Configuration:")
    print(f"   Experiment: {EXPERIMENT_NAME}")
    print(f"   Catalog: {UC_CATALOG_NAME}")
    print(f"   Schema: {UC_SCHEMA_NAME}")
    print(f"   SQL Warehouse: {SQL_WAREHOUSE_ID}")
    
    # Create or get the experiment
    print(f"\n🔧 Creating/retrieving experiment...")
    try:
        experiment_id = mlflow.create_experiment(name=EXPERIMENT_NAME)
        print(f"   ✅ Created new experiment with ID: {experiment_id}")
    except MlflowException as e:
        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        experiment_id = experiment.experiment_id if experiment else None
        if experiment_id:
            print(f"   ✅ Using existing experiment with ID: {experiment_id}")
        else:
            print(f"   ❌ Failed to create or find experiment: {e}")
            return None
    
    # Link the experiment to the UC schema location
    print(f"\n🔗 Linking experiment to Unity Catalog schema...")
    try:
        result = set_experiment_trace_location(
            location=UCSchemaLocation(
                catalog_name=UC_CATALOG_NAME,
                schema_name=UC_SCHEMA_NAME
            ),
            experiment_id=experiment_id,
        )
        
        print(f"   ✅ Successfully linked!")
        print(f"\n📊 Trace Table Information:")
        print(f"   Full table name: {result.full_otel_spans_table_name}")
        print(f"\n💡 You can now query your traces with SQL:")
        print(f"   SELECT * FROM {result.full_otel_spans_table_name} LIMIT 10")
        
        return result
        
    except Exception as e:
        print(f"   ❌ Failed to set trace location: {e}")
        return None


if __name__ == "__main__":
    result = setup_uc_trace_table()
    
    if result:
        print("\n" + "=" * 60)
        print("✅ Setup complete! Your traces will now be stored in UC.")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ Setup failed. Please check the errors above.")
        print("=" * 60)

