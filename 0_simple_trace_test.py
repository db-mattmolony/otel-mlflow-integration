"""
Example 0: Simple Trace Test (No OpenAI Required)
This is the absolute simplest test - just sends a basic trace to Databricks.
Use this to verify your OpenTelemetry -> Databricks connection works.
"""
import os
import time
import mlflow

from config import (
    DATABRICKS_HOST,
    DATABRICKS_TOKEN,
    MLFLOW_EXPERIMENT_NAME_BASIC,
)


def setup_mlflow():
    """Configure MLflow to connect to Databricks"""
    # Set environment variables BEFORE any MLflow calls
    os.environ["DATABRICKS_HOST"] = DATABRICKS_HOST
    os.environ["DATABRICKS_TOKEN"] = DATABRICKS_TOKEN
    
    # Use databricks as tracking URI instead of the full HTTPS URL
    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME_BASIC)
    
    print(f"✓ MLflow configured")
    print(f"  Tracking URI: databricks")
    print(f"  Host: {DATABRICKS_HOST}")
    print(f"  Experiment: {MLFLOW_EXPERIMENT_NAME_BASIC}")


@mlflow.trace
def simple_function(x: int, y: int) -> int:
    """A simple function that will be traced"""
    with mlflow.start_span(name="add_numbers") as span:
        span.set_attribute("x", x)
        span.set_attribute("y", y)
        time.sleep(0.1)  # Simulate some work
        result = x + y
        span.set_attribute("result", result)
    
    return result


@mlflow.trace
def multi_step_process(input_data: str) -> dict:
    """A multi-step process with nested spans"""
    
    # Step 1: Parse input
    with mlflow.start_span(name="parse_input") as span:
        span.set_attribute("input_length", len(input_data))
        time.sleep(0.05)
        parsed = input_data.split()
        span.set_attribute("word_count", len(parsed))
    
    # Step 2: Process data
    with mlflow.start_span(name="process_data") as span:
        time.sleep(0.1)
        processed = [word.upper() for word in parsed]
        span.set_attribute("processed_count", len(processed))
    
    # Step 3: Generate result
    with mlflow.start_span(name="generate_result") as span:
        result = {
            "original": input_data,
            "processed": processed,
            "word_count": len(parsed)
        }
        span.set_attribute("result_keys", list(result.keys()))
    
    return result


def main():
    """Run simple trace tests"""
    print("=" * 60)
    print("Example 0: Simple Trace Test (No LLM Required)")
    print("=" * 60)
    print("\nThis test doesn't require OpenAI - just tests the")
    print("OpenTelemetry -> Databricks connection.\n")
    
    # Setup
    setup_mlflow()
    
    # Test 1: Simple traced function
    print("\n--- Test 1: Simple traced function ---")
    try:
        result = simple_function(5, 3)
        print(f"Result: {result}")
        print("✓ Trace sent to Databricks")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test 2: Multi-step process
    print("\n--- Test 2: Multi-step process ---")
    try:
        result = multi_step_process("hello world from opentelemetry")
        print(f"Result: {result}")
        print("✓ Trace with nested spans sent to Databricks")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test 3: Multiple traces
    print("\n--- Test 3: Multiple traces ---")
    try:
        for i in range(3):
            result = simple_function(i, i + 1)
            print(f"  Trace {i+1}: {result}")
            time.sleep(0.2)
        print("✓ Multiple traces sent to Databricks")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Tests complete!")
    print("\nNext steps:")
    print("  1. Go to your Databricks workspace")
    print("  2. Navigate to: Machine Learning > Experiments")
    print(f"  3. Open: {MLFLOW_EXPERIMENT_NAME_BASIC}")
    print("  4. Click the 'Traces' tab")
    print("  5. You should see the traces from this test")
    print("\nExpected traces:")
    print("  - simple_function (x=5, y=3)")
    print("  - multi_step_process (with 3 nested spans)")
    print("  - simple_function (3 more traces)")
    print("=" * 60)


if __name__ == "__main__":
    main()

