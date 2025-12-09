"""
FastAPI with OpenTelemetry + MLflow Tracing to Databricks

Combines OpenTelemetry SDK with MLflow Tracing SDK per:
https://mlflow.org/docs/latest/genai/tracing/app-instrumentation/opentelemetry/
"""
import os

from config import (
    DATABRICKS_HOST,
    DATABRICKS_TOKEN,
    MLFLOW_EXPERIMENT_NAME_OTEL,
    OPENAI_API_KEY,
    UC_CATALOG_NAME,
    UC_SCHEMA_NAME
)

# Configure Databricks connection
os.environ["DATABRICKS_HOST"] = DATABRICKS_HOST
os.environ["DATABRICKS_TOKEN"] = DATABRICKS_TOKEN

# Use the OpenTelemetry tracer provider instead of MLflow's default tracer provider
os.environ["MLFLOW_USE_DEFAULT_TRACER_PROVIDER"] = "false"

# ============================================================
# Imports
# ============================================================
import mlflow
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from openai import OpenAI
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

# Get OTEL tracer for manual spans
tracer = trace.get_tracer(__name__, "1.0.0")


# ============================================================
# App Lifespan - Configure MLflow at startup
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Configure MLflow and enable auto-tracing at startup"""
    mlflow.set_tracking_uri("databricks")
    mlflow.tracing.set_destination(
        destination=mlflow.entities.UCSchemaLocation(
            catalog_name=UC_CATALOG_NAME,
            schema_name=UC_SCHEMA_NAME,
        )
    )
    experiment = mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME_OTEL)
    
    # Enable MLflow OpenAI auto-tracing - captures model, tokens, etc. automatically
    mlflow.openai.autolog()
    
    print("✓ MLflow configured")
    print(f"✓ Experiment ID: {experiment.experiment_id}")
    print(f"✓ Databricks: {DATABRICKS_HOST}")
    print("✓ OpenAI autolog enabled")
    
    yield
    print("Shutting down...")


# ============================================================
# FastAPI App
# ============================================================
app = FastAPI(title="OpenTelemetry + MLflow API", lifespan=lifespan)

# Enable FastAPI auto-instrumentation - creates OTEL span for each endpoint
FastAPIInstrumentor.instrument_app(app)


@app.get("/")
async def root():
    """Health check"""
    return {"status": "ok"}


@app.post("/chat")
@mlflow.trace
async def chat(request: Request):
    """Chat endpoint with combined OTEL + MLflow tracing"""
    body = await request.json()
    query = body.get("query", "Hello!")
    
    # OTEL Signal 1: Custom span for query preprocessing
    with tracer.start_as_current_span("query_preprocessing") as prep_span:
        prep_span.set_attribute("query.original", query)
        prep_span.set_attribute("query.length", len(query))
        prep_span.set_attribute("query.word_count", len(query.split()))
        
        # Add an OTEL event to mark preprocessing complete
        prep_span.add_event("preprocessing_complete", {
            "sanitized": True,
            "language_detected": "en"
        })
    
    # OpenAI call - automatically traced by mlflow.openai.autolog()
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": query}
        ],
        max_tokens=200
    )
    
    answer = response.choices[0].message.content
    
    # OTEL Signal 2: Custom span for response postprocessing
    with tracer.start_as_current_span("response_postprocessing") as post_span:
        post_span.set_attribute("response.length", len(answer))
        post_span.set_attribute("response.word_count", len(answer.split()))
        
        # Add event with timing info
        post_span.add_event("postprocessing_complete", {
            "truncated": False,
            "formatted": True
        })
    
    return {
        "query": query,
        "answer": answer,
        "model": "gpt-4o-mini",
    }


if __name__ == "__main__":
    import uvicorn
    print("OpenTelemetry + MLflow FastAPI Agent")
    print("Using: FastAPIInstrumentor + mlflow.openai.autolog()")
    uvicorn.run(app, host="0.0.0.0", port=8000)
