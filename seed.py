"""
FastAPI with Explicit OpenTelemetry Tracing to Databricks

Uses OpenTelemetry TracerProvider with manual spans and MLflow spans
for Databricks export.

References:
- https://mlflow.org/docs/latest/genai/tracing/app-instrumentation/opentelemetry/
- https://docs.databricks.com/aws/en/mlflow3/genai/tracing/prod-tracing-external
"""
import os

from config import (
    DATABRICKS_HOST,
    DATABRICKS_TOKEN,
    MLFLOW_EXPERIMENT_NAME_OTEL,
    OPENAI_API_KEY,
    MLFLOW_TRACING_SQL_WAREHOUSE_ID,
    UC_CATALOG_NAME,
    UC_SCHEMA_NAME
)

# Configure Databricks connection
os.environ["DATABRICKS_HOST"] = DATABRICKS_HOST
os.environ["DATABRICKS_TOKEN"] = DATABRICKS_TOKEN
os.environ["MLFLOW_TRACKING_URI"] = "databricks"
os.environ["MLFLOW_EXPERIMENT_NAME"] = MLFLOW_EXPERIMENT_NAME_OTEL

# Tell MLflow to use our OpenTelemetry TracerProvider
os.environ["MLFLOW_USE_DEFAULT_TRACER_PROVIDER"] = "false"

# ============================================================
# OpenTelemetry Setup
# ============================================================
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.resources import Resource, SERVICE_NAME

# Create service resource
resource = Resource(attributes={
    SERVICE_NAME: "fastapi-otel-agent",
    "service.version": "1.0.0",
    "deployment.environment": "development",
})

# Set up TracerProvider
provider = TracerProvider(resource=resource)
trace.set_tracer_provider(provider)

# Get tracer for manual spans
tracer = trace.get_tracer(__name__, "1.0.0")

# ============================================================
# FastAPI + MLflow
# ============================================================
import mlflow
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from openai import OpenAI


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Configure MLflow at startup"""
    mlflow.set_tracking_uri("databricks")
    
    mlflow.tracing.set_destination(
    destination=mlflow.entities.UCSchemaLocation(
        catalog_name=UC_CATALOG_NAME,
        schema_name=UC_SCHEMA_NAME,
    )
)
    experiment = mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME_OTEL)
    
    print("✓ OpenTelemetry TracerProvider configured")
    print(f"✓ Service: fastapi-otel-agent v1.0.0")
    print(f"✓ Databricks: {DATABRICKS_HOST}")
    print(f"✓ Experiment ID: {experiment.experiment_id}")
    
    yield
    print("Shutting down...")


app = FastAPI(title="OpenTelemetry + MLflow API", lifespan=lifespan)


@app.get("/")
async def root():
    """Health check"""
    return {
        "status": "ok",
        "otel_provider": str(trace.get_tracer_provider()),
    }


def process_chat(query: str) -> str:
    """Process chat with OpenTelemetry and MLflow spans"""
    # MLflow span for Databricks - set input/output directly
    with mlflow.start_span(name="chat_completion") as mlflow_span:
        mlflow_span.set_inputs(query)
        
        # OpenTelemetry span for detailed tracing
        with tracer.start_as_current_span("openai_completion") as otel_span:
            otel_span.set_attribute("model", "gpt-4o-mini")
            
            client = OpenAI(api_key=OPENAI_API_KEY)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": query}
                ],
                max_tokens=150
            )
            
            otel_span.set_attribute("tokens", response.usage.total_tokens)
        
        answer = response.choices[0].message.content
        mlflow_span.set_outputs(answer)
    
    return answer


@app.post("/chat")
async def chat(request: Request):
    """Chat endpoint"""
    body = await request.json()
    query = body.get("query", "Hello!")
    answer = process_chat(query)
    return {
        "query": query,
        "answer": answer,
        "model": "gpt-4o-mini",
    }

if __name__ == "__main__":
    import uvicorn
    print("OpenTelemetry + MLflow FastAPI Agent")
    print("Using: TracerProvider, manual OTel spans, MLflow spans")
    uvicorn.run(app, host="0.0.0.0", port=8000)
