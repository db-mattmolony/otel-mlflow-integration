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


def process_rag(query: str) -> str:
    """RAG with OpenTelemetry and MLflow spans"""
    with mlflow.start_span(name="rag_completion") as mlflow_span:
        mlflow_span.set_inputs(query)
        
        with tracer.start_as_current_span("retrieval") as span:
            documents = [
                "MLflow is an open-source platform for ML lifecycle.",
                "OpenTelemetry provides observability for applications."
            ]
            span.set_attribute("docs.count", len(documents))
        
        with tracer.start_as_current_span("build_context") as span:
            context = "\n".join(documents)
            span.set_attribute("context.length", len(context))
        
        with tracer.start_as_current_span("llm_generation") as span:
            client = OpenAI(api_key=OPENAI_API_KEY)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}],
                max_tokens=200
            )
            span.set_attribute("tokens", response.usage.total_tokens)
        
        answer = response.choices[0].message.content
        mlflow_span.set_outputs(answer)
    
    return answer


@app.post("/rag/v1/answer")
async def answer_question(request: Request):
    """RAG endpoint"""
    body = await request.json()
    query = body.get("query", "")
    answer = process_rag(query)
    return {
        "query": query,
        "answer": answer,
    }


def process_agent(task: str) -> str:
    """Agent with OpenTelemetry and MLflow spans"""
    with mlflow.start_span(name="agent_execution") as mlflow_span:
        mlflow_span.set_inputs(task)
        
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        with tracer.start_as_current_span("planning") as span:
            planning = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": f"Break down this task: {task}"}],
                max_tokens=150
            )
            span.set_attribute("tokens", planning.usage.total_tokens)
        
        with tracer.start_as_current_span("execution") as span:
            execution = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": f"Execute this task: {task}"}],
                max_tokens=200
            )
            span.set_attribute("tokens", execution.usage.total_tokens)
        
        result = execution.choices[0].message.content
        mlflow_span.set_outputs(result)
    
    return result


@app.post("/agent/task")
async def execute_task(request: Request):
    """Agent endpoint"""
    body = await request.json()
    task = body.get("task", "")
    result = process_agent(task)
    return {
        "task": task,
        "result": result,
    }


if __name__ == "__main__":
    import uvicorn
    print("OpenTelemetry + MLflow FastAPI Agent")
    print("Using: TracerProvider, manual OTel spans, MLflow spans")
    uvicorn.run(app, host="0.0.0.0", port=8000)
