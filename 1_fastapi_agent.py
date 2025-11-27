"""
Example 3: Combined MLflow + OpenTelemetry with FastAPI
This demonstrates combining MLflow's auto-tracing with OpenTelemetry's FastAPI instrumentation.
Based on: https://mlflow.org/docs/latest/genai/tracing/app-instrumentation/opentelemetry/
"""
import os
import mlflow
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from openai import OpenAI
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

from config import (
    DATABRICKS_HOST,
    DATABRICKS_TOKEN,
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME_OTEL,
    OPENAI_API_KEY,
)

# IMPORTANT: Use the OpenTelemetry tracer provider instead of MLflow's default
# This allows both SDKs to work together
os.environ["MLFLOW_USE_DEFAULT_TRACER_PROVIDER"] = "false"


# Enable MLflow OpenAI auto-tracing at application startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Configure MLflow and enable auto-tracing on startup"""
    # Configure Databricks host/token for MLflow so credentials are sent correctly
    os.environ["DATABRICKS_HOST"] = DATABRICKS_HOST
    os.environ["DATABRICKS_TOKEN"] = DATABRICKS_TOKEN

    # Use Databricks tracking URI (recommended for PAT-based auth)
    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME_OTEL)
    mlflow.openai.autolog()
    
    print("✓ MLflow configured and OpenAI auto-tracing enabled")
    print(f"✓ Tracking URI: databricks (host: {DATABRICKS_HOST})")
    print(f"✓ Experiment: {MLFLOW_EXPERIMENT_NAME_OTEL}")
    
    yield
    
    print("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="OpenTelemetry + MLflow Agent API",
    lifespan=lifespan
)

# NOTE:
# We intentionally do NOT instrument FastAPI with OpenTelemetry here,
# so that only MLflow/OpenAI completion traces appear in the Databricks UI.
# FastAPIInstrumentor.instrument_app(app)


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "ok",
        "message": "OpenTelemetry + MLflow Tracing Agent",
        "tracking_uri": MLFLOW_TRACKING_URI
    }


@app.post("/chat")
async def chat(request: Request):
    """
    Simple chat endpoint that calls OpenAI.
    Both the FastAPI request and OpenAI call will be traced.
    """
    body = await request.json()
    query = body.get("query", "Hello!")
    
    # Call OpenAI (auto-traced by MLflow)
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": query}
        ],
        max_tokens=150
    )
    
    answer = response.choices[0].message.content
    
    return {
        "query": query,
        "answer": answer,
        "model": "gpt-4o-mini",
        "tokens_used": response.usage.total_tokens
    }


@app.post("/rag/v1/answer")
async def answer_question(request: Request):
    """
    RAG-style endpoint with multiple processing steps.
    Demonstrates nested spans with custom attributes.
    """
    body = await request.json()
    query = body.get("query", "")
    
    # Step 1: Retrieve relevant documents (simulated)
    documents = [
        "MLflow is an open-source platform for ML lifecycle.",
        "OpenTelemetry provides observability for applications."
    ]

    # Step 2: Generate context from documents
    context = "\n".join(documents)

    # Step 3: Generate answer with LLM
    client = OpenAI(api_key=OPENAI_API_KEY)

    prompt = f"""Based on the following context, answer the question.
    
Context:
{context}

Question: {query}

Answer:"""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200
    )
    
    answer = response.choices[0].message.content

    return {
        "query": query,
        "answer": answer,
        "context_used": len(documents),
        "model": "gpt-4o-mini"
    }


@app.post("/agent/task")
async def execute_task(request: Request):
    """
    Complex agent endpoint that demonstrates multiple LLM calls and processing steps.
    """
    body = await request.json()
    task = body.get("task", "")
    
    results = []
    
    # Step 1: Break down the task
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    planning_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"Break down this task into 3 simple steps: {task}"
        }],
        max_tokens=150
    )
    
    steps = planning_response.choices[0].message.content
    results.append({"step": "planning", "output": steps})
    
    # Step 2: Execute the task
    execution_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"Execute this task: {task}"
        }],
        max_tokens=200
    )
    
    result = execution_response.choices[0].message.content
    results.append({"step": "execution", "output": result})
    
    return {
        "task": task,
        "results": results,
        "steps_completed": len(results)
    }


def main():
    """Run the FastAPI server"""
    import uvicorn
    
    print("=" * 60)
    print("Example 3: Combined MLflow + OpenTelemetry FastAPI Agent")
    print("=" * 60)
    print("\nStarting server...")
    print("API will be available at: http://localhost:8000")
    print("\nAvailable endpoints:")
    print("  - GET  /          - Health check")
    print("  - POST /chat      - Simple chat")
    print("  - POST /rag/v1/answer - RAG-style Q&A")
    print("  - POST /agent/task    - Complex task execution")
    print("\nDocs available at: http://localhost:8000/docs")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()

