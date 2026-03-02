"""
FastAPI agent using only native MLflow 3.0 tracing.

No OpenTelemetry instrumentation is used in this example.
"""
import os
from contextlib import asynccontextmanager

import mlflow
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from openai import OpenAI

from config import (
    DATABRICKS_HOST,
    DATABRICKS_TOKEN,
    MLFLOW_EXPERIMENT_NAME_OTEL,
    OPENAI_API_KEY,
    UC_CATALOG_NAME,
    UC_SCHEMA_NAME,
)

# Configure Databricks connection
os.environ["DATABRICKS_HOST"] = DATABRICKS_HOST
os.environ["DATABRICKS_TOKEN"] = DATABRICKS_TOKEN

MODEL_NAME = "databricks-gpt-oss-120b"
GATEWAY_BASE_URL = "https://1444828305810485.ai-gateway.cloud.databricks.com/mlflow/v1"


def _normalize_message_content(content: object) -> str:
    """Normalize OpenAI-compatible message content to a single string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_chunks = []
        for item in content:
            if isinstance(item, dict):
                text_value = item.get("text") or item.get("content")
                if isinstance(text_value, str) and text_value:
                    text_chunks.append(text_value)
        return "\n".join(text_chunks).strip()
    return str(content) if content is not None else ""


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Configure MLflow and enable tracing on startup."""
    mlflow.set_tracking_uri("databricks")
    mlflow.tracing.set_destination(
        destination=mlflow.entities.UCSchemaLocation(
            catalog_name=UC_CATALOG_NAME,
            schema_name=UC_SCHEMA_NAME,
        )
    )
    experiment = mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME_OTEL)

    # Native MLflow OpenAI autologging (no OTEL required).
    mlflow.openai.autolog()

    print("✓ MLflow configured")
    print(f"✓ Experiment ID: {experiment.experiment_id}")
    print(f"✓ Databricks: {DATABRICKS_HOST}")
    print("✓ OpenAI autolog enabled")

    yield
    print("Shutting down...")


app = FastAPI(title="MLflow 3.0 FastAPI Agent", lifespan=lifespan)


@app.get("/")
async def root():
    return {"status": "ok"}


def _run_chat_completion(query: str) -> str:
    """Generate an answer and record clean trace previews."""
    clean_query = (query or "Hello!").strip() or "Hello!"

    # with mlflow.start_span(name="chat_completion") as root_span:
    #     root_span.set_inputs(clean_query)
    #     root_span.set_attribute("query.length", len(clean_query))
    #     root_span.set_attribute("query.word_count", len(clean_query.split()))

    client = OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=GATEWAY_BASE_URL,
    )
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": clean_query},
        ],
        max_tokens=800,
    )

    raw_content = response.choices[0].message.content
    answer = _normalize_message_content(raw_content)
    answer = answer.strip().strip('"').strip("'")
        # root_span.set_attribute("response.length", len(answer))
        # root_span.set_attribute("response.word_count", len(answer.split()))
        # root_span.set_outputs(answer)
        # MLflow docs: request/response columns are preview fields; set these explicitly.
        # mlflow.update_current_trace(request_preview=clean_query, response_preview=answer)

    return answer


@app.post("/chat")
async def chat(request: Request):
    """Accept text/plain or JSON body and return plain text."""
    query = ""
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        body = await request.json()
        if isinstance(body, dict):
            query = str(body.get("query", ""))
        elif isinstance(body, str):
            query = body
    else:
        query = (await request.body()).decode("utf-8")

    answer = _run_chat_completion(query)
    return PlainTextResponse(answer)


if __name__ == "__main__":
    import uvicorn

    print("MLflow 3.0 FastAPI Agent")
    print("Using: mlflow.start_span + mlflow.update_current_trace + mlflow.openai.autolog()")
    uvicorn.run(app, host="0.0.0.0", port=8000)
