# OpenTelemetry + MLflow Integration Demo

This project demonstrates how to integrate OpenTelemetry tracing with MLflow and send traces to Databricks using a simple, end-to-end workflow. The goal is to:

- **Step 0**: Establish a working connection to Databricks with MLflow tracing
- **Step 1**: Deploy a FastAPI app that emits OpenTelemetry/MLflow trace data for LLM calls
- **Step 2**: Run Databricks-related chat messages through the traced FastAPI endpoint and inspect what they look like in MLflow Experiments.

## 📋 Overview

This repo is organized as a three-step workflow:

1. **Step 0 – Simple MLflow trace test** (`0_simple_trace_test.py`)  
   Verifies that your local environment can send basic MLflow traces to Databricks.
2. **Step 1 – FastAPI + MLflow tracing agent** (`1_fastapi_agent.py`)  
   Runs a FastAPI app that uses MLflow OpenAI auto-tracing to capture LLM calls as traces.
3. **Step 2 – Databricks chat test suite** (`2_run_databricks_chat_tests.py`)  
   Sends a small suite of Databricks-related questions through the `/chat` endpoint so you can see how real conversations show up in the MLflow Experiments “Traces” tab.

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Databricks workspace with MLflow
- OpenAI API key (for the examples)
- Databricks personal access token

### Installation

1. **Clone or navigate to this directory:**
```bash
cd otel-mlflow-integration
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Configure your credentials:**

Create a `.env` file in the project root:

```bash
# Databricks Configuration
DATABRICKS_HOST=https://your-workspace.cloud.databricks.com
DATABRICKS_TOKEN=dapi1234567890abcdef

# MLflow Configuration  
MLFLOW_TRACKING_URI=${DATABRICKS_HOST}

# Experiment used by basic examples (0_simple_trace_test, etc.)
MLFLOW_EXPERIMENT_NAME_BASIC=/Users/your.email@company.com/otel-tracing-basic

# Experiment used by the FastAPI + OpenTelemetry example
MLFLOW_EXPERIMENT_NAME_OTEL=/Workspace/Users/your.email@databricks.com/otel-tracing-otel

# OpenAI Configuration
OPENAI_API_KEY=sk-proj-...
```

**Important:** Replace the placeholder values with your actual credentials.

## 📝 Workflow Steps

### Step 0: Simple MLflow trace test (no OpenAI required)

This is the absolute simplest test – it just sends a few basic traces to Databricks so you can confirm that MLflow is correctly configured and your Databricks experiment is receiving data.

```bash
python 0_simple_trace_test.py
```

**What it does:**
- Configures MLflow to talk to Databricks using your host and token
- Uses `@mlflow.trace` and nested spans to create a few toy traces
- Sends multiple traces into the experiment configured by `MLFLOW_EXPERIMENT_NAME_BASIC`
- Prints next steps so you can find the traces in the Databricks UI

**Best for:**
- ✅ Verifying connectivity to Databricks + MLflow
- ✅ Understanding what a very simple trace looks like
- ✅ Running without using any OpenAI tokens

### Step 1: FastAPI + MLflow tracing agent

Runs a FastAPI app that uses MLflow’s OpenAI auto-tracing to record LLM calls as traces in Databricks.

**Start the server:**

```bash
python 1_fastapi_agent.py
```

The API will be available at `http://localhost:8000`.

**What it does:**
- Configures MLflow to use the Databricks tracking server
- Enables `mlflow.openai.autolog()` so OpenAI chat completions are traced automatically
- Exposes several endpoints (`/`, `/chat`, `/rag/v1/answer`, `/agent/task`) that each generate rich trace data

**Best for:**
- ✅ Seeing end-to-end traces for real LLM-powered API calls
- ✅ Exploring how different endpoints generate different span structures
- ✅ Providing a foundation you can adapt for your own traced FastAPI services

### Step 2: Run Databricks chat tests through the trace

Use this script to send a small suite of Databricks-related questions through the `/chat` endpoint and then inspect the resulting traces in MLflow Experiments.

In one terminal, make sure the FastAPI app is running:

```bash
python 1_fastapi_agent.py
```

In another terminal, run the chat test suite:

```bash
python 2_run_databricks_chat_tests.py
```

**What it does:**
- Calls the FastAPI health-check endpoint to verify the server is up
- Sends several Databricks/MLflow/OpenTelemetry questions to `/chat`
- Prints concise summaries of the responses
- Produces a set of realistic, comparable traces you can explore in the Databricks Experiments “Traces” tab

**Or, use curl manually (against the running FastAPI app):**

```bash
# Health check
curl http://localhost:8000/

# Simple chat
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is OpenTelemetry?"}'

# RAG-style Q&A
curl -X POST http://localhost:8000/rag/v1/answer \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain MLflow tracing"}'

# Complex agent task
curl -X POST http://localhost:8000/agent/task \
  -H "Content-Type: application/json" \
  -d '{"task": "Analyze distributed tracing benefits"}'
```

**Key Features:**
- ✅ REST API endpoints for simple chat, RAG-style Q&A, and multi-step agent tasks
- ✅ MLflow OpenAI auto-tracing for every LLM call
- ✅ Custom nested spans that reflect multi-step workflows
- ✅ Interactive API docs at `/docs`

## 🔍 Viewing Traces in Databricks

After running the examples, view your traces in Databricks:

1. Navigate to your Databricks workspace
2. Go to **Machine Learning** → **Experiments**
3. Find your experiment (e.g., `/Users/your.email@company.com/otel-tracing-test`)
4. Click on **Traces** tab
5. You should see traces with:
   - FastAPI request spans (Example 3)
   - OpenAI API call spans
   - Custom processing spans
   - Span attributes and metadata

## 🏗️ Architecture

### MLflow Tracing SDK Approach (Example 1)

```
Your Application
    ↓
MLflow SDK (built on OpenTelemetry)
    ↓
Databricks MLflow
```

### Pure OpenTelemetry Approach (Example 2)

```
Your Application
    ↓
OpenTelemetry SDK
    ↓
OTLP HTTP Exporter
    ↓
Databricks /v1/traces endpoint
```

### Combined Approach (Example 3)

```
FastAPI Application
    ↓
OTel FastAPI Instrumentation ──┐
MLflow OpenAI Auto-tracing ────┤
    ↓                          │
Unified OpenTelemetry Provider │
    ↓                          │
Databricks MLflow ←────────────┘
```

## 🔧 Configuration Details

### Key Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `DATABRICKS_HOST` | Your Databricks workspace URL | Yes |
| `DATABRICKS_TOKEN` | Databricks personal access token | Yes |
| `MLFLOW_TRACKING_URI` | MLflow tracking server URL | Yes |
| `MLFLOW_EXPERIMENT_NAME_BASIC` | MLflow experiment path for basic examples | Yes (for examples 0–2) |
| `MLFLOW_EXPERIMENT_NAME_OTEL` | MLflow experiment path for FastAPI + OTEL | Yes (for FastAPI example) |
| `OPENAI_API_KEY` | OpenAI API key | Yes (for examples) |
| `MLFLOW_USE_DEFAULT_TRACER_PROVIDER` | Set to `false` for combined approach | Only for Example 3 |

### Getting a Databricks Token

1. Log in to your Databricks workspace
2. Click your username in the top right
3. Select **User Settings**
4. Click **Developer** → **Access tokens**
5. Click **Generate new token**
6. Copy and save the token securely

### Creating an MLflow Experiment

```python
import mlflow
mlflow.set_tracking_uri("https://your-workspace.cloud.databricks.com")
mlflow.create_experiment("/Users/your.email@company.com/otel-tracing-test")
```

Or create via Databricks UI:
1. Go to **Machine Learning** → **Experiments**
2. Click **Create Experiment**
3. Name it and note the experiment ID

## 📊 Trace Structure Examples

### Simple Trace (Example 1)
```
agent_execution
├── preprocess
├── llm_call (OpenAI)
└── postprocess
```

### Complex Trace (Example 3 - RAG endpoint)
```
HTTP POST /rag/v1/answer
├── retrieve_documents
├── generate_context
└── generate_answer
    └── OpenAI API call
```

## 🐛 Troubleshooting

### Common Issues

**1. Authentication Error**
```
Error: 403 Forbidden
```
- Check your `DATABRICKS_TOKEN` is valid
- Ensure token has permission to access MLflow

**2. Experiment Not Found**
```
Error: Experiment not found
```
- Create the experiment first in Databricks
- Update `MLFLOW_EXPERIMENT_NAME_BASIC` / `MLFLOW_EXPERIMENT_NAME_OTEL` to match exact paths

**3. Traces Not Appearing**
```
No traces visible in Databricks
```
- Check the **Traces** tab (not Runs)
- Wait a few seconds for traces to propagate
- Verify `DATABRICKS_HOST` is correct
- For Example 2, ensure experiment ID is valid

**3a. "Failed to export span batch code: 404" (Example 2)**
```
Failed to export span batch code: 404, reason: 
```
- ✅ This is a **harmless warning** from OpenTelemetry's cleanup process
- Your traces ARE sent successfully during execution
- The 404 only occurs during final cleanup/shutdown
- You can safely ignore this message
- Check Databricks - your traces will be there!

**4. OpenAI API Error**
```
Error: OpenAI API key invalid
```
- Set valid `OPENAI_API_KEY` in your `.env` file
- Check OpenAI account has credits

**5. Import Error**
```
ModuleNotFoundError: No module named 'opentelemetry'
```
- Run `pip install -r requirements.txt`
- Ensure you're using the correct Python environment

## 🎯 Best Practices

1. **Use MLflow SDK for Python applications** - Simplest approach with auto-tracing
2. **Use pure OpenTelemetry for other languages** - Java, Go, Rust, etc.
3. **Combine approaches for web frameworks** - Get both framework and LLM traces
4. **Add custom attributes** - Enrich traces with business context
5. **Use descriptive span names** - Makes traces easier to understand
6. **Set proper experiment IDs** - Organize traces by project

## 🔗 Additional Resources

- [MLflow OpenTelemetry Documentation](https://mlflow.org/docs/latest/genai/tracing/app-instrumentation/opentelemetry/)
- [MLflow Tracing Guide](https://mlflow.org/docs/latest/genai/tracing/)
- [OpenTelemetry Documentation](https://opentelemetry.io/docs/)
- [Databricks MLflow Documentation](https://docs.databricks.com/mlflow/index.html)

## 📄 License

This is a demonstration project. Feel free to use and modify as needed.

## 🤝 Contributing

This is a learning/demo project. Feel free to extend with additional examples:
- Other LLM providers (Anthropic, Cohere, etc.)
- Different frameworks (LangChain, LlamaIndex)
- Other languages (Java, Go)
- Production deployment patterns

## 📞 Support

For issues with:
- **MLflow/Databricks**: Check Databricks documentation or support
- **OpenTelemetry**: See OpenTelemetry community resources
- **This demo**: Review code comments and troubleshooting section above

---

**Happy Tracing! 🎉**

