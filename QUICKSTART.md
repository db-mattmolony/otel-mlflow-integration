# 🚀 Quick Start Guide

Get up and running in 5 minutes!

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Configure Credentials

Create a `.env` file (copy from `env.template`):

```bash
cp env.template .env
```

Edit `.env` and update these values:

```bash
DATABRICKS_HOST=https://your-workspace.cloud.databricks.com
DATABRICKS_TOKEN=dapi1234567890abcdef

# Experiment used by basic examples (0_simple_trace_test, etc.)
MLFLOW_EXPERIMENT_NAME_BASIC=/Users/your.email@company.com/otel-tracing-basic

# Experiment used by the FastAPI + OpenTelemetry example
MLFLOW_EXPERIMENT_NAME_OTEL=/Workspace/Users/your.email@databricks.com/otel-tracing-otel

OPENAI_API_KEY=sk-proj-your-key-here
```

### Where to get these values:

**DATABRICKS_HOST**: Your workspace URL (e.g., `https://dbc-abc123-xyz.cloud.databricks.com`)

**DATABRICKS_TOKEN**: 
1. User Settings → Developer → Access Tokens
2. Generate New Token
3. Copy the token

**OPENAI_API_KEY**: 
- Get from https://platform.openai.com/api-keys
- Only needed for Examples 1 & 3

## Step 3: Verify Setup

```bash
python setup_check.py
```

This will:
- ✅ Check your credentials
- ✅ Test Databricks connection
- ✅ Create MLflow experiment if needed
- ✅ Verify OpenAI API key

## Step 4: Run Your First Test

### Option A: Simple Test (No OpenAI needed)

```bash
python 0_simple_trace_test.py
```

Best for: Testing the basic connection without using OpenAI credits.

### Option B: MLflow Agent (OpenAI required)

```bash
python 1_basic_mlflow_agent.py
```

Best for: Testing the full MLflow auto-tracing with OpenAI.

### Option C: FastAPI Server (OpenAI required)

Terminal 1:
```bash
python 3_combined_fastapi_agent.py
```

Terminal 2:
```bash
python test_api.py
```

Best for: Testing a production-like REST API with traces.

## Step 5: View Traces in Databricks

1. Open your Databricks workspace
2. Go to **Machine Learning** → **Experiments**
3. Find your experiment (e.g., `/Users/your.email@company.com/otel-tracing-test`)
4. Click the **Traces** tab
5. 🎉 See your traces!

## What You'll See

Each trace will show:
- Span hierarchy (parent → child spans)
- Execution time for each span
- Custom attributes you added
- OpenAI API calls (if using)
- Request/response metadata

## Troubleshooting

### "403 Forbidden" error
→ Check your `DATABRICKS_TOKEN`

### "Experiment not found" error
→ Run `setup_check.py` to create it

### Traces not appearing
→ Wait 5-10 seconds and refresh
→ Make sure you're in the "Traces" tab, not "Runs"

### OpenAI errors
→ Check `OPENAI_API_KEY` is valid
→ Try Example 0 first (no OpenAI needed)

## Next Steps

- 📖 Read [README.md](README.md) for detailed documentation
- 🔍 Explore different examples (0-3)
- 🎨 Customize spans and attributes for your use case
- 🚀 Build your own traced agents!

## Examples Overview

| Example | File | OpenAI? | Best For |
|---------|------|---------|----------|
| 0 | `0_simple_trace_test.py` | ❌ | Testing connection |
| 1 | `1_basic_mlflow_agent.py` | ✅ | MLflow SDK approach |
| 2 | `2_mlflow_tracing_agent.py` | ✅ | MLflow Tracing SDK (OpenTelemetry-backed) |
| 3 | `3_combined_fastapi_agent.py` | ✅ | Production REST API |

---

**Need help?** See [README.md](README.md) for more details!

