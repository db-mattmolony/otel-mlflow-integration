"""
Run a small suite of Databricks-related questions against the FastAPI app.

How to use (two terminals):
1. Terminal 1 – start the FastAPI server:
   - Activate your venv (if needed)
   - Run:  python 1_combined_fastapi_agent.py
   - Leave this terminal running so the API is available at http://localhost:8000

2. Terminal 2 – run this test script:
   - Activate the same venv (if needed)
   - Run:  python run_databricks_chat_tests.py
   - This will call the /chat endpoint several times with Databricks-related questions
   - Then you can inspect traces in your Databricks MLflow experiment (Traces tab)

This script exercises only:
- /chat – simple Q&A about Databricks / MLflow / OpenTelemetry
"""

import json
from typing import List

import httpx


BASE_URL = "http://localhost:8000"


CHAT_QUERIES = [
    "How does Databricks AutoML accelerate model development for both novice and experienced data scientists?",
    "What types of models can Databricks AutoML automatically train and tune?",
    "How does AutoML generate transparent, production-ready notebooks for further customisation?",
    "What features of AutoML help ensure reproducible and governed ML workflows within Unity Catalog?",
    "How does AutoML evaluate competing models and select the best candidate for deployment?",
]


def pretty_print_response(title: str, response: httpx.Response) -> None:
    """Print a compact summary of a JSON response."""
    print(f"\n=== {title} ===")
    print(f"Status: {response.status_code}")
    try:
        data = response.json()
    except json.JSONDecodeError:
        print("Non-JSON response:")
        print(response.text[:500])
        return

    # Print only key fields to keep output manageable
    if "answer" in data:
        print("Answer:")
        print(data["answer"])
    elif "results" in data:
        print("Results:")
        for i, step in enumerate(data["results"], 1):
            print(f"  Step {i} ({step.get('step')}):")
            print(f"    {step.get('output')}")
    else:
        print("Response JSON:")
        print(json.dumps(data, indent=2)[:1000])


def main() -> None:
    print("=" * 60)
    print("Running Databricks chat test suite against FastAPI app")
    print("=" * 60)

    with httpx.Client(timeout=30.0) as client:
        # Health check
        try:
            health = client.get(f"{BASE_URL}/")
            print("\n--- Health check ---")
            print(f"Status: {health.status_code}")
            print(health.text[:200])
        except Exception as e:
            print(f"Health check failed: {e}")
            return

        # /chat tests only
        for i, query in enumerate(CHAT_QUERIES, 1):
            try:
                resp = client.post(
                    f"{BASE_URL}/chat",
                    json={"query": query},
                )
                pretty_print_response(f"Chat {i}: {query}", resp)
            except Exception as e:
                print(f"Chat {i} failed: {e}")

    print("\n" + "=" * 60)
    print("Databricks chat test suite complete.")
    print("Check your Databricks MLflow experiment's Traces tab to inspect spans.")
    print("=" * 60)


if __name__ == "__main__":
    main()


