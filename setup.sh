#!/bin/bash
# Setup script for OpenTelemetry + MLflow Integration

set -e

echo "=================================================="
echo "Setting up OpenTelemetry + MLflow Integration"
echo "=================================================="

# Check if venv exists
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

echo ""
echo "Activating virtual environment..."
source venv/bin/activate

echo ""
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "=================================================="
echo "✅ Setup complete!"
echo "=================================================="
echo ""
echo "To use the virtual environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "Then you can run the examples:"
echo "  python 0_simple_trace_test.py"
echo "  python 1_basic_mlflow_agent.py"
echo "  python 3_combined_fastapi_agent.py"
echo ""
echo "To deactivate the virtual environment:"
echo "  deactivate"
echo "=================================================="

