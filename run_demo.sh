#!/bin/bash
# ==============================================================================
# CosmiClassifiers — AI-Enabled Detection of Exoplanets Demo Launcher
# TESS Exoplanet Detection & Analysis Studio Launcher
# ==============================================================================

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

# Ensure venv exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    ./venv/bin/pip install -r requirements.txt
fi

echo "========================================================================"
echo "🪐 Starting CosmiClassifiers Exoplanet Detection System"
echo "========================================================================"

# Start FastAPI backend service on port 8000
echo "[1/2] Starting FastAPI Backend REST Service on http://localhost:8000..."
./venv/bin/uvicorn backend.api:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
echo "      Backend running with PID: $BACKEND_PID"
echo "      API Swagger documentation available at: http://localhost:8000/docs"

# Trap exit to kill backend on exit
cleanup() {
    echo ""
    echo "Shutting down backend service (PID: $BACKEND_PID)..."
    kill $BACKEND_PID 2>/dev/null || true
    echo "CosmiClassifiers stopped cleanly."
}
trap cleanup EXIT INT TERM

# Start Streamlit Frontend
echo ""
echo "[2/2] Starting Streamlit Interactive Dashboard on http://localhost:8501..."
# Streamlit may ask for an email on its first launch. Answer with a blank value
# so the launcher also works in non-interactive terminals and background jobs.
printf '\n' | ./venv/bin/streamlit run app.py --server.port 8501 --server.headless false

