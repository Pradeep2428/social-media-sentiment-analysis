#!/bin/bash
# One-click launcher for macOS / Linux. First run installs packages (needs internet).
cd "$(dirname "$0")" || exit 1
PY=$(command -v python3 || command -v python)
[ -z "$PY" ] && { echo "Python 3 is not installed. Install it from python.org, then run this again."; read -r -p "Press Enter to close"; exit 1; }
if [ ! -d .venv ]; then
  echo "First run: setting things up (a few minutes)..."
  "$PY" -m venv .venv && .venv/bin/pip install --quiet -r requirements.txt || { echo "Setup failed. Check the internet connection."; read -r -p "Press Enter to close"; rm -rf .venv; exit 1; }
fi
echo "Starting SentiLens at http://localhost:8501  (close this window to stop it)"
.venv/bin/streamlit run app.py --server.port 8501
