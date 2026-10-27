@echo off
rem One-click launcher for Windows. First run installs packages (needs internet).
cd /d "%~dp0"
where python >nul 2>nul || (echo Python is not installed. Install it from python.org and tick "Add Python to PATH". & pause & exit /b 1)
if not exist .venv (
  echo First run: setting things up, this takes a few minutes...
  python -m venv .venv && .venv\Scripts\pip install --quiet -r requirements.txt || (echo Setup failed. Check the internet connection. & rmdir /s /q .venv & pause & exit /b 1)
)
echo Starting SentiLens at http://localhost:8501  (close this window to stop it)
.venv\Scripts\streamlit run app.py --server.port 8501
