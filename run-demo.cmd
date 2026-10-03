@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  where py >nul 2>nul
  if not errorlevel 1 (
    py -3 -m venv .venv
  ) else (
    python -m venv .venv
  )
  if errorlevel 1 goto :failed
)
".venv\Scripts\python.exe" -c "import streamlit, disaster_triage" >nul 2>nul
if errorlevel 1 (
  ".venv\Scripts\python.exe" -m pip install -e ".[demo]"
  if errorlevel 1 goto :failed
)
".venv\Scripts\python.exe" -m streamlit run demo.py
if errorlevel 1 goto :failed
exit /b 0
:failed
echo.
echo Setup or launch failed. Make sure Python 3.10 or newer is installed and the first setup has internet access.
pause
exit /b 1
