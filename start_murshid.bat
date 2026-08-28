@echo off
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Virtual environment not found — creating one...
    python -m venv .venv
)

echo Checking dependencies...
".venv\Scripts\python.exe" -m pip install -q -r requirements.txt

echo Starting Murshid...
".venv\Scripts\python.exe" run.py

echo.
echo Murshid has stopped.
pause
