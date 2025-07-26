@echo off
REM Virtual Environment Setup Script for finance_dashboard (Windows)

echo Setting up Python virtual environment...

REM Create virtual environment
python -m venv finance_dashboard_env

REM Activate virtual environment
call finance_dashboard_env\Scripts\activate.bat

REM Upgrade pip
pip install --upgrade pip

REM Install requirements
pip install -r requirements.txt

echo Virtual environment setup complete!
echo To activate: finance_dashboard_env\Scripts\activate.bat
echo To run dashboard: streamlit run run.py
pause
