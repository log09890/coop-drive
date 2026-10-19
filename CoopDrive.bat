@echo off
rem Coop-Drive: double-click to start. First run creates the Python 3.7 environment (needs internet once),
rem then CARLA + the dashboard start; everything else is typed in the dashboard terminal (help).
chcp 65001 >nul
cd /d "%~dp0"
if not exist venv\Scripts\python.exe (
    echo First run: creating the Python 3.7 environment...
    py -3.7 -m venv venv || (echo Python 3.7 64-bit is required: https://www.python.org/downloads/release/python-379/ & pause & exit /b 1)
    venv\Scripts\python.exe -m pip install --upgrade pip
    venv\Scripts\python.exe -m pip install -r requirements-app.txt --extra-index-url https://download.pytorch.org/whl/cu117 || (pause & exit /b 1)
)
venv\Scripts\python.exe app.py
if errorlevel 1 pause
