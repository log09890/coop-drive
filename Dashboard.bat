@echo off
rem Double-click: starts CARLA if needed, waits for it, opens the Coop-Drive dashboard. Extra args go to dashboard.py (e.g. --town Town02 --vn).
rem If the server dies, it is restarted and the dashboard retried, 3 times.
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
set TRIES=0

:start
tasklist /FI "IMAGENAME eq CarlaUE4-Win64-Shipping.exe" | "%SystemRoot%\System32\find.exe" /I "CarlaUE4" >nul
if errorlevel 1 (
    echo Dang bat CARLA server...
    start "" "E:\CARLA_0.9.10.1\WindowsNoEditor\CarlaUE4.exe" -quality-level=Low -windowed -ResX=640 -ResY=480
)

echo Cho CARLA san sang...
:wait
venv\Scripts\python.exe -c "import sys; sys.path.insert(0, '.'); from simulation.connection import carla; c = carla.Client('localhost', 2000); c.set_timeout(2); c.get_server_version()" >nul 2>&1
if errorlevel 1 (
    timeout /t 2 /nobreak >nul
    goto wait
)

venv\Scripts\python.exe dashboard.py %*
if not errorlevel 1 goto :eof

set /a TRIES+=1
tasklist /FI "IMAGENAME eq CarlaUE4-Win64-Shipping.exe" | "%SystemRoot%\System32\find.exe" /I "CarlaUE4" >nul
if errorlevel 1 if %TRIES% LSS 3 (
    echo CARLA server da tat, khoi dong lai ^(lan %TRIES%/3^)...
    timeout /t 5 /nobreak >nul
    goto start
)
pause
