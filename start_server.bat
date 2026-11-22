@echo off
REM AXIPLAT Server Launcher (visible window)
setlocal

set "BAT_DIR=%~dp0"
cd /d "%BAT_DIR%"

echo ========================================
echo AXIPLAT Server Launcher
echo ========================================
echo.

if exist "venv\Scripts\python.exe" (
    echo [OK] Using project virtual environment
    "venv\Scripts\python.exe" app.py
) else (
    echo [INFO] No venv folder found - using system Python.
    echo        Run setup_autostart.bat once to create a proper venv.
    echo.
    python app.py
)

echo.
echo Server stopped.
pause
