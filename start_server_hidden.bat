@echo off
REM AXIPLAT Server Launcher (hidden background start)
setlocal

set "BAT_DIR=%~dp0"
cd /d "%BAT_DIR%"

if exist "%BAT_DIR%venv\Scripts\python.exe" (
    set "PYTHON=%BAT_DIR%venv\Scripts\python.exe"
) else (
    set "PYTHON=python"
)

set "VBS_SCRIPT=%TEMP%\axiplat_launcher_%RANDOM%.vbs"

(
echo Set WshShell = CreateObject^("WScript.Shell"^)
echo WshShell.CurrentDirectory = "%BAT_DIR%"
echo WshShell.Run """%PYTHON%"" app.py", 0, False
) > "%VBS_SCRIPT%"

cscript //nologo "%VBS_SCRIPT%"
del "%VBS_SCRIPT%" >nul 2>&1
exit /b 0
