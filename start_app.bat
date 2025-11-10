@echo off
setlocal

cd /d "%~dp0"

REM Prefer project virtual environment python if it already exists
if exist .venv\Scripts\python.exe (
    set "PYTHON=.venv\Scripts\python.exe"
) else (
    for %%I in (python.exe) do if not defined PYTHON set "PYTHON=%%~$PATH:I"
)

if not defined PYTHON (
    echo Python 3.11+ is required but was not found on PATH.
    echo Install it from https://www.python.org/downloads/ and try again.
    pause
    exit /b 1
)

"%PYTHON%" "%~dp0start_app.py"
if errorlevel 1 pause

endlocal

