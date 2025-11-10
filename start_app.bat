@echo off
setlocal
cd /d "%~dp0"

REM Prefer the project virtual environment if present
if exist .venv\Scripts\pythonw.exe (
    set "PYTHONW=.venv\Scripts\pythonw.exe"
) else (
    for %%I in (pythonw.exe) do set "PYTHONW=%%~$PATH:I"
)

if not defined PYTHONW (
    echo Could not find pythonw.exe. Please ensure Python is installed and added to PATH.
    pause
    exit /b 1
)

"%PYTHONW%" start_app.pyw

if errorlevel 1 (
    echo The application reported an error. Check last_error.log for details.
    pause
)

endlocal

