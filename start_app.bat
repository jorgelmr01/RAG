@echo off
setlocal EnableDelayedExpansion

REM ---------------------------------------------------------------------------
REM  Navigate to the folder that contains this script
REM ---------------------------------------------------------------------------
cd /d "%~dp0"

REM ---------------------------------------------------------------------------
REM  Locate a Python interpreter (prefer the launcher if available)
REM ---------------------------------------------------------------------------
set "PY_EXEC="
set "PY_ARGS="
for %%P in (py.exe) do if not defined PY_EXEC set "PY_EXEC=%%~$PATH:P"
if defined PY_EXEC (
    "%PY_EXEC%" -3 --version >nul 2>&1
    if errorlevel 1 (
        set "PY_ARGS="
    ) else (
        set "PY_ARGS=-3"
    )
)
if not defined PY_EXEC (
    for %%P in (python.exe) do if not defined PY_EXEC set "PY_EXEC=%%~$PATH:P"
)
if not defined PY_EXEC (
    echo Python 3.11+ is required but was not found on PATH.
    echo Install it from https://www.python.org/downloads/ and try again.
    pause
    exit /b 1
)

REM ---------------------------------------------------------------------------
REM  Create the virtual environment if it does not exist
REM ---------------------------------------------------------------------------
set "VENV_DIR=.venv"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"
if not exist "%VENV_PY%" (
    echo Creating virtual environment...
    "%PY_EXEC%" %PY_ARGS% -m venv "%VENV_DIR%"
    if errorlevel 1 goto :error
)

REM ---------------------------------------------------------------------------
REM  Install or update dependencies only when requirements.txt changes
REM ---------------------------------------------------------------------------
for %%I in (requirements.txt) do set "REQ_FINGERPRINT=%%~zI|%%~tI"
set "STAMP=%VENV_DIR%\requirements.fingerprint"
set "NEED_INSTALL=1"
if exist "%STAMP%" (
    set /p SAVED_FINGERPRINT=<"%STAMP%"
    if /I "!SAVED_FINGERPRINT!"=="!REQ_FINGERPRINT!" set "NEED_INSTALL=0"
)
if "!NEED_INSTALL!"=="1" (
    echo Installing dependencies (this runs only when requirements.txt changes)...
    "%VENV_PY%" -m pip install --upgrade pip
    if errorlevel 1 goto :error
    "%VENV_PY%" -m pip install -r requirements.txt
    if errorlevel 1 goto :error
    >"%STAMP%" echo !REQ_FINGERPRINT!
)

REM ---------------------------------------------------------------------------
REM  Run the app using pythonw when available to avoid an extra console
REM ---------------------------------------------------------------------------
set "PYTHONW=%VENV_DIR%\Scripts\pythonw.exe"
if not exist "%PYTHONW%" set "PYTHONW=%VENV_PY%"

"%PYTHONW%" start_app.pyw
if errorlevel 1 (
    echo.
    echo The application reported an error. Check last_error.log for details.
    pause
    goto :eof
)

endlocal
exit /b 0

:error
echo.
echo Failed to prepare the environment. Please confirm Python is installed and retry.
pause
endlocal
exit /b 1