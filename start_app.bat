@echo off
setlocal enabledelayedexpansion

REM Change to the script directory
cd /d "%~dp0"

REM Set console title
title Document RAG Assistant - Setup

REM Prefer project virtual environment python if it already exists
if exist .venv\Scripts\python.exe (
    set "PYTHON=.venv\Scripts\python.exe"
    goto :run
)

REM Try multiple methods to find Python
REM Method 1: Check common Python installation locations
set "PYTHON_PATHS=python.exe python3.exe py.exe"
for %%P in (%PYTHON_PATHS%) do (
    where %%P >nul 2>&1
    if !errorlevel! equ 0 (
        set "PYTHON=%%P"
        goto :found
    )
)

REM Method 2: Check Python Launcher (py.exe) with version
py -3.11 --version >nul 2>&1
if !errorlevel! equ 0 (
    set "PYTHON=py -3.11"
    goto :found
)

py -3.12 --version >nul 2>&1
if !errorlevel! equ 0 (
    set "PYTHON=py -3.12"
    goto :found
)

REM Method 3: Check registry for Python installations (Windows)
for /f "tokens=*" %%I in ('reg query "HKLM\SOFTWARE\Python\PythonCore" /s /k 2^>nul ^| findstr /i "InstallPath"') do (
    for /f "tokens=2*" %%J in ('reg query "%%I" /ve 2^>nul') do (
        if exist "%%K\python.exe" (
            set "PYTHON=%%K\python.exe"
            goto :found
        )
    )
)

REM If we get here, Python was not found
:notfound
echo.
echo ========================================================================
echo   ERROR: Python 3.11+ is required but was not found
echo ========================================================================
echo.
echo   Python was not found on your system. Please install it:
echo.
echo   1. Go to: https://www.python.org/downloads/
echo   2. Download Python 3.13.3, 3.12, or 3.14
echo   3. During installation, CHECK THIS BOX:
echo      [X] Add Python to PATH
echo   4. After installing, close this window and try again
echo.
echo   For detailed instructions, see GETTING_STARTED.md
echo.
pause
exit /b 1

:found
REM Verify Python version
"%PYTHON%" --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo ERROR: Found Python but could not run it.
    echo Please reinstall Python from https://www.python.org/downloads/
    pause
    exit /b 1
)

:run
REM Run the Python launcher script
echo.
echo Starting Document RAG Assistant setup...
echo.
"%PYTHON%" "%~dp0start_app.py"

REM Check exit code
if errorlevel 1 (
    echo.
    echo ========================================================================
    echo   Setup failed. See error messages above for details.
    echo ========================================================================
    echo.
    echo   Common solutions:
    echo   - Make sure Python 3.11, 3.12, 3.13, or 3.14 is installed
    echo   - Check that you have internet connection
    echo   - Make sure you have write permissions in this folder
    echo   - Try deleting the .venv folder and running again
    echo.
    pause
    exit /b 1
)

endlocal

