@echo off
REM Simple script to create a clean ZIP file for sharing
REM This excludes .venv, projects, and other unnecessary files

echo Creating shareable ZIP file...
echo.

REM Get the current folder name
for %%I in ("%~dp0.") do set "FOLDERNAME=%%~nxI"

REM Create ZIP excluding unnecessary folders
powershell -Command "Compress-Archive -Path 'app.py','start_app.bat','start_app.command','start_app.py','requirements.txt','readme.md','GETTING_STARTED.md','.gitignore','src' -DestinationPath '%FOLDERNAME%-shareable.zip' -Force"

if %errorlevel% equ 0 (
    echo.
    echo ✅ Success! Created '%FOLDERNAME%-shareable.zip'
    echo.
    echo This ZIP file is ready to share. It excludes:
    echo   - .venv folder (too large, will be recreated)
    echo   - projects folder (personal data)
    echo   - .env file (your API key)
    echo.
) else (
    echo.
    echo ❌ Error creating ZIP file.
    echo Make sure you're running this from the project folder.
    echo.
)

pause

