#!/bin/bash
# Simple script to create a clean ZIP file for sharing
# This excludes .venv, projects, and other unnecessary files

echo "Creating shareable ZIP file..."
echo

# Get the current folder name
FOLDERNAME=$(basename "$(pwd)")

# Create a temporary directory with only the files we want
TEMP_DIR=$(mktemp -d)
cp -r app.py start_app.bat start_app.command start_app.py requirements.txt readme.md GETTING_STARTED.md .gitignore src "$TEMP_DIR/"

# Create ZIP file
cd "$TEMP_DIR"
zip -r "../${FOLDERNAME}-shareable.zip" . > /dev/null
cd - > /dev/null

# Clean up
rm -rf "$TEMP_DIR"

if [ $? -eq 0 ]; then
    echo
    echo "✅ Success! Created '${FOLDERNAME}-shareable.zip'"
    echo
    echo "This ZIP file is ready to share. It excludes:"
    echo "  - .venv folder (too large, will be recreated)"
    echo "  - projects folder (personal data)"
    echo "  - .env file (your API key)"
    echo
else
    echo
    echo "❌ Error creating ZIP file."
    echo "Make sure you're running this from the project folder."
    echo
fi

