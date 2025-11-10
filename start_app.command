#!/bin/bash
# Double-click launcher for macOS. Creates a virtual environment, installs
# dependencies, and starts the Document RAG Assistant.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR=".venv"
PYTHON_BIN="$SCRIPT_DIR/$VENV_DIR/bin/python3"
PYTHONW="$PYTHON_BIN"
REQ_FILE="$SCRIPT_DIR/requirements.txt"
STAMP_FILE="$SCRIPT_DIR/$VENV_DIR/requirements.fingerprint"

function show_dialog() {
  local title="$1"
  local message="$2"
  if command -v osascript >/dev/null 2>&1; then
    osascript -e "display dialog \"$message\" buttons {\"OK\"} default button \"OK\" with title \"$title\""
  else
    echo "[$title] $message"
  fi
}

function run_python() {
  "$PYTHON_BIN" "$@"
}

function run_cmd() {
  "$@" || {
    show_dialog "Document RAG Assistant" "Command failed: $*"
    exit 1
  }
}

# Step 1: ensure Python 3 is available
if ! command -v python3 >/dev/null 2>&1; then
  show_dialog "Document RAG Assistant" "Python 3.11+ is required but was not found. Install it from https://www.python.org/downloads/mac-osx/ and try again."
  exit 1
fi

# Step 2: create virtual environment if missing
if [ ! -f "$PYTHON_BIN" ]; then
  show_dialog "Document RAG Assistant" "Creating Python virtual environment (this runs once and may take a minute)..."
  run_cmd python3 -m venv "$VENV_DIR"
fi

# Step 3: install dependencies if requirements changed or missing
need_install=true
if [ -f "$STAMP_FILE" ] && [ -f "$REQ_FILE" ]; then
  current_fingerprint="$(stat -f "%z|%m" "$REQ_FILE")"
  saved_fingerprint="$(cat "$STAMP_FILE")"
  if [ "$current_fingerprint" = "$saved_fingerprint" ]; then
    if run_python -c "import gradio, langchain, langchain_openai, langchain_chroma" >/dev/null 2>&1; then
      need_install=false
    fi
  fi
fi

if [ "$need_install" = true ]; then
  show_dialog "Document RAG Assistant" "Installing project dependencies (this may take a minute)..."
  run_python -m pip install --upgrade pip
  run_python -m pip install -r "$REQ_FILE"
  if [ -f "$REQ_FILE" ]; then
    stat -f "%z|%m" "$REQ_FILE" > "$STAMP_FILE"
  fi
fi

# Step 4: launch the app
run_python app.py &
show_dialog "Document RAG Assistant" "The app is starting in your browser. Keep this window open while you use it."

exit 0


