#!/bin/bash
# Double-click launcher for macOS. Delegates all setup logic to start_app.py.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$SCRIPT_DIR"

if ! command -v python3 >/dev/null 2>&1; then
  osascript -e 'display dialog "Python 3.11+ is required but was not found. Install it from https://www.python.org/downloads/mac-osx/ and try again." buttons {"OK"} default button "OK" with title "Document RAG Assistant"' >/dev/null 2>&1 || {
    echo "Python 3.11+ is required but was not found."
  }
  exit 1
fi

python3 start_app.py

status=$?
if [ $status -ne 0 ]; then
  echo
  read -p "Press Enter to close this window..." _
fi

exit $status

