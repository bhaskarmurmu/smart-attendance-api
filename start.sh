#!/bin/sh
# start.sh - safe startup wrapper to ensure $PORT is expanded by a shell
# Uses uvicorn if available; falls back to python run.py which already
# handles PORT parsing in Python.

set -e

# Prefer python entrypoint which parses PORT as int and provides a fallback
if command -v python >/dev/null 2>&1 && [ -f "./run.py" ]; then
  exec python run.py
fi

# If run.py is not present, fall back to running uvicorn directly with shell
# expansion so $PORT isn't passed as a literal string.
PORT_ARG=${PORT:-8000}
exec uvicorn main:app --host 0.0.0.0 --port "$PORT_ARG"
