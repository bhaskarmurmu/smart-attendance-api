"""Entrypoint that reads PORT from the environment and runs uvicorn safely.

Using a Python entrypoint avoids issues where process managers pass the literal
string "$PORT" to uvicorn (no shell expansion). We parse PORT as an int and
fall back to 8000.
"""
import os
import sys
import uvicorn

try:
    # Import the FastAPI app from main
    from main import app
except Exception as e:
    print(f"Failed to import app from main: {e}")
    sys.exit(1)


def get_port() -> int:
    port = os.getenv("PORT", "8000")
    try:
        return int(port)
    except (TypeError, ValueError):
        print(f"Invalid PORT '{port}', falling back to 8000")
        return 8000


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=get_port())
