#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate venv — update path to match your server setup before deploying
# source ~/venvs/mindmate/bin/activate

uvicorn server:app --host 0.0.0.0 --port 8001 --workers 1
