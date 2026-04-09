#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${OPENWEBUI_VENV_DIR:-$REPO_ROOT/.venv_openwebui}"
DATA_DIR="${OPENWEBUI_DATA_DIR:-$REPO_ROOT/.open-webui}"
HOST="${OPENWEBUI_HOST:-127.0.0.1}"
PORT="${OPENWEBUI_PORT:-8080}"

source "$VENV_DIR/bin/activate"
mkdir -p "$DATA_DIR"

DATA_DIR="$DATA_DIR" open-webui serve --host "$HOST" --port "$PORT"
