#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${OPENWEBUI_VENV_DIR:-$REPO_ROOT/.venv_openwebui}"
DATA_DIR="${OPENWEBUI_DATA_DIR:-$REPO_ROOT/.open-webui}"

/opt/homebrew/bin/python3.11 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
python -m pip install "open-webui>=0.6,<1" "openai>=1,<2"
mkdir -p "$DATA_DIR"

echo "OpenWebUI environment is ready in $VENV_DIR"
