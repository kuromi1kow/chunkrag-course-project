#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${DEMO_VENV_DIR:-$REPO_ROOT/.venv_demo}"

/opt/homebrew/bin/python3.11 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
python -m pip install -e "${REPO_ROOT}[demo]"

echo "Demo dashboard environment is ready in $VENV_DIR"
