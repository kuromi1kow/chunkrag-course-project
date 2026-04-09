#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${DEMO_VENV_DIR:-$REPO_ROOT/.venv_demo}"
PORT="${DEMO_DASHBOARD_PORT:-8501}"
RESULTS_PATH="${RESULTS_PATH:-$REPO_ROOT/outputs/rigorous_smoke/aggregate_results.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/outputs/rigorous_smoke}"

source "$VENV_DIR/bin/activate"
streamlit run "$REPO_ROOT/apps/rag_demo_dashboard.py" --server.port "$PORT" -- \
  --results-path "$RESULTS_PATH" \
  --output-dir "$OUTPUT_DIR"
