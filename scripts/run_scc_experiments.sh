#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${1:-$REPO_ROOT/configs/scc_rigorous_qwen.json}"
OUTPUT_DIR="${2:-$REPO_ROOT/outputs/scc_rigorous_qwen}"

source "$REPO_ROOT/scripts/setup_scc_env.sh"

python "$REPO_ROOT/scripts/run_experiments.py" \
  --config "$CONFIG_PATH" \
  --output-dir "$OUTPUT_DIR"

python "$REPO_ROOT/scripts/plot_results.py" \
  --results "$OUTPUT_DIR/aggregate_results.json" \
  --output-dir "$OUTPUT_DIR/figures"
