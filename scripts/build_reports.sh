#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPORTS_DIR="$ROOT_DIR/reports"
MODE="${1:-all}"

build_one() {
  local stem="$1"
  local tex_file="${stem}_acl.tex"
  local final_pdf="${stem}.pdf"

  (cd "$REPORTS_DIR" && tectonic "$tex_file")
  cp "$REPORTS_DIR/${stem}_acl.pdf" "$REPORTS_DIR/$final_pdf"
}

python3 "$ROOT_DIR/scripts/export_report_tables.py"

# Regenerate paper figures (used by reports/final_report_acl.tex).
# Prefer the project venv if available, otherwise fall back to system python.
if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PY_FIGURES="$ROOT_DIR/.venv/bin/python"
elif [[ -x "$ROOT_DIR/.venv_figs/bin/python" ]]; then
  PY_FIGURES="$ROOT_DIR/.venv_figs/bin/python"
else
  PY_FIGURES="python3"
fi
"$PY_FIGURES" "$ROOT_DIR/scripts/build_paper_figures.py"

case "$MODE" in
  all)
    build_one "midway_report"
    build_one "final_report"
    ;;
  midway)
    build_one "midway_report"
    ;;
  final)
    build_one "final_report"
    ;;
  *)
    echo "Usage: $0 [all|midway|final]" >&2
    exit 1
    ;;
esac
