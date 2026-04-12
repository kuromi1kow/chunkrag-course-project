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
