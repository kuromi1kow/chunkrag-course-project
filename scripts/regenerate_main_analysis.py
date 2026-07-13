#!/usr/bin/env python3
"""One-command locked analysis and paper regeneration (Specification Sections 25, 29--30)."""

import argparse
import json
from pathlib import Path

from chunkrag.mainstudy.analysis import regenerate_analysis
from chunkrag.mainstudy.canonical import read_json
from chunkrag.mainstudy.paper import regenerate_paper_artifacts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/chunkrag-main-v1"))
    parser.add_argument("--completion-manifest", type=Path, required=True)
    parser.add_argument("--analysis-output", type=Path, default=Path("artifacts/chunkrag-main-v1/analysis/confirmatory.json"))
    parser.add_argument("--paper-output-dir", type=Path, default=Path("reports/main-study-generated"))
    args = parser.parse_args()
    digest = regenerate_analysis(args.artifact_root, args.completion_manifest, args.analysis_output)
    outputs = regenerate_paper_artifacts(read_json(args.analysis_output), args.paper_output_dir)
    print(json.dumps({"analysis_sha256": digest, "paper_outputs": [str(path) for path in outputs]}, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
