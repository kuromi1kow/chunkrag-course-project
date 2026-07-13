#!/usr/bin/env python3
"""One-command locked analysis and paper regeneration (Specification Sections 25, 29--30)."""

import argparse
import json
from pathlib import Path

from chunkrag.mainstudy.analysis import regenerate_analysis
from chunkrag.mainstudy.canonical import atomic_write_json, file_sha256, read_json
from chunkrag.mainstudy.constants import PROTOCOL_ID, PROTOCOL_SHA256
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
    output_rows = [{"path": path.name, "sha256": file_sha256(path), "bytes": path.stat().st_size} for path in outputs]
    manifest = {
        "schema_version": PROTOCOL_ID, "protocol_sha256": PROTOCOL_SHA256,
        "analysis_sha256": digest, "outputs": sorted(output_rows, key=lambda row: row["path"]),
    }
    paper_manifest_hash = atomic_write_json(args.paper_output_dir / "paper-output-hash-manifest.json", manifest)
    artifact_manifest = args.artifact_root / "audit" / "paper-output-hash-manifest.json"
    atomic_write_json(artifact_manifest, {**manifest, "manifest_sha256": paper_manifest_hash})
    for path in [*outputs, args.paper_output_dir / "paper-output-hash-manifest.json", artifact_manifest]:
        path.chmod(path.stat().st_mode & ~0o222)
    print(json.dumps({"analysis_sha256": digest, "paper_manifest_sha256": paper_manifest_hash, "paper_outputs": [str(path) for path in outputs]}, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
