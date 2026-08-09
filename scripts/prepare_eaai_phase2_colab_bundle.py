#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import tarfile
import tempfile
from pathlib import Path

from chunkrag.eaai_phase2.integrity import repository_root, verify_baseline
from chunkrag.eaai_phase2.io import sha256_file


DEFAULT_OUTPUT = "artifacts/eaai_phase2/eaai_phase2_baseline_private.tar.gz"


def build_bundle(output: Path) -> dict[str, object]:
    root = repository_root()
    verification = verify_baseline(root)
    manifest_path = root / "reports" / "eaai_phase2_baseline_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.",
        dir=output.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
                with tarfile.open(fileobj=compressed, mode="w") as archive:
                    for row in manifest["files"]:
                        source = root / row["path"]
                        payload = source.read_bytes()
                        if hashlib.sha256(payload).hexdigest() != row["sha256"]:
                            raise RuntimeError(f"Baseline changed while bundling: {row['path']}")
                        info = tarfile.TarInfo(name=str(row["path"]))
                        info.size = len(payload)
                        info.mode = 0o644
                        info.mtime = 0
                        info.uid = 0
                        info.gid = 0
                        info.uname = ""
                        info.gname = ""
                        archive.addfile(info, io.BytesIO(payload))

        bundle_hash = sha256_file(temporary)
        if output.exists():
            if sha256_file(output) != bundle_hash:
                raise FileExistsError(f"Refusing to replace a different private bundle: {output}")
            temporary.unlink()
        else:
            temporary.replace(output)
        sidecar = output.with_suffix(output.suffix + ".sha256")
        sidecar_payload = f"{bundle_hash}  {output.name}\n"
        if sidecar.exists() and sidecar.read_text(encoding="utf-8") != sidecar_payload:
            raise FileExistsError(f"SHA-256 sidecar conflict: {sidecar}")
        sidecar.write_text(sidecar_payload, encoding="utf-8")
        return {
            "output": str(output),
            "sha256": bundle_hash,
            "bytes": output.stat().st_size,
            "files": verification.verified_files,
            "baseline_tree_sha256": verification.tree_sha256,
            "private_benchmark_material": True,
        }
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a private deterministic bundle of frozen EAAI baseline files "
            "required by the Colab preflight. Do not publish this archive."
        )
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = repository_root()
    verification = verify_baseline(root)
    if args.dry_run:
        result = {
            "status": "validated_without_writing",
            "files": verification.verified_files,
            "baseline_tree_sha256": verification.tree_sha256,
            "planned_output": str((root / args.output).resolve()),
        }
    else:
        output = Path(args.output)
        if not output.is_absolute():
            output = root / output
        result = build_bundle(output.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
