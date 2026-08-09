from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from chunkrag.eaai_phase2.constants import BASELINE_TREE_SHA256, PROTOCOL_COMMIT
from chunkrag.eaai_phase2.io import sha256_file


@dataclass(frozen=True, slots=True)
class BaselineVerification:
    expected_files: int
    verified_files: int
    tree_sha256: str


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def verify_baseline(
    root: str | Path | None = None,
    manifest_path: str | Path = "reports/eaai_phase2_baseline_manifest.json",
) -> BaselineVerification:
    repo = Path(root) if root is not None else repository_root()
    manifest_file = repo / manifest_path
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if manifest.get("tree_sha256") != BASELINE_TREE_SHA256:
        raise ValueError(
            "Baseline manifest tree hash differs from the frozen Phase 2 protocol: "
            f"{manifest.get('tree_sha256')} != {BASELINE_TREE_SHA256}"
        )
    failures: list[str] = []
    rows = manifest.get("files")
    if not isinstance(rows, list):
        raise TypeError("Baseline manifest files field must be a list")
    for row in rows:
        path = repo / str(row["path"])
        if not path.is_file():
            failures.append(f"missing:{row['path']}")
            continue
        actual_size = path.stat().st_size
        actual_hash = sha256_file(path)
        if actual_size != int(row["bytes"]) or actual_hash != str(row["sha256"]):
            failures.append(
                f"changed:{row['path']}:{actual_size}:{actual_hash}"
            )
    if failures:
        preview = "\n".join(failures[:20])
        raise RuntimeError(
            f"Frozen EAAI baseline verification failed for {len(failures)} files:\n{preview}"
        )
    return BaselineVerification(len(rows), len(rows), str(manifest["tree_sha256"]))


def _git_output(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def verify_protocol_commit(root: str | Path | None = None) -> str:
    repo = Path(root) if root is not None else repository_root()
    protocol = "reports/eaai_phase2_protocol.md"
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", PROTOCOL_COMMIT, "HEAD"],
        cwd=repo,
        check=False,
    )
    if ancestor.returncode != 0:
        raise RuntimeError(f"Frozen protocol commit is not an ancestor of HEAD: {PROTOCOL_COMMIT}")
    working_bytes = (repo / protocol).read_bytes()
    frozen_bytes = subprocess.run(
        ["git", "show", f"{PROTOCOL_COMMIT}:{protocol}"],
        cwd=repo,
        check=True,
        capture_output=True,
    ).stdout
    if not working_bytes.startswith(frozen_bytes):
        raise RuntimeError(
            "Protocol differs from its frozen bytes; only text appended after the "
            "original dated deviation log is permitted"
        )
    return sha256_file(repo / protocol)


def verify_clean_paths(paths: Iterable[str], root: str | Path | None = None) -> None:
    repo = Path(root) if root is not None else repository_root()
    path_list = list(paths)
    if not path_list:
        return
    status = _git_output(repo, "status", "--porcelain", "--", *path_list)
    if status:
        raise RuntimeError(
            "Phase 2 scientific implementation must be committed before full inference:\n"
            + status
        )


def require_within(path: str | Path, allowed_root: str | Path) -> Path:
    resolved_path = Path(path).resolve()
    resolved_root = Path(allowed_root).resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"Output path escapes allowed root: {resolved_path}") from exc
    return resolved_path
