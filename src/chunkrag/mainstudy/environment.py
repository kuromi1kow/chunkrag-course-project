"""Runtime provenance and clean-tree gates (Specification Sections 24, 25, 27, 28)."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from .canonical import file_sha256
from .canonical import atomic_write_json, canonical_json_hash
from .protocol import ProtocolError


DIRECT_VERSIONS = {
    "accelerate": "0.34.2", "datasets": "2.21.0", "faiss-cpu": "1.14.3",
    "huggingface-hub": "0.36.2", "langchain-text-splitters": "0.3.11",
    "matplotlib": "3.11.0", "numpy": "1.26.4", "pandas": "2.3.3",
    "rank-bm25": "0.2.2", "sentence-transformers": "3.4.1",
    "sentencepiece": "0.2.1", "spacy": "3.8.14", "torch": "2.13.0",
    "transformers": "4.57.6", "tqdm": "4.68.4",
}


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def git_state(root: Path) -> dict[str, Any]:
    return {
        "commit": _git(root, "rev-parse", "HEAD"),
        "dirty": bool(_git(root, "status", "--porcelain")),
        "branch": _git(root, "branch", "--show-current"),
    }


def require_clean_git(root: Path) -> dict[str, Any]:
    state = git_state(root)
    if state["dirty"]:
        raise ProtocolError("Canonical execution requires a clean Git worktree")
    return state


def verify_direct_versions(*, installed: bool = True) -> dict[str, str | None]:
    observed: dict[str, str | None] = {}
    for package, expected in DIRECT_VERSIONS.items():
        try:
            value = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            value = None
        observed[package] = value
        if installed and value != expected:
            raise ProtocolError(f"Dependency {package} expected {expected}, found {value}")
    return observed


def hardware_manifest() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total", "--format=csv,noheader"],
            check=True, capture_output=True, text=True,
        )
        payload["gpus"] = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    except (OSError, subprocess.CalledProcessError):
        payload["gpus"] = []
    return payload


def environment_manifest(lock_path: Path, *, check_installed: bool) -> dict[str, Any]:
    return {
        "lock_sha256": file_sha256(lock_path),
        "direct_versions": verify_direct_versions(installed=check_installed),
        "hardware": hardware_manifest(),
    }


def freeze_transitive_environment(path: Path, lock_path: Path) -> str:
    packages = sorted(
        ({"name": distribution.metadata["Name"], "version": distribution.version}
         for distribution in importlib.metadata.distributions()),
        key=lambda row: (str(row["name"]).lower(), str(row["version"])),
    )
    payload = {
        "python": sys.version,
        "implementation": platform.python_implementation(),
        "direct_lock_sha256": file_sha256(lock_path),
        "packages": packages,
        "hardware": hardware_manifest(),
    }
    payload["environment_hash"] = canonical_json_hash(payload)
    return atomic_write_json(path, payload)


def require_canonical_a100(manifest: dict[str, Any]) -> None:
    gpus = manifest.get("hardware", {}).get("gpus", [])
    if not gpus or any("A100" not in gpu for gpu in gpus):
        raise ProtocolError("Canonical Colab/SCC output requires NVIDIA A100 hardware")


def write_runtime_template(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
