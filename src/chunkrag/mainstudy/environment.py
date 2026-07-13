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
        "node": platform.node(),
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
    try:
        import torch
        payload["torch_version"] = torch.__version__
        payload["cuda_build"] = torch.version.cuda
        payload["cudnn_version"] = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
    except ImportError:
        payload.update({"torch_version": None, "cuda_build": None, "cudnn_version": None})
    return payload


def installed_packages() -> list[dict[str, str]]:
    return sorted(
        ({"name": str(distribution.metadata["Name"]), "version": distribution.version}
         for distribution in importlib.metadata.distributions()
         if distribution.metadata["Name"] and str(distribution.metadata["Name"]).lower() != "chunkrag"),
        key=lambda row: (row["name"].lower(), row["version"]),
    )


def environment_manifest(lock_path: Path, *, check_installed: bool) -> dict[str, Any]:
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    required = {"schema_version", "python", "implementation", "packages"}
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ProtocolError("Resolved environment lock is missing required fields")
    direct_path = lock_path.with_name("requirements-main-study.lock")
    if payload.get("direct_lock_sha256") != file_sha256(direct_path):
        raise ProtocolError("Resolved environment lock does not match the frozen direct lock")
    observed = installed_packages()
    if check_installed:
        verify_direct_versions(installed=True)
        expected = {(row["name"].lower(), row["version"]) for row in payload["packages"]}
        actual = {(row["name"].lower(), row["version"]) for row in observed}
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise ProtocolError(f"Transitive environment mismatch; missing={missing}, extra={extra}")
        if sys.version.split()[0] != payload["python"] or platform.python_implementation() != payload["implementation"]:
            raise ProtocolError("Python runtime does not match the resolved environment lock")
    return {
        "lock_sha256": file_sha256(lock_path),
        "direct_versions": verify_direct_versions(installed=check_installed),
        "packages": observed,
        "hardware": hardware_manifest(),
    }


def freeze_transitive_environment(path: Path, lock_path: Path) -> str:
    payload = {
        "schema_version": "chunkrag-main-environment-v1",
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "direct_lock_sha256": file_sha256(lock_path),
        "packages": installed_packages(),
    }
    return atomic_write_json(path, payload)


def require_canonical_a100(manifest: dict[str, Any]) -> None:
    hardware = manifest.get("hardware", manifest)
    gpus = hardware.get("gpus", [])
    if not gpus or any("A100" not in gpu for gpu in gpus):
        raise ProtocolError("Canonical Colab/SCC output requires NVIDIA A100 hardware")
    if not hardware.get("cuda_build"):
        raise ProtocolError("Canonical GPU output requires a recorded CUDA build")


def require_canonical_runtime(manifest: dict[str, Any], *, gpu_required: bool) -> None:
    if sys.version_info[:2] != (3, 11):
        raise ProtocolError("Canonical execution requires Python 3.11")
    if gpu_required:
        require_canonical_a100(manifest)


def write_runtime_template(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
