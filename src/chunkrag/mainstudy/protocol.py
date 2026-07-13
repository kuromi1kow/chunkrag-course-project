"""Protocol authority and config validation (Specification Sections 1, 7, 23, 33)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .canonical import canonical_json_hash, file_sha256
from .constants import (
    DATASET_ORDER,
    EXPERIMENT_ORDER,
    JITTER_SEEDS,
    POLICY_ORDER,
    PROTOCOL_ID,
    PROTOCOL_SHA256,
)


class ProtocolError(RuntimeError):
    """Raised when implementation inputs disagree with the immutable specification."""


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def protocol_path(root: Path | None = None) -> Path:
    return (root or repo_root()) / "reports" / "phase2_immutable_specification.md"


def verify_protocol(root: Path | None = None) -> str:
    actual = file_sha256(protocol_path(root))
    if actual != PROTOCOL_SHA256:
        raise ProtocolError(f"Protocol checksum mismatch: expected {PROTOCOL_SHA256}, got {actual}")
    checksum_path = (root or repo_root()) / "reports" / "phase2_immutable_specification.sha256"
    declared = checksum_path.read_text(encoding="utf-8").split()[0]
    if declared != PROTOCOL_SHA256:
        raise ProtocolError(f"Companion checksum mismatch: {declared}")
    return actual


def _require_equal(config: dict[str, Any], path: tuple[str, ...], expected: Any) -> None:
    value: Any = config
    for key in path:
        if not isinstance(value, dict) or key not in value:
            raise ProtocolError(f"Missing config field: {'.'.join(path)}")
        value = value[key]
    if value != expected:
        raise ProtocolError(f"Frozen field {'.'.join(path)} expected {expected!r}, got {value!r}")


def validate_protocol_config(config: dict[str, Any]) -> None:
    _require_equal(config, ("protocol_id",), PROTOCOL_ID)
    _require_equal(config, ("protocol_sha256",), PROTOCOL_SHA256)
    _require_equal(config, ("dataset_order",), list(DATASET_ORDER))
    _require_equal(config, ("policy_order",), list(POLICY_ORDER))
    _require_equal(config, ("jitter_seeds",), list(JITTER_SEEDS))
    _require_equal(config, ("experiment_order",), list(EXPERIMENT_ORDER))
    _require_equal(config, ("chunking", "target_tokens"), 192)
    _require_equal(config, ("chunking", "minimum_tokens"), 64)
    _require_equal(config, ("chunking", "maximum_tokens"), 254)
    _require_equal(config, ("retrieval", "dense_top_k"), 50)
    _require_equal(config, ("retrieval", "sparse_top_k"), 50)
    _require_equal(config, ("retrieval", "rerank_top_k"), 50)
    _require_equal(config, ("retrieval", "frozen_top_k"), 16)
    _require_equal(config, ("retrieval", "operational_top_k"), 4)
    _require_equal(config, ("statistics", "alpha"), 0.05)
    _require_equal(config, ("statistics", "bootstrap_draws"), 20000)
    _require_equal(config, ("statistics", "sign_flip_draws"), 99999)
    _require_equal(config, ("sharding", "questions_per_shard"), 50)
    if len(config.get("datasets", {})) != 3 or len(config.get("models", {})) != 5:
        raise ProtocolError("Frozen config must contain exactly three datasets and five model roles")


def load_protocol_config(path: Path | None = None, *, verify: bool = True) -> dict[str, Any]:
    root = repo_root()
    if verify:
        verify_protocol(root)
    config_path = path or root / "configs" / "main_study.json"
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ProtocolError("Main-study config must be a JSON object")
    validate_protocol_config(config)
    config["config_sha256"] = canonical_json_hash(config)
    return config


__all__ = [
    "PROTOCOL_ID",
    "PROTOCOL_SHA256",
    "ProtocolError",
    "load_protocol_config",
    "protocol_path",
    "repo_root",
    "validate_protocol_config",
    "verify_protocol",
]
