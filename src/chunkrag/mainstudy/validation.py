"""Repository and artifact validation entry points (Specification Sections 24--25 and E7)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .canonical import file_sha256, read_json
from .completion import completed_stages
from .constants import EXPERIMENT_ORDER, PROTOCOL_SHA256
from .experiments import condition_ids_e1, condition_ids_e2, full_plan
from .paper import validate_output_assignments
from .coverage import validate_coverage
from .protocol import load_protocol_config, verify_protocol


def validate_statistical_primitives() -> dict[str, Any]:
    from .statistics import cluster_bootstrap, cluster_sign_flip, holm_adjust, rank_biserial, tost
    values = [1.0, -1.0, 2.0, -2.0, 0.5, -0.5]
    clusters = ["a", "a", "b", "b", "c", "c"]
    first = cluster_bootstrap(values, clusters, "synthetic:self-test", draws=500)
    second = cluster_bootstrap(values, clusters, "synthetic:self-test", draws=500)
    if first != second:
        raise ValueError("Cluster bootstrap is not deterministic")
    adjusted = holm_adjust({"a": 0.01, "b": 0.04, "c": 0.20})
    if any(not 0 <= value <= 1 for value in adjusted.values()) or rank_biserial([1, -1, 1]) != 1 / 3:
        raise ValueError("Statistical effect/multiplicity self-test failed")
    if not 0 <= cluster_sign_flip(values, clusters, "synthetic:self-test", draws=99) <= 1:
        raise ValueError("Cluster sign-flip self-test failed")
    equivalence = tost([0.0, 0.1, -0.1, 0.0, 0.05, -0.05], ["a", "a", "b", "b", "c", "c"], 2.0)
    if not 0 <= equivalence["p_tost"] <= 1:
        raise ValueError("TOST self-test failed")
    return {"schema_version": "chunkrag-main-v1", "status": "valid", "bootstrap_repeated": True, "checks": ["bootstrap", "sign-flip", "holm", "rank-biserial", "tost"]}


def validate_repository(root: Path, *, check_environment: bool = False) -> dict[str, Any]:
    protocol_hash = verify_protocol(root)
    config = load_protocol_config(root / "configs" / "main_study.json")
    lock_hash = file_sha256(root / "requirements-main-study.transitive.json")
    if len(condition_ids_e1()) != 24 or len(condition_ids_e2()) != 31:
        raise ValueError("Frozen condition counts failed")
    plan = full_plan()
    if set(item.experiment for item in plan) != set(EXPERIMENT_ORDER):
        raise ValueError("Execution plan does not cover E0--E7")
    validate_output_assignments()
    coverage = validate_coverage(root)
    if check_environment:
        from .environment import environment_manifest

        environment_manifest(root / "requirements-main-study.transitive.json", check_installed=True)
    return {
        "protocol_sha256": protocol_hash, "config_sha256": config["config_sha256"],
        "environment_lock_sha256": lock_hash, "work_items": len(plan),
        "experiments": list(EXPERIMENT_ORDER), "status": "valid",
        "protocol_sections_covered": coverage["sections"],
    }


def validate_completion_manifest(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if payload.get("protocol_sha256") != PROTOCOL_SHA256:
        raise ValueError("Completion manifest protocol mismatch")
    if payload.get("completed_experiments") != list(EXPERIMENT_ORDER):
        raise ValueError("Completion manifest does not contain E0--E7 in order")
    artifact_root = path.parent.parent
    if path.resolve() != (artifact_root / "audit" / "completion.json").resolve():
        raise ValueError("Completion manifest is not at the canonical path")
    provenance = {key: payload.get(key) for key in ("git_commit", "config_sha256", "environment_hash")}
    if not all(isinstance(value, str) and value for value in provenance.values()):
        raise ValueError("Completion manifest lacks canonical provenance")
    if completed_stages(artifact_root, **provenance) != list(EXPERIMENT_ORDER):
        raise ValueError("Completion manifest does not match validated E0--E7 stages")
    for reference in [*payload.get("stage_markers", []), *payload.get("artifacts", [])]:
        target = artifact_root / reference["path"]
        if not target.is_file() or file_sha256(target) != reference["sha256"]:
            raise ValueError(f"Completion artifact mismatch: {reference['path']}")
        if "bytes" in reference and target.stat().st_size != int(reference["bytes"]):
            raise ValueError(f"Completion artifact size mismatch: {reference['path']}")
        if target.stat().st_mode & 0o222:
            raise ValueError(f"Completion artifact is writable: {reference['path']}")
    if path.stat().st_mode & 0o222:
        raise ValueError("Completion manifest is writable")
    return payload
