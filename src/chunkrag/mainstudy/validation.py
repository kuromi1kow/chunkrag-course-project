"""Repository and artifact validation entry points (Specification Sections 24--25 and E7)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .canonical import file_sha256, read_json
from .constants import EXPERIMENT_ORDER, PROTOCOL_SHA256
from .experiments import condition_ids_e1, condition_ids_e2, full_plan
from .paper import validate_output_assignments
from .coverage import validate_coverage
from .protocol import load_protocol_config, verify_protocol


def validate_repository(root: Path, *, check_environment: bool = False) -> dict[str, Any]:
    protocol_hash = verify_protocol(root)
    config = load_protocol_config(root / "configs" / "main_study.json")
    lock_hash = file_sha256(root / "requirements-main-study.lock")
    if len(condition_ids_e1()) != 24 or len(condition_ids_e2()) != 31:
        raise ValueError("Frozen condition counts failed")
    plan = full_plan()
    if set(item.experiment for item in plan) != set(EXPERIMENT_ORDER):
        raise ValueError("Execution plan does not cover E0--E7")
    validate_output_assignments()
    coverage = validate_coverage(root)
    if check_environment:
        from .environment import verify_direct_versions

        verify_direct_versions(installed=True)
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
    return payload
