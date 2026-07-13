"""Hashed work/stage completion registry (Specification Sections 23, 26, 29, E7)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .canonical import atomic_write_json, canonical_json_hash, read_json
from .constants import EXPERIMENT_ORDER, PROTOCOL_ID, PROTOCOL_SHA256
from .experiments import WorkItem, plan_experiment


def marker_path(artifact_root: Path, work_id: str) -> Path:
    return artifact_root / "audit" / "work-items" / f"{canonical_json_hash(work_id)}.json"


def mark_work_complete(
    artifact_root: Path, item: WorkItem, artifact_hashes: list[str], *,
    git_commit: str, config_sha256: str, environment_hash: str,
) -> str:
    payload = {
        "schema_version": PROTOCOL_ID, "protocol_sha256": PROTOCOL_SHA256,
        "work_id": item.work_id, "experiment": item.experiment,
        "artifact_hashes": artifact_hashes, "git_commit": git_commit,
        "config_sha256": config_sha256, "environment_hash": environment_hash,
        "status": "complete",
    }
    return atomic_write_json(marker_path(artifact_root, item.work_id), payload)


def completed_work_ids(artifact_root: Path, experiment: str) -> set[str]:
    result: set[str] = set()
    for path in (artifact_root / "audit" / "work-items").glob("*.json"):
        row = read_json(path)
        if row.get("protocol_sha256") == PROTOCOL_SHA256 and row.get("experiment") == experiment and row.get("status") == "complete":
            result.add(str(row["work_id"]))
    return result


def stage_is_complete(artifact_root: Path, experiment: str) -> bool:
    expected = {item.work_id for item in plan_experiment(experiment)}
    return completed_work_ids(artifact_root, experiment) == expected


def finalize_stage(artifact_root: Path, experiment: str) -> str | None:
    if not stage_is_complete(artifact_root, experiment):
        return None
    markers = [read_json(marker_path(artifact_root, item.work_id)) for item in plan_experiment(experiment)]
    commits = {row["git_commit"] for row in markers}
    configs = {row["config_sha256"] for row in markers}
    environments = {row["environment_hash"] for row in markers}
    if len(commits) != 1 or len(configs) != 1 or len(environments) != 1:
        raise ValueError(f"Mixed provenance within completed stage {experiment}")
    payload = {
        "schema_version": PROTOCOL_ID, "protocol_sha256": PROTOCOL_SHA256,
        "experiment": experiment, "work_ids": sorted(row["work_id"] for row in markers),
        "work_marker_hashes": sorted(canonical_json_hash(row) for row in markers),
        "git_commit": next(iter(commits)), "config_sha256": next(iter(configs)),
        "environment_hash": next(iter(environments)), "status": "complete",
    }
    return atomic_write_json(artifact_root / "audit" / "stages" / f"{experiment}.json", payload)


def completed_stages(artifact_root: Path) -> list[str]:
    completed = []
    for experiment in EXPERIMENT_ORDER:
        path = artifact_root / "audit" / "stages" / f"{experiment}.json"
        if path.is_file() and read_json(path).get("status") == "complete":
            completed.append(experiment)
    return completed
