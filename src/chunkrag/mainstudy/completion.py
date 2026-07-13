"""Hashed work/stage completion registry (Specification Sections 23, 26, 29, E7)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .canonical import atomic_write_json, canonical_json_hash, file_sha256, read_json
from .constants import EXPERIMENT_ORDER, PROTOCOL_ID, PROTOCOL_SHA256
from .experiments import WorkItem, plan_experiment


def _merge_stage_generation(
    artifact_root: Path, experiment: str, *, config_sha256: str, environment_hash: str,
) -> list[dict[str, Any]]:
    if experiment not in {"E2", "E3"}:
        return []
    from .checkpoint import merge_shards
    from .canonical import read_jsonl
    groups: dict[tuple[str, str], list[WorkItem]] = {}
    for item in plan_experiment(experiment):
        if item.dataset is not None:
            groups.setdefault((item.dataset, item.condition_id), []).append(item)
    merged = []
    for (dataset, condition), items in sorted(groups.items()):
        shard_root = artifact_root / "generation" / "mistral" / dataset / condition
        paths = [shard_root / f"part-{item.shard_index:03d}.jsonl" for item in sorted(items, key=lambda row: int(row.shard_index or 0))]
        expected_ids = [row["question_id"] for row in read_jsonl(artifact_root / "manifests" / "questions" / f"{dataset}.jsonl")]
        output = artifact_root / "generation" / "mistral" / dataset / f"{condition}.jsonl"
        digest = merge_shards(
            paths, expected_ids, "question_id", output, schema="generation", require_state=True,
            expected_config_sha256=config_sha256, expected_environment_hash=environment_hash,
        )
        merged.append({"path": output.relative_to(artifact_root).as_posix(), "sha256": digest, "records": len(expected_ids)})
    return merged


def marker_path(artifact_root: Path, work_id: str) -> Path:
    return artifact_root / "audit" / "work-items" / f"{canonical_json_hash(work_id)}.json"


def mark_work_complete(
    artifact_root: Path, item: WorkItem, artifact_hashes: list[str], *,
    git_commit: str, config_sha256: str, environment_hash: str,
) -> str:
    if not artifact_hashes:
        raise ValueError("A completed work item must return at least one immutable artifact")
    artifact_entries: list[dict[str, Any]] = []
    files = sorted(
        (path for path in artifact_root.rglob("*") if path.is_file() and "/audit/work-items/" not in path.as_posix()),
        key=lambda path: (path.stat().st_mtime_ns, path.as_posix()), reverse=True,
    )
    hash_cache: dict[Path, str] = {}
    def digest_for(path: Path) -> str:
        if path not in hash_cache:
            hash_cache[path] = file_sha256(path)
        return hash_cache[path]
    for digest in artifact_hashes:
        match = next((path for path in files if digest_for(path) == digest), None)
        if match is None:
            raise ValueError(f"Work item returned an artifact hash with no file: {digest}")
        artifact_entries.append({
            "path": match.relative_to(artifact_root).as_posix(), "sha256": digest,
            "bytes": match.stat().st_size,
        })
    payload = {
        "schema_version": PROTOCOL_ID, "protocol_sha256": PROTOCOL_SHA256,
        "work_id": item.work_id, "experiment": item.experiment,
        "artifact_hashes": sorted(set(artifact_hashes)), "artifacts": sorted(artifact_entries, key=lambda row: row["path"]),
        "git_commit": git_commit,
        "config_sha256": config_sha256, "environment_hash": environment_hash,
        "status": "complete",
    }
    return atomic_write_json(marker_path(artifact_root, item.work_id), payload)


def _validate_marker(
    artifact_root: Path, row: Mapping[str, Any], *, git_commit: str | None,
    config_sha256: str | None, environment_hash: str | None,
) -> None:
    expected = {"git_commit": git_commit, "config_sha256": config_sha256, "environment_hash": environment_hash}
    if row.get("protocol_sha256") != PROTOCOL_SHA256 or row.get("status") != "complete":
        raise ValueError("Invalid work completion marker")
    for key, value in expected.items():
        if value is not None and row.get(key) != value:
            raise ValueError(f"Completion marker {key} mismatch")
    artifacts = row.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("Completion marker has no verifiable artifact references")
    observed_hashes: set[str] = set()
    for reference in artifacts:
        path = artifact_root / str(reference["path"])
        if not path.is_file() or path.stat().st_size != int(reference["bytes"]):
            raise ValueError(f"Missing or resized completion artifact: {path}")
        digest = file_sha256(path)
        if digest != reference["sha256"]:
            raise ValueError(f"Completion artifact hash mismatch: {path}")
        observed_hashes.add(digest)
    if observed_hashes != set(row.get("artifact_hashes", [])):
        raise ValueError("Completion marker artifact hash set is inconsistent")


def completed_work_ids(
    artifact_root: Path, experiment: str, *, git_commit: str | None = None,
    config_sha256: str | None = None, environment_hash: str | None = None,
) -> set[str]:
    result: set[str] = set()
    for path in (artifact_root / "audit" / "work-items").glob("*.json"):
        row = read_json(path)
        if row.get("experiment") == experiment:
            _validate_marker(
                artifact_root, row, git_commit=git_commit, config_sha256=config_sha256,
                environment_hash=environment_hash,
            )
            result.add(str(row["work_id"]))
    return result


def stage_is_complete(artifact_root: Path, experiment: str, **provenance: str) -> bool:
    expected = {item.work_id for item in plan_experiment(experiment)}
    return completed_work_ids(artifact_root, experiment, **provenance) == expected


def finalize_stage(
    artifact_root: Path, experiment: str, *, git_commit: str, config_sha256: str,
    environment_hash: str,
) -> str | None:
    provenance = {"git_commit": git_commit, "config_sha256": config_sha256, "environment_hash": environment_hash}
    if not stage_is_complete(artifact_root, experiment, **provenance):
        return None
    markers = [read_json(marker_path(artifact_root, item.work_id)) for item in plan_experiment(experiment)]
    commits = {row["git_commit"] for row in markers}
    configs = {row["config_sha256"] for row in markers}
    environments = {row["environment_hash"] for row in markers}
    if len(commits) != 1 or len(configs) != 1 or len(environments) != 1:
        raise ValueError(f"Mixed provenance within completed stage {experiment}")
    merged_artifacts = _merge_stage_generation(
        artifact_root, experiment, config_sha256=config_sha256, environment_hash=environment_hash,
    )
    payload = {
        "schema_version": PROTOCOL_ID, "protocol_sha256": PROTOCOL_SHA256,
        "experiment": experiment, "work_ids": sorted(row["work_id"] for row in markers),
        "work_marker_hashes": sorted(canonical_json_hash(row) for row in markers),
        "git_commit": next(iter(commits)), "config_sha256": next(iter(configs)),
        "environment_hash": next(iter(environments)), "status": "complete",
        "merged_artifacts": merged_artifacts,
    }
    return atomic_write_json(artifact_root / "audit" / "stages" / f"{experiment}.json", payload)


def completed_stages(
    artifact_root: Path, *, git_commit: str | None = None, config_sha256: str | None = None,
    environment_hash: str | None = None,
) -> list[str]:
    completed = []
    for experiment in EXPERIMENT_ORDER:
        path = artifact_root / "audit" / "stages" / f"{experiment}.json"
        if path.is_file():
            row = read_json(path)
            if row.get("status") != "complete" or row.get("protocol_sha256") != PROTOCOL_SHA256 or row.get("experiment") != experiment:
                continue
            if any(value is not None and row.get(key) != value for key, value in (
                ("git_commit", git_commit), ("config_sha256", config_sha256), ("environment_hash", environment_hash),
            )):
                continue
            expected_work = {item.work_id for item in plan_experiment(experiment)}
            if completed_work_ids(
                artifact_root, experiment, git_commit=git_commit,
                config_sha256=config_sha256, environment_hash=environment_hash,
            ) != expected_work:
                continue
            marker_rows = [read_json(marker_path(artifact_root, item.work_id)) for item in plan_experiment(experiment)]
            if set(row.get("work_ids", [])) != expected_work or set(row.get("work_marker_hashes", [])) != {canonical_json_hash(item) for item in marker_rows}:
                continue
            if any(not (artifact_root / reference["path"]).is_file() or file_sha256(artifact_root / reference["path"]) != reference["sha256"] for reference in row.get("merged_artifacts", [])):
                continue
            completed.append(experiment)
    return completed


def build_completion_manifest(
    artifact_root: Path, *, git_commit: str, config_sha256: str, environment_hash: str,
    source_hash: str,
) -> str:
    completed = completed_stages(
        artifact_root, git_commit=git_commit, config_sha256=config_sha256,
        environment_hash=environment_hash,
    )
    if completed != list(EXPERIMENT_ORDER):
        raise ValueError("Cannot create completion manifest before validated E0--E7 completion")
    stage_refs = []
    for experiment in EXPERIMENT_ORDER:
        path = artifact_root / "audit" / "stages" / f"{experiment}.json"
        stage_refs.append({"experiment": experiment, "path": path.relative_to(artifact_root).as_posix(), "sha256": file_sha256(path)})
    excluded = {"audit/completion.json", "audit/analysis.lock.json", "analysis/confirmatory.json"}
    artifacts = []
    for path in sorted(item for item in artifact_root.rglob("*") if item.is_file()):
        relative = path.relative_to(artifact_root).as_posix()
        if relative in excluded:
            continue
        artifacts.append({"path": relative, "sha256": file_sha256(path), "bytes": path.stat().st_size})
    payload = {
        "schema_version": PROTOCOL_ID, "protocol_sha256": PROTOCOL_SHA256,
        "git_commit": git_commit, "source_hash": source_hash, "config_sha256": config_sha256,
        "environment_hash": environment_hash, "completed_experiments": list(EXPERIMENT_ORDER),
        "stage_markers": stage_refs, "artifacts": artifacts, "artifacts_locked_read_only": True,
    }
    return atomic_write_json(artifact_root / "audit" / "completion.json", payload)
