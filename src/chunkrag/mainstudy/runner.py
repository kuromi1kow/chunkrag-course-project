"""Unified experiment runner (Specification Sections 26--29)."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

from .artifacts import ArtifactStore, run_manifest_template
from .checkpoint import merge_shards
from .canonical import read_json, read_jsonl
from .colab import validate_colab_runtime
from .completion import completed_stages, finalize_stage, mark_work_complete
from .environment import environment_manifest, require_clean_git
from .experiments import filter_plan, plan_experiment, validate_dependencies
from .logging import JsonlLogger, configure_console_logging
from .protocol import ProtocolError, load_protocol_config, repo_root
from .canonical import atomic_write_json, identifier_hash, source_sha256
from .validation import validate_repository


MODES = ("run", "dry-run", "validation-only", "merge-only")
PLATFORMS = ("local", "colab")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Protocol-frozen ChunkRAG E0--E7 runner")
    value.add_argument("--experiment", required=True, choices=[f"E{i}" for i in range(8)])
    value.add_argument("--mode", choices=MODES, default="dry-run")
    value.add_argument("--platform", choices=PLATFORMS, default="local")
    value.add_argument("--dataset")
    value.add_argument("--condition-id")
    value.add_argument("--shard-index", type=int)
    value.add_argument("--artifact-root", type=Path)
    value.add_argument("--completed", nargs="*", default=[])
    value.add_argument("--runtime-manifest", type=Path)
    value.add_argument("--shard-dir", type=Path)
    value.add_argument("--merge-output", type=Path)
    value.add_argument("--id-field", default="question_id")
    value.add_argument("--expected-ids", type=Path)
    value.add_argument("--verbose", action="store_true")
    return value


def build_plan(args: argparse.Namespace) -> list[dict[str, Any]]:
    items = filter_plan(
        plan_experiment(args.experiment), dataset=args.dataset,
        condition_id=args.condition_id, shard_index=args.shard_index,
    )
    if not items:
        raise ProtocolError("Runner filters selected no frozen work items")
    return [item.to_dict() for item in items]


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    configure_console_logging(args.verbose)
    root = repo_root()
    config = load_protocol_config()
    artifact_root = args.artifact_root or root / config["artifact_root"]
    if args.mode == "validation-only":
        print(json.dumps(validate_repository(root), sort_keys=True, indent=2))
        return 0
    items = filter_plan(
        plan_experiment(args.experiment), dataset=args.dataset,
        condition_id=args.condition_id, shard_index=args.shard_index,
    )
    if not items:
        raise ProtocolError("Runner filters selected no frozen work items")
    if args.mode == "dry-run":
        print(json.dumps([item.to_dict() for item in items], sort_keys=True, indent=2))
        return 0
    state = require_clean_git(root)
    lock = environment_manifest(root / "requirements-main-study.lock", check_installed=True)
    if args.platform == "colab":
        if args.runtime_manifest is None:
            raise ProtocolError("Colab execution requires --runtime-manifest")
        runtime = json.loads(args.runtime_manifest.read_text(encoding="utf-8"))
        validate_colab_runtime(runtime, expected_environment_hash=lock["lock_sha256"], expected_git_commit=state["commit"])
    store = ArtifactStore(artifact_root)
    store.initialize()
    actual_completed = completed_stages(artifact_root)
    validate_dependencies(args.experiment, actual_completed)
    if args.completed and not set(args.completed).issubset(set(actual_completed)):
        raise ProtocolError(f"Claimed --completed stages lack hashed completion markers: {sorted(set(args.completed) - set(actual_completed))}")
    logger = JsonlLogger(artifact_root / "audit" / "runner-events.jsonl")
    if args.mode == "merge-only":
        if args.shard_dir is None or args.merge_output is None:
            raise ProtocolError("Merge-only requires --shard-dir and --merge-output")
        if args.expected_ids is not None:
            expected_payload = read_json(args.expected_ids)
            expected_ids = list(expected_payload["expected_ids"] if isinstance(expected_payload, dict) else expected_payload)
        elif args.dataset is not None and args.id_field == "question_id":
            expected_ids = [row["question_id"] for row in read_jsonl(artifact_root / "manifests" / "questions" / f"{args.dataset}.jsonl")]
        else:
            raise ProtocolError("Merge-only requires --expected-ids unless merging question IDs for a dataset")
        digest = merge_shards(sorted(args.shard_dir.glob("part-*.jsonl")), expected_ids, args.id_field, args.merge_output)
        logger.emit("merge-completed", experiment=args.experiment, output=str(args.merge_output), sha256=digest)
        print(json.dumps({"output": str(args.merge_output), "sha256": digest}, sort_keys=True))
        return 0
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    produced: dict[str, list[str]] = {}
    from .stages import execute_work_item

    for item in items:
        logger.emit("work-started", work_id=item.work_id)
        hashes = execute_work_item(item, config, store)
        produced[item.work_id] = hashes
        marker_hash = mark_work_complete(
            artifact_root, item, hashes, git_commit=state["commit"],
            config_sha256=config["config_sha256"], environment_hash=lock["lock_sha256"],
        )
        logger.emit("work-completed", work_id=item.work_id, artifact_hashes=hashes)
    stage_hash = finalize_stage(artifact_root, args.experiment)
    if stage_hash:
        logger.emit("stage-completed", experiment=args.experiment, marker_sha256=stage_hash)
    manifest = run_manifest_template(
        git_commit=state["commit"],
        source_hash=source_sha256(root, root / "requirements-main-study.lock"),
        config_hash=config["config_sha256"], environment_hash=lock["lock_sha256"],
        planned_counts={item.work_id: item.expected_records for item in items},
        hardware=lock["hardware"],
    )
    manifest.update({
        "started_utc": started, "ended_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "completed_counts": {item.work_id: item.expected_records for item in items},
        "artifact_hashes": produced, "shards": [item.work_id for item in items],
        "status": "complete", "failures": [],
    })
    invocation_id = identifier_hash(state["commit"], started, *[item.work_id for item in items])
    atomic_write_json(artifact_root / "audit" / "invocations" / f"{invocation_id}.json", manifest)
    return 0
