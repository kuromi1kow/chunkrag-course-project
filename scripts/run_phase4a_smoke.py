#!/usr/bin/env python3
"""Execute the non-scientific five-question Phase 4A pipeline smoke test.

This command never writes to the canonical ``chunkrag-main-v1`` artifact root.  It
uses the frozen configuration and the real E0--E6 implementation on five already
frozen questions per dataset, then performs a smoke-specific E7/lock audit.  Smoke
outputs are explicitly ineligible for paper conclusions or canonical completion.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

from chunkrag.mainstudy import execution
from chunkrag.mainstudy.artifacts import ArtifactStore, validate_record_links
from chunkrag.mainstudy.canonical import (
    atomic_write_json, atomic_write_jsonl, canonical_json_hash, file_sha256,
    identifier_hash, read_json, read_jsonl, source_sha256, tree_sha256,
)
from chunkrag.mainstudy.checkpoint import ShardCheckpoint, merge_shards
from chunkrag.mainstudy.constants import (
    DATASET_ORDER, JITTER_SEEDS, PROTOCOL_ID, PROTOCOL_SHA256, STRUCTURED_POLICIES,
)
from chunkrag.mainstudy.data import (
    cluster_records, load_pinned_dataset, materialize_hotpot_rows,
    materialize_squad_rows, materialize_techqa_rows,
)
from chunkrag.mainstudy.environment import (
    environment_manifest, require_canonical_runtime, require_clean_git,
)
from chunkrag.mainstudy.evaluation import techqa_judge_template_hash
from chunkrag.mainstudy.experiments import WorkItem
from chunkrag.mainstudy.gold import gold_manifest
from chunkrag.mainstudy.human import HUMAN_CONDITIONS, agreement_report, blindness_scan, validate_label_rows
from chunkrag.mainstudy.prompts import prompt_template_hash
from chunkrag.mainstudy.protocol import ProtocolError, load_protocol_config, repo_root
from chunkrag.mainstudy.schemas import validate_record
from chunkrag.mainstudy.statistics import judge_acceptance


QUESTION_COUNT = 5
BASE_COMMIT = "0043d951cef84be6c737b3a1cc90e1948579e25c"
E1_CONDITIONS = [
    "fixed192", *STRUCTURED_POLICIES,
    *(f"{policy}-jitter-{seed}" for policy in STRUCTURED_POLICIES for seed in JITTER_SEEDS),
]
TECHQA_GENERATION = {
    "fixed192": "fixed192__matched-4096",
    "recursive192": "recursive192__matched-4096",
    "sentence192": "sentence192__matched-4096",
    "semantic192": "semantic192__matched-4096",
    "semantic192-jitter-1103": "semantic192-jitter-1103__matched-4096",
    "gold": "gold-4096",
}


class PlannedInterruption(BaseException):
    pass


def _validate_output_paths(artifact_root: Path, paper_output_dir: Path) -> None:
    canonical_parts = Path("chunkrag-main-v1").parts
    artifact_parts = artifact_root.resolve().parts
    if any(
        artifact_parts[index:index + len(canonical_parts)] == canonical_parts
        for index in range(len(artifact_parts) - len(canonical_parts) + 1)
    ):
        raise ProtocolError("Smoke output must not overlap the canonical artifact root")
    for path, label in ((artifact_root, "artifact"), (paper_output_dir, "paper-output")):
        if path.exists() and any(path.iterdir()):
            raise ProtocolError(f"Phase 4A requires an empty {label} directory: {path}")


def _write_stage(root: Path, stage: str, work: list[str], hashes: list[str]) -> str:
    return atomic_write_json(root / "audit" / "smoke-stages" / f"{stage}.json", {
        "schema_version": PROTOCOL_ID, "protocol_sha256": PROTOCOL_SHA256,
        "stage": stage, "smoke_only": True, "work_ids": work,
        "artifact_hashes": sorted(set(hashes)), "status": "complete",
    })


def _subset_corpus(dataset: str, corpus: list[dict[str, Any]], questions: list[dict[str, Any]], rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Keep smoke evidence plus a deterministic retrieval-depth exercise slice."""
    if dataset in {"squad_v2", "techqa"}:
        wanted = {document_id for question in questions for document_id in question["gold_document_ids"]}
    else:
        selected_ids = {question["question_id"] for question in questions}
        selected_indices = {index for index, row in enumerate(rows) if str(row["id"]) in selected_ids}
        wanted = {
            document["document_id"] for document in corpus
            if any(int(source["row_index"]) in selected_indices for source in document["source_provenance"])
        }
    # The additional documents are an explicitly non-scientific smoke fixture.  They
    # are drawn without a model-dependent decision and guarantee that top-50/top-16
    # retrieval paths can be exercised without materializing every canonical document.
    wanted.update(row["document_id"] for row in sorted(corpus, key=lambda item: item["document_id"])[:60])
    return [row for row in corpus if row["document_id"] in wanted]


def _materialize_smoke(config: Mapping[str, Any], store: ArtifactStore) -> tuple[dict[str, list[str]], list[str]]:
    ids: dict[str, list[str]] = {}
    hashes: list[str] = []
    manifests: list[dict[str, Any]] = []
    for dataset in DATASET_ORDER:
        raw = load_pinned_dataset(config["datasets"][dataset])
        rows = list(raw)
        materializer = {
            "squad_v2": materialize_squad_rows,
            "hotpot_qa": materialize_hotpot_rows,
            "techqa": materialize_techqa_rows,
        }[dataset]
        corpus, frozen = materializer(rows, config["datasets"][dataset]["revision"])
        corpus_again, frozen_again = materializer(rows, config["datasets"][dataset]["revision"])
        if canonical_json_hash([corpus, frozen]) != canonical_json_hash([corpus_again, frozen_again]):
            raise ProtocolError(f"Repeated E0 materialization differs: {dataset}")
        questions = sorted(frozen, key=lambda row: (int(row["selection_rank"]), row["question_id"]))[:QUESTION_COUNT]
        if [int(row["selection_rank"]) for row in questions] != list(range(QUESTION_COUNT)):
            raise ProtocolError(f"Smoke did not select the first frozen ranks: {dataset}")
        smoke_corpus = _subset_corpus(dataset, corpus, questions, rows)
        corpus_by_id = {row["document_id"]: row for row in smoke_corpus}
        if any(document_id not in corpus_by_id for row in questions for document_id in row["gold_document_ids"]):
            raise ProtocolError(f"Smoke corpus dropped gold evidence: {dataset}")
        clusters = cluster_records(dataset, questions)
        gold = [gold_manifest(question, corpus_by_id) for question in questions]
        refs = [
            store.write_jsonl(f"manifests/corpora/{dataset}.jsonl", smoke_corpus, "corpus", "document_id"),
            store.write_jsonl(f"manifests/questions/{dataset}.jsonl", questions, "question", "question_id"),
            store.write_jsonl(f"manifests/clusters/{dataset}.jsonl", clusters, "cluster", "cluster_id"),
            store.write_jsonl(f"manifests/gold/{dataset}.jsonl", gold, "gold", "gold_id"),
        ]
        ids[dataset] = [row["question_id"] for row in questions]
        cache_paths = [Path(entry["filename"]).resolve() for entry in getattr(raw, "cache_files", [])]
        cache_root = Path(os.path.commonpath([str(path.parent) for path in cache_paths])) if cache_paths else None
        manifest = {
            "schema_version": PROTOCOL_ID, "smoke_only": True, "dataset": dataset,
            "repository": config["datasets"][dataset]["repository"],
            "revision": config["datasets"][dataset]["revision"],
            "fingerprint": getattr(raw, "_fingerprint", None), "frozen_full_count": len(frozen),
            "smoke_question_ids": ids[dataset], "smoke_documents": len(smoke_corpus),
            "cache_snapshot_sha256": tree_sha256(cache_root, cache_paths) if cache_root else canonical_json_hash([]),
            "question_sha256": refs[1].sha256, "corpus_sha256": refs[0].sha256,
        }
        hashes.extend(reference.sha256 for reference in refs)
        hashes.append(atomic_write_json(store.root / "manifests" / "datasets" / f"{dataset}.json", manifest))
        manifests.append(manifest)
    hashes.append(atomic_write_json(store.root / "manifests" / "smoke-datasets.json", {
        "schema_version": PROTOCOL_ID, "protocol_sha256": PROTOCOL_SHA256,
        "smoke_only": True, "question_count_per_dataset": QUESTION_COUNT,
        "datasets": manifests,
    }))
    return ids, hashes


def _run_item(item: WorkItem, config: Mapping[str, Any], store: ArtifactStore) -> list[str]:
    return execution.HANDLERS[item.experiment](item, config, store)


def _canonical_smoke_shard_question_ids(
    store: ArtifactStore, dataset: str, shard_index: int, frozen_smoke_ids: list[str],
) -> list[str]:
    """Return the exact ID order used by canonical generation checkpointing."""
    canonical = [
        str(row["question_id"])
        for row in execution._questions_for_shard(store, dataset, shard_index)
    ]
    if len(canonical) != len(set(canonical)):
        raise ProtocolError(f"Canonical smoke shard contains duplicate IDs: {dataset}/{shard_index}")
    if set(canonical) != set(frozen_smoke_ids):
        missing = sorted(set(frozen_smoke_ids) - set(canonical))
        extra = sorted(set(canonical) - set(frozen_smoke_ids))
        raise ProtocolError(
            f"Canonical smoke shard substituted question IDs: missing={missing}, extra={extra}"
        )
    return canonical


def _build_human_package(store: ArtifactStore) -> tuple[list[dict[str, Any]], list[str]]:
    questions = read_jsonl(store.root / "manifests" / "questions" / "techqa.jsonl")
    question_by_id = {row["question_id"]: row for row in questions}
    generations: dict[tuple[str, str], dict[str, Any]] = {}
    for label, condition in TECHQA_GENERATION.items():
        for row in read_jsonl(store.root / "generation" / "mistral" / "techqa" / condition / "part-000.jsonl"):
            generations[(row["question_id"], label)] = row
    package: list[dict[str, Any]] = []
    linkage: list[dict[str, Any]] = []
    for question_id in sorted(question_by_id):
        question = question_by_id[question_id]
        candidates = []
        for label in HUMAN_CONDITIONS:
            generation = generations[(question_id, label)]
            artifact_hash = canonical_json_hash(generation)
            annotation_id = identifier_hash("human", question_id, artifact_hash)
            order_hash = identifier_hash("chunkrag-human-order-v1", question_id, artifact_hash)
            candidates.append({
                "annotation_record_id": annotation_id, "order_hash": order_hash,
                "question": question["question"], "reference": question["references"][0],
                "candidate": generation["normalized_output"], "groundedness_subset": True,
                "consumed_context": generation["consumed_context"],
            })
            linkage.append({"annotation_record_id": annotation_id, "generation_id": generation["generation_id"], "condition": label})
        package.extend(sorted(candidates, key=lambda row: row["order_hash"]))
    blindness_scan(package)
    hashes = [
        atomic_write_json(store.root / "evaluation" / "human" / "techqa-package.json", {"schema_version": PROTOCOL_ID, "smoke_only": True, "records": package}),
        atomic_write_json(store.root / "evaluation" / "human" / "techqa-training.json", {"schema_version": PROTOCOL_ID, "smoke_only": True, "records": package[:20]}),
        atomic_write_json(store.root / "evaluation" / "human" / "techqa-linkage-private.json", {"schema_version": PROTOCOL_ID, "smoke_only": True, "records": linkage}),
    ]
    expected = {row["annotation_record_id"] for row in package}
    label_rows = []
    for row in package:
        value = int(row["annotation_record_id"][:2], 16) % 3
        label_rows.append({
            "annotation_record_id": row["annotation_record_id"], "annotator_code": "SMOKE",
            "correctness": value, "completeness": value, "groundedness": value,
            "cannot_assess_reason": None, "completed_utc": "SMOKE-NOT-HUMAN",
            "rubric_version": "smoke-schema-v1",
        })
    validate_label_rows(label_rows, expected, adjudicated=False)
    validate_label_rows(label_rows, expected, adjudicated=True)
    for name in ("human-labels-a.jsonl", "human-labels-b.jsonl", "human-adjudicated.jsonl"):
        hashes.append(atomic_write_jsonl(store.root / "evaluation" / "human" / name, label_rows, "annotation_record_id"))
    return package, hashes


def _human_validation(store: ArtifactStore, package: list[Mapping[str, Any]]) -> str:
    root = store.root / "evaluation" / "human"
    labels_a, labels_b = read_jsonl(root / "human-labels-a.jsonl"), read_jsonl(root / "human-labels-b.jsonl")
    labels = {row["annotation_record_id"]: row for row in read_jsonl(root / "human-adjudicated.jsonl")}
    linkage = {row["annotation_record_id"]: row for row in read_json(root / "techqa-linkage-private.json")["records"]}
    judges: dict[str, Mapping[str, Any]] = {}
    invalid: dict[str, list[bool]] = defaultdict(list)
    for path in sorted((store.root / "evaluation" / "judge" / "techqa").glob("**/part-*.jsonl")):
        for row in read_jsonl(path):
            judges[row["generation_id"]] = row["judge"]
            invalid[path.parent.name].append(not any(attempt["status"] == "parsed" for attempt in row["judge"]["attempts"]))
    human = {key: [] for key in ("correctness", "completeness", "groundedness")}
    judge = {key: [] for key in human}
    for row in package:
        label = labels[row["annotation_record_id"]]
        parsed = judges[linkage[row["annotation_record_id"]]["generation_id"]]["parsed"]
        for dimension in human:
            human[dimension].append(int(label[dimension]))
            judge[dimension].append(int(parsed[dimension]))
    validation = judge_acceptance(judge, human, invalid_fraction_by_condition={key: sum(values) / len(values) for key, values in invalid.items()})
    validation.update({"schema_version": PROTOCOL_ID, "smoke_only": True, "synthetic_labels": True, "agreement": agreement_report(labels_a, labels_b)})
    return atomic_write_json(root / "judge-validation.json", validation)


def _validate_ids(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    retrieval = [row for path in (root / "retrieval").glob("**/*.jsonl") for row in read_jsonl(path)]
    generation = [row for path in (root / "generation").glob("**/part-*.jsonl") for row in read_jsonl(path)]
    evaluation = [row for path in (root / "evaluation").glob("**/part-*.jsonl") for row in read_jsonl(path)]
    gold = [row for path in (root / "manifests" / "gold").glob("*.jsonl") for row in read_jsonl(path)]
    for dataset in DATASET_ORDER:
        manifest = read_json(root / "manifests" / "datasets" / f"{dataset}.json")
        spec = config["datasets"][dataset]
        if manifest["repository"] != spec["repository"] or manifest["revision"] != spec["revision"]:
            raise ProtocolError("Dataset repository or revision mismatch")
    for row in retrieval:
        validate_record("retrieval", row)
        expected = identifier_hash(PROTOCOL_SHA256, row["question_id"], row["condition_id"], row["config_hash"])
        if (
            row["retrieval_id"] != expected
            or len(row["dense_candidates"]) != 50
            or len(row["sparse_candidates"]) != 50
            or len(row["fused_candidates"]) != 50
            or len(row["top16_chunk_ids"]) != 16
        ):
            raise ProtocolError("Retrieval identifier or frozen retrieval depth mismatch")
    model_snapshots: dict[str, str] = {}
    for role, spec in config["models"].items():
        from huggingface_hub import snapshot_download

        snapshot = Path(snapshot_download(spec["repository"], revision=spec["revision"], local_files_only=True))
        model_snapshots[role] = tree_sha256(snapshot)
    generation_count = 0
    for role_dir in sorted(path for path in (root / "generation").iterdir() if path.is_dir()):
        role = role_dir.name
        for dataset_dir in sorted(path for path in role_dir.iterdir() if path.is_dir()):
            dataset = dataset_dir.name
            for path in sorted(dataset_dir.glob("**/part-*.jsonl")):
                for row in read_jsonl(path):
                    generation_count += 1
                    validate_record("generation", row)
                    expected = identifier_hash(row["retrieval_or_gold_hash"], row["question_id"], row["model_snapshot_hash"], row["packing_id"], row["prompt_version_hash"])
                    spec = config["models"][role]
                    if (
                        row["generation_id"] != expected
                        or row["prompt_version_hash"] != prompt_template_hash(dataset)
                        or row["model_repository"] != spec["repository"]
                        or row["model_revision"] != spec["revision"]
                        or row["model_snapshot_hash"] != model_snapshots[role]
                    ):
                        raise ProtocolError("Generation identifier, prompt hash, or model revision mismatch")
    for row in evaluation:
        validate_record("evaluation", row)
        if row["evaluation_id"] != identifier_hash(row["generation_id"], row["evaluator_config_hash"]):
            raise ProtocolError("Evaluation identifier mismatch")
        if row.get("judge") is not None:
            judge = row["judge"]
            spec = config["models"]["qwen"]
            if (
                row["evaluator_config_hash"] != techqa_judge_template_hash(spec)
                or judge["prompt_version"] != "techqa-judge-v1"
                or judge["model_repository"] != spec["repository"]
                or judge["model_revision"] != spec["revision"]
                or judge["model_snapshot_hash"] != model_snapshots["qwen"]
            ):
                raise ProtocolError("TechQA judge prompt or model provenance mismatch")
    validate_record_links([*retrieval, *gold], generation)
    validate_record_links(generation, evaluation)
    if generation_count != len(generation):
        raise ProtocolError("Generation validation did not cover every top-level generation record")
    return {
        "retrieval": len(retrieval), "generation": len(generation),
        "evaluation": len(evaluation), "model_snapshots": model_snapshots,
    }


def _deterministic_projection(row: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(row)
    result.pop("record_hash", None)
    result.pop("latency", None)
    result.pop("hardware", None)
    return result


def _rerun_one_shard(config: Mapping[str, Any], store: ArtifactStore) -> dict[str, Any]:
    root = store.root
    item = WorkItem("E2", "squad_v2", "fixed192__operational-1024", 0, QUESTION_COUNT, ("E1",))
    stable_paths = [
        root / "generation" / "mistral" / "squad_v2" / item.condition_id / "part-000.jsonl",
        root / "generation" / "mistral" / "squad_v2" / item.condition_id / "part-000.state.json",
        root / "evaluation" / "automatic" / "mistral" / "squad_v2" / item.condition_id / "part-000.jsonl",
    ]
    hashes_before = {path.relative_to(root).as_posix(): file_sha256(path) for path in stable_paths}
    execution.execute_e2(item, config, store)
    hashes_after = {path.relative_to(root).as_posix(): file_sha256(path) for path in stable_paths}
    idempotent_equal = hashes_before == hashes_after
    if not idempotent_equal:
        raise ProtocolError("Completed-shard rerun changed immutable smoke artifacts")
    rerun_root = root / "audit" / "deterministic-rerun"
    rerun_store = ArtifactStore(rerun_root)
    rerun_store.initialize()
    for kind in ("questions", "corpora", "clusters", "gold"):
        source = root / "manifests" / kind / "squad_v2.jsonl"
        target = rerun_root / "manifests" / kind / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    for relative in (Path("chunks/squad_v2/fixed192.jsonl"), Path("retrieval/primary/squad_v2/fixed192.jsonl")):
        target = rerun_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / relative, target)
    execution.execute_e2(item, config, rerun_store)
    original_path = root / "generation" / "mistral" / "squad_v2" / item.condition_id / "part-000.jsonl"
    rerun_path = rerun_root / "generation" / "mistral" / "squad_v2" / item.condition_id / "part-000.jsonl"
    original, rerun = read_jsonl(original_path), read_jsonl(rerun_path)
    original_eval = read_jsonl(root / "evaluation" / "automatic" / "mistral" / "squad_v2" / item.condition_id / "part-000.jsonl")
    rerun_eval = read_jsonl(rerun_root / "evaluation" / "automatic" / "mistral" / "squad_v2" / item.condition_id / "part-000.jsonl")
    core_equal = [_deterministic_projection(row) for row in original] == [_deterministic_projection(row) for row in rerun]
    metrics_equal = [row["metrics"] for row in original_eval] == [row["metrics"] for row in rerun_eval]
    result = {
        "completed_shard_idempotent_hashes_equal": idempotent_equal,
        "completed_shard_hashes_before": hashes_before,
        "completed_shard_hashes_after": hashes_after,
        "question_ids": [row["question_id"] for row in original],
        "generation_ids_equal": [row["generation_id"] for row in original] == [row["generation_id"] for row in rerun],
        "prompt_ids_equal": [row["prompt_token_ids"] for row in original] == [row["prompt_token_ids"] for row in rerun],
        "raw_outputs_equal": [row["raw_output"] for row in original] == [row["raw_output"] for row in rerun],
        "normalized_outputs_equal": [row["normalized_output"] for row in original] == [row["normalized_output"] for row in rerun],
        "metrics_equal": metrics_equal, "deterministic_projection_equal": core_equal,
        "raw_file_hash_equal": file_sha256(original_path) == file_sha256(rerun_path),
        "record_hashes_equal": [row["record_hash"] for row in original] == [row["record_hash"] for row in rerun],
    }
    if not all(result[key] for key in ("generation_ids_equal", "prompt_ids_equal", "raw_outputs_equal", "normalized_outputs_equal", "metrics_equal", "deterministic_projection_equal")):
        raise ProtocolError("Deterministic smoke rerun changed protocol-required content")
    return result


def _write_smoke_analysis(root: Path, completion: Path, output_dir: Path) -> dict[str, Any]:
    manifest = read_json(completion)
    if manifest.get("smoke_only") is not True or manifest.get("completed_stages") != [f"E{i}" for i in range(8)]:
        raise ProtocolError("Smoke analysis gate rejected completion manifest")
    if (root / "audit" / "smoke-analysis.lock.json").exists():
        raise ProtocolError("Smoke analysis is already locked")
    for reference in manifest["artifacts"]:
        path = root / reference["path"]
        if not path.is_file() or file_sha256(path) != reference["sha256"] or path.stat().st_size != reference["bytes"] or path.stat().st_mode & 0o222:
            raise ProtocolError(f"Smoke analysis gate rejected artifact: {reference['path']}")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"schema_version": PROTOCOL_ID, "smoke_only": True, "datasets": {}, "statistics_self_test": read_json(root / "audit" / "statistical-self-tests.json")}
    for dataset in DATASET_ORDER:
        questions = read_jsonl(root / "manifests" / "questions" / f"{dataset}.jsonl")
        summary["datasets"][dataset] = {"question_ids": [row["question_id"] for row in questions], "records": len(questions)}
    analysis_hash = atomic_write_json(root / "analysis" / "smoke-summary.json", summary)
    table = output_dir / "phase4a_smoke_table.tex"
    table.write_text("\\begin{tabular}{lr}\nStage & Status \\\\\n\\hline\n" + "\n".join(f"E{i} & complete \\\\" for i in range(8)) + "\n\\end{tabular}\n", encoding="utf-8")
    from chunkrag.mainstudy.paper import _load_pyplot

    plt = _load_pyplot()
    figure = output_dir / "phase4a_smoke_pipeline.pdf"
    fig, axis = plt.subplots(figsize=(7.0, 1.8)); axis.plot(range(8), [1] * 8, "o-"); axis.set_xticks(range(8), [f"E{i}" for i in range(8)]); axis.set_yticks([]); fig.savefig(figure, bbox_inches="tight"); plt.close(fig)
    outputs = [table, figure]
    paper_manifest = atomic_write_json(output_dir / "phase4a-paper-output-manifest.json", {"schema_version": PROTOCOL_ID, "smoke_only": True, "analysis_sha256": analysis_hash, "outputs": [{"path": path.name, "sha256": file_sha256(path), "bytes": path.stat().st_size} for path in outputs]})
    lock = atomic_write_json(root / "audit" / "smoke-analysis.lock.json", {"schema_version": PROTOCOL_ID, "smoke_only": True, "completion_sha256": file_sha256(completion), "analysis_sha256": analysis_hash, "paper_manifest_sha256": paper_manifest})
    for path in [root / "analysis" / "smoke-summary.json", root / "audit" / "smoke-analysis.lock.json", *outputs, output_dir / "phase4a-paper-output-manifest.json"]:
        path.chmod(path.stat().st_mode & ~0o222)
    return {"analysis_sha256": analysis_hash, "paper_manifest_sha256": paper_manifest, "lock_sha256": lock, "outputs": [str(path) for path in outputs]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--paper-output-dir", type=Path, required=True)
    args = parser.parse_args()
    _validate_output_paths(args.artifact_root, args.paper_output_dir)
    root, config = repo_root(), load_protocol_config()
    state = require_clean_git(root)
    if subprocess.run(["git", "merge-base", "--is-ancestor", BASE_COMMIT, state["commit"]], cwd=root).returncode != 0:
        raise ProtocolError("Smoke implementation is not descended from the authorized Phase 3.6 commit")
    lock = environment_manifest(root / "requirements-main-study.transitive.json", check_installed=True)
    require_canonical_runtime(lock, gpu_required=True)
    source_hash = source_sha256(root, root / "requirements-main-study.transitive.json")
    store = ArtifactStore(args.artifact_root); store.initialize()
    started = time.time(); stages: dict[str, str] = {}; work: dict[str, list[str]] = defaultdict(list); all_hashes: list[str] = []
    ids, hashes = _materialize_smoke(config, store); all_hashes += hashes; work["E0"] = [f"E0/{dataset}/materialize" for dataset in DATASET_ORDER]; stages["E0"] = _write_stage(store.root, "E0", work["E0"], hashes)
    from chunkrag.mainstudy.validation import validate_statistical_primitives
    statistical_hash = atomic_write_json(store.root / "audit" / "statistical-self-tests.json", validate_statistical_primitives()); all_hashes.append(statistical_hash)
    hashes = []
    for dataset in DATASET_ORDER:
        for condition in E1_CONDITIONS:
            item = WorkItem("E1", dataset, condition, None, QUESTION_COUNT, ("E0",)); hashes += _run_item(item, config, store); work["E1"].append(item.work_id)
    all_hashes += hashes; stages["E1"] = _write_stage(store.root, "E1", work["E1"], hashes)
    hashes = []; primary = WorkItem("E2", "squad_v2", "fixed192__operational-1024", 0, QUESTION_COUNT, ("E1",))
    original_retry, calls = execution._generate_with_retries, {"count": 0}
    def interrupt_after_two(*values: Any, **kwargs: Any):
        calls["count"] += 1
        if calls["count"] == 3: raise PlannedInterruption()
        return original_retry(*values, **kwargs)
    execution._generate_with_retries = interrupt_after_two
    try:
        _run_item(primary, config, store)
        raise ProtocolError("Planned checkpoint interruption did not occur")
    except PlannedInterruption:
        pass
    finally:
        execution._generate_with_retries = original_retry
    checkpoint_ids = _canonical_smoke_shard_question_ids(
        store, "squad_v2", 0, ids["squad_v2"],
    )
    checkpoint = ShardCheckpoint(store.root / "generation" / "mistral" / "squad_v2" / primary.condition_id, "E2", "squad_v2", primary.condition_id, 0, checkpoint_ids, config["config_sha256"], lock["lock_sha256"])
    checkpoint_state = checkpoint.validate_partial(lambda row: str(row["question_id"]))
    if len(checkpoint_state["completed"]) != 2: raise ProtocolError("Planned interruption did not preserve two records")
    interrupted_ids = list(checkpoint_state["completed"])
    recovered_id = checkpoint_state["completed"][-1]; checkpoint_state["completed"].remove(recovered_id); checkpoint_state["record_hashes"].pop(recovered_id)
    atomic_write_json(checkpoint.state_path, checkpoint_state, overwrite=True)
    with checkpoint.temp_path.open("ab") as handle: handle.write(b'{"incomplete"'); handle.flush(); os.fsync(handle.fileno())
    hashes += _run_item(primary, config, store); work["E2"].append(primary.work_id)
    resumed_state = read_json(checkpoint.state_path)
    if set(resumed_state["completed"]) != set(ids["squad_v2"]):
        raise ProtocolError("Checkpoint recovery/resume did not complete the frozen smoke shard")
    checkpoint_result = {
        "planned_interruption_after_records": len(interrupted_ids),
        "interrupted_question_ids": interrupted_ids,
        "state_record_recovered": recovered_id in resumed_state["completed"],
        "incomplete_tail_trimmed": not checkpoint.temp_path.exists(),
        "final_validated_sha256": checkpoint.validate_final(lambda row: str(row["question_id"])),
        "completed_question_ids": list(resumed_state["completed"]),
    }
    for dataset in DATASET_ORDER:
        if dataset != "squad_v2":
            item = WorkItem("E2", dataset, primary.condition_id, 0, QUESTION_COUNT, ("E1",)); hashes += _run_item(item, config, store); work["E2"].append(item.work_id)
    for condition in TECHQA_GENERATION.values():
        if condition == "gold-4096": continue
        item = WorkItem("E2", "techqa", condition, 0, QUESTION_COUNT, ("E1",))
        if item.work_id not in work["E2"]: hashes += _run_item(item, config, store); work["E2"].append(item.work_id)
    all_hashes += hashes; stages["E2"] = _write_stage(store.root, "E2", work["E2"], hashes)
    hashes = []
    for dataset in DATASET_ORDER:
        item = WorkItem("E3", dataset, "gold-4096", 0, QUESTION_COUNT, ("E2",)); hashes += _run_item(item, config, store); work["E3"].append(item.work_id)
    all_hashes += hashes; stages["E3"] = _write_stage(store.root, "E3", work["E3"], hashes)
    merged = store.root / "generation" / "mistral" / "squad_v2" / "fixed192__operational-1024.smoke-merged.jsonl"
    merge_hash = merge_shards([checkpoint.final_path], ids["squad_v2"], "question_id", merged, schema="generation", require_state=True, expected_config_sha256=config["config_sha256"], expected_environment_hash=lock["lock_sha256"]); all_hashes.append(merge_hash)
    merge_rows = read_jsonl(merged)
    merge_result = {
        "sha256": merge_hash, "records": len(merge_rows),
        "question_ids_exact": {row["question_id"] for row in merge_rows} == set(ids["squad_v2"]),
        "state_required": True, "provenance_validated": True,
    }
    if not merge_result["question_ids_exact"]:
        raise ProtocolError("Smoke merge lost or substituted a frozen question ID")
    package, hashes = _build_human_package(store)
    from chunkrag.mainstudy.completion import mark_work_complete
    package_item = WorkItem("E4", "techqa", "human-package", None, len(package), ("E3",)); mark_work_complete(store.root, package_item, hashes, git_commit=state["commit"], config_sha256=config["config_sha256"], environment_hash=lock["lock_sha256"]); work["E4"].append(package_item.work_id)
    for condition in TECHQA_GENERATION.values():
        item = WorkItem("E4", "techqa", f"judge__{condition}", 0, QUESTION_COUNT, ("E3",)); hashes += _run_item(item, config, store); work["E4"].append(item.work_id)
    hashes.append(_human_validation(store, package)); work["E4"].append("E4/techqa/human-validation")
    all_hashes += hashes; stages["E4"] = _write_stage(store.root, "E4", work["E4"], hashes)
    hashes = []
    for dataset in DATASET_ORDER:
        item = WorkItem("E5", dataset, "minilm__hybrid__fixed192", None, QUESTION_COUNT, ("E4",)); hashes += _run_item(item, config, store); work["E5"].append(item.work_id)
    all_hashes += hashes; stages["E5"] = _write_stage(store.root, "E5", work["E5"], hashes)
    hashes = []
    for dataset in ("squad_v2", "hotpot_qa"):
        for condition in ("fixed192__matched-4096", "gold-4096"):
            item = WorkItem("E6", dataset, condition, 0, QUESTION_COUNT, ("E5",)); hashes += _run_item(item, config, store); work["E6"].append(item.work_id)
    all_hashes += hashes; stages["E6"] = _write_stage(store.root, "E6", work["E6"], hashes)
    id_counts = _validate_ids(store.root, config); rerun = _rerun_one_shard(config, store)
    e7 = atomic_write_json(store.root / "audit" / "e7-smoke-validation.json", {"schema_version": PROTOCOL_ID, "smoke_only": True, "status": "complete", "identifier_counts": id_counts, "deterministic_rerun": rerun}); all_hashes.append(e7); work["E7"] = ["E7/global/audit"]; stages["E7"] = _write_stage(store.root, "E7", work["E7"], [e7])
    provenance = {"git_commit": state["commit"], "source_hash": source_hash, "protocol_sha256": PROTOCOL_SHA256, "config_sha256": config["config_sha256"], "environment_hash": lock["lock_sha256"], "hardware": lock["hardware"]}
    atomic_write_json(store.root / "audit" / "smoke-run.json", {"schema_version": PROTOCOL_ID, "smoke_only": True, **provenance, "base_commit": BASE_COMMIT, "question_ids": ids, "work": work, "checkpoint_recovery": checkpoint_result, "merge": merge_result, "runtime_seconds_before_analysis": time.time() - started, "status": "complete"})
    inventory = []
    excluded = {"audit/smoke-completion.json", "audit/smoke-analysis.lock.json", "analysis/smoke-summary.json"}
    for path in sorted(item for item in store.root.rglob("*") if item.is_file()):
        relative = path.relative_to(store.root).as_posix()
        if relative not in excluded: inventory.append({"path": relative, "sha256": file_sha256(path), "bytes": path.stat().st_size})
    completion = store.root / "audit" / "smoke-completion.json"
    atomic_write_json(completion, {"schema_version": PROTOCOL_ID, "smoke_only": True, **provenance, "completed_stages": [f"E{i}" for i in range(8)], "stage_hashes": stages, "artifacts": inventory, "artifacts_locked_read_only": True})
    store.lock_read_only(); completion.chmod(completion.stat().st_mode & ~0o222)
    analysis = _write_smoke_analysis(store.root, completion, args.paper_output_dir)
    second_gate_rejected = False
    try: _write_smoke_analysis(store.root, completion, args.paper_output_dir)
    except ProtocolError: second_gate_rejected = True
    if not second_gate_rejected: raise ProtocolError("Smoke analysis lock allowed a second execution")
    final = {"schema_version": PROTOCOL_ID, "smoke_only": True, **provenance, "question_ids": ids, "stage_hashes": stages, "checkpoint_recovery": checkpoint_result, "merge": merge_result, "completion_sha256": file_sha256(completion), "analysis": analysis, "analysis_second_run_rejected": second_gate_rejected, "deterministic_rerun": rerun, "runtime_seconds": time.time() - started, "status": "GO"}
    report = args.paper_output_dir / "phase4a-final-report.json"; atomic_write_json(report, final); report.chmod(report.stat().st_mode & ~0o222)
    print(json.dumps(final, sort_keys=True, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
