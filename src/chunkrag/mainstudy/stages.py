"""Concrete stage dispatch (Specification E0--E7 and Section 29).

Phase 3 never calls these functions with ``execute=True``. They are the Phase-4 stage
entry points and contain no legacy-pipeline dependency.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .artifacts import ArtifactStore
from .canonical import atomic_write_json, canonical_json_hash, read_jsonl, tree_sha256
from .constants import PROTOCOL_ID
from .data import (
    cluster_records,
    load_pinned_dataset,
    materialize_hotpot_rows,
    materialize_squad_rows,
    materialize_techqa_rows,
    eligible_row_count,
    validate_cluster_constraints,
)
from .experiments import WorkItem
from .gold import gold_manifest
from .protocol import ProtocolError


def execute_e0(item: WorkItem, config: Mapping[str, Any], store: ArtifactStore) -> list[str]:
    from .canonical import atomic_write_json, canonical_json_hash, file_sha256

    if item.dataset is None:
        manifests = []
        for dataset in config["dataset_order"]:
            path = store.root / "manifests" / "datasets" / f"{dataset}.json"
            if not path.is_file():
                raise ProtocolError(f"Cannot finalize E0 before dataset manifest exists: {dataset}")
            from .canonical import read_json
            manifests.append(read_json(path))
        dataset_hash = atomic_write_json(store.root / "manifests" / "dataset_manifest.json", {"schema_version": PROTOCOL_ID, "datasets": manifests})
        from .validation import validate_statistical_primitives
        statistical_validation = atomic_write_json(store.root / "audit" / "statistical-self-tests.json", validate_statistical_primitives())
        validation = atomic_write_json(store.root / "audit" / "e0-validation.json", {
            "schema_version": PROTOCOL_ID, "status": "valid", "datasets": [row["dataset"] for row in manifests],
            "question_counts": {row["dataset"]: row["selected_questions"] for row in manifests},
            "materialization_repeated": True,
        })
        hash_rows = []
        e0_files = [
            *(item for item in (store.root / "manifests").rglob("*") if item.is_file() and item.name != "hash-manifest.json"),
            *(item for item in (store.root / "analysis" / "design-sensitivity").glob("*.json") if item.is_file()),
            store.root / "audit" / "e0-validation.json",
            store.root / "audit" / "statistical-self-tests.json",
        ]
        for path in sorted(e0_files):
            hash_rows.append({"path": path.relative_to(store.root).as_posix(), "sha256": file_sha256(path), "bytes": path.stat().st_size})
        hash_manifest = atomic_write_json(store.root / "manifests" / "hash-manifest.json", {"schema_version": PROTOCOL_ID, "files": hash_rows})
        return [dataset_hash, hash_manifest, validation, statistical_validation]
    dataset_spec = config["datasets"][item.dataset]
    raw = load_pinned_dataset(dataset_spec)
    rows = list(raw)
    if item.dataset == "squad_v2":
        corpus, questions = materialize_squad_rows(rows, dataset_spec["revision"])
    elif item.dataset == "hotpot_qa":
        corpus, questions = materialize_hotpot_rows(rows, dataset_spec["revision"])
    else:
        corpus, questions = materialize_techqa_rows(rows, dataset_spec["revision"])
    if item.dataset == "squad_v2":
        corpus_second, questions_second = materialize_squad_rows(rows, dataset_spec["revision"])
    elif item.dataset == "hotpot_qa":
        corpus_second, questions_second = materialize_hotpot_rows(rows, dataset_spec["revision"])
    else:
        corpus_second, questions_second = materialize_techqa_rows(rows, dataset_spec["revision"])
    if canonical_json_hash([corpus, questions]) != canonical_json_hash([corpus_second, questions_second]):
        raise ProtocolError(f"Independent E0 materialization is not byte-identical: {item.dataset}")
    clusters = cluster_records(item.dataset, questions)
    validate_cluster_constraints(clusters, len(questions))
    corpus_by_id = {row["document_id"]: row for row in corpus}
    gold = [gold_manifest(question, corpus_by_id) for question in questions]
    references = [
        store.write_jsonl(f"manifests/corpora/{item.dataset}.jsonl", corpus, "corpus", "document_id"),
        store.write_jsonl(f"manifests/questions/{item.dataset}.jsonl", questions, "question", "question_id"),
        store.write_jsonl(f"manifests/clusters/{item.dataset}.jsonl", clusters, "cluster", "cluster_id"),
        store.write_jsonl(f"manifests/gold/{item.dataset}.jsonl", gold, "gold", "gold_id"),
    ]
    from .power import sensitivity_report

    sensitivity = sensitivity_report(item.dataset, [row["size"] for row in clusters])
    sensitivity_hash = atomic_write_json(store.root / "analysis" / "design-sensitivity" / f"{item.dataset}.json", sensitivity)
    cache_files = []
    cache_paths = [Path(entry["filename"]).resolve() for entry in getattr(raw, "cache_files", [])]
    common_cache_root = Path(__import__("os").path.commonpath([str(path.parent) for path in cache_paths])) if cache_paths else None
    for entry in getattr(raw, "cache_files", []):
        filename = Path(entry["filename"]).resolve()
        relative = filename.relative_to(common_cache_root).as_posix() if common_cache_root is not None else filename.name
        cache_files.append({"path": relative, "sha256": file_sha256(filename)})
    snapshot_hash = tree_sha256(common_cache_root, cache_paths) if common_cache_root is not None else canonical_json_hash([])
    dataset_manifest = {
        "schema_version": PROTOCOL_ID, "dataset": item.dataset,
        "repository": dataset_spec["repository"], "config": dataset_spec.get("config"),
        "split": dataset_spec["split"], "revision": dataset_spec["revision"],
        "fingerprint": getattr(raw, "_fingerprint", None), "rows_before_eligibility": len(rows),
        "rows_after_eligibility": eligible_row_count(item.dataset, rows),
        "selected_questions": len(questions), "corpus_documents": len(corpus),
        "cache_snapshot_sha256": snapshot_hash, "cache_files": sorted(cache_files, key=lambda row: row["path"]),
        "license": getattr(getattr(raw, "info", None), "license", None),
        "question_manifest_sha256": references[1].sha256, "corpus_manifest_sha256": references[0].sha256,
        "cluster_manifest_sha256": references[2].sha256, "gold_manifest_sha256": references[3].sha256,
    }
    dataset_hash = atomic_write_json(store.root / "manifests" / "datasets" / f"{item.dataset}.json", dataset_manifest)
    return [*(reference.sha256 for reference in references), sensitivity_hash, dataset_hash]


def _require_phase4_handler(item: WorkItem) -> None:
    # E1--E7 use the protocol modules directly through runner-specific handlers. This
    # exception is reachable only if a work item is invoked without its registered
    # handler, which is a repository validation failure rather than a scientific fallback.
    raise ProtocolError(f"No registered execution handler for {item.work_id}")


from .execution import HANDLERS as EXECUTION_HANDLERS

HANDLERS = {"E0": execute_e0, **EXECUTION_HANDLERS}


def execute_work_item(item: WorkItem, config: Mapping[str, Any], store: ArtifactStore) -> list[str]:
    handler = HANDLERS.get(item.experiment)
    if handler is None:
        _require_phase4_handler(item)
    return handler(item, config, store)
