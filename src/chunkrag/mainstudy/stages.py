"""Concrete stage dispatch (Specification E0--E7 and Section 29).

Phase 3 never calls these functions with ``execute=True``. They are the Phase-4 stage
entry points and contain no legacy-pipeline dependency.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .artifacts import ArtifactStore
from .canonical import atomic_write_json, canonical_json_hash, read_jsonl
from .constants import PROTOCOL_ID
from .data import (
    cluster_records,
    load_pinned_dataset,
    materialize_hotpot_rows,
    materialize_squad_rows,
    materialize_techqa_rows,
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
        return [atomic_write_json(store.root / "manifests" / "dataset_manifest.json", {"schema_version": PROTOCOL_ID, "datasets": manifests})]
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
    for entry in getattr(raw, "cache_files", []):
        filename = Path(entry["filename"])
        cache_files.append({"name": filename.name, "sha256": file_sha256(filename)})
    dataset_manifest = {
        "schema_version": PROTOCOL_ID, "dataset": item.dataset,
        "repository": dataset_spec["repository"], "config": dataset_spec.get("config"),
        "split": dataset_spec["split"], "revision": dataset_spec["revision"],
        "fingerprint": getattr(raw, "_fingerprint", None), "rows_before_eligibility": len(rows),
        "selected_questions": len(questions), "corpus_documents": len(corpus),
        "cache_files": sorted(cache_files, key=lambda row: row["name"]),
        "license": getattr(getattr(raw, "info", None), "license", None),
        "artifact_hashes": [reference.sha256 for reference in references],
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
