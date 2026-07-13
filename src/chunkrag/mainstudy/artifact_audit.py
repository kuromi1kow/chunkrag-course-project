"""Fail-closed validation of the canonical E0--E6 artifact graph before E7 completion."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .canonical import canonical_json_hash, file_sha256, identifier_hash, read_json, read_jsonl, tree_sha256
from .constants import DATASET_ORDER, EXPECTED_QUESTION_COUNTS, JITTER_SEEDS, POLICY_ORDER, PROTOCOL_ID, PROTOCOL_SHA256
from .experiments import condition_ids_e1, condition_ids_e2
from .prompts import prompt_template_hash
from .evaluation import techqa_judge_template_hash
from .schemas import validate_records


class ArtifactAuditError(RuntimeError):
    pass


def _require(path: Path) -> Path:
    if not path.is_file():
        raise ArtifactAuditError(f"Missing canonical artifact: {path}")
    return path


def _jsonl(path: Path, schema: str, expected: int | None = None) -> list[dict[str, Any]]:
    rows = read_jsonl(_require(path))
    validate_records(schema, rows)
    if expected is not None and len(rows) != expected:
        raise ArtifactAuditError(f"Record count mismatch for {path}: {len(rows)} != {expected}")
    return rows


def validate_main_artifacts(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    report: dict[str, Any] = {"schema_version": PROTOCOL_ID, "datasets": {}, "status": "valid"}
    from huggingface_hub import snapshot_download
    model_snapshots: dict[str, str] = {}
    for role, spec in config["models"].items():
        snapshot = Path(snapshot_download(spec["repository"], revision=spec["revision"], local_files_only=True))
        model_snapshots[role] = tree_sha256(snapshot)
    report["model_snapshots"] = model_snapshots
    for required in (
        root / "manifests" / "dataset_manifest.json", root / "manifests" / "hash-manifest.json",
        root / "audit" / "e0-validation.json",
        root / "audit" / "statistical-self-tests.json",
    ):
        _require(required)
    retrieval_hashes: set[str] = set()
    gold_hashes: set[str] = set()
    generation_by_hash: dict[str, dict[str, Any]] = {}
    for dataset in DATASET_ORDER:
        expected = EXPECTED_QUESTION_COUNTS[dataset]
        question_path = root / "manifests" / "questions" / f"{dataset}.jsonl"
        corpus_path = root / "manifests" / "corpora" / f"{dataset}.jsonl"
        questions = _jsonl(question_path, "question", expected)
        corpus = _jsonl(corpus_path, "corpus")
        corpus_by_id = {row["document_id"]: row for row in corpus}
        _jsonl(root / "manifests" / "clusters" / f"{dataset}.jsonl", "cluster")
        gold = _jsonl(root / "manifests" / "gold" / f"{dataset}.jsonl", "gold", expected)
        gold_hashes.update(canonical_json_hash(row) for row in gold)
        dataset_manifest = read_json(_require(root / "manifests" / "datasets" / f"{dataset}.json"))
        for key, path in (("question_manifest_sha256", question_path), ("corpus_manifest_sha256", corpus_path)):
            if dataset_manifest.get(key) != file_sha256(path):
                raise ArtifactAuditError(f"E0 manifest hash mismatch: {dataset}/{key}")
        question_hash, corpus_hash = file_sha256(question_path), file_sha256(corpus_path)
        chunk_by_condition: dict[str, dict[str, dict[str, Any]]] = {}
        for condition in condition_ids_e1():
            chunk_path = root / "chunks" / dataset / f"{condition}.jsonl"
            chunks = _jsonl(chunk_path, "chunk")
            chunk_by_condition[condition] = {row["chunk_id"]: row for row in chunks}
            by_document: dict[str, list[dict[str, Any]]] = {}
            for chunk in chunks:
                if not 0 < int(chunk["token_count"]) <= 254:
                    raise ArtifactAuditError(f"Invalid canonical chunk length: {dataset}/{condition}")
                by_document.setdefault(str(chunk["document_id"]), []).append(chunk)
                expected_chunk_id = identifier_hash(dataset, condition, chunk["document_id"], chunk["token_start"], chunk["token_end"], chunk["text_sha256"])
                if chunk["chunk_id"] != expected_chunk_id:
                    raise ArtifactAuditError(f"Chunk identifier mismatch: {dataset}/{condition}")
            if set(by_document) != set(corpus_by_id):
                raise ArtifactAuditError(f"Chunk documents do not equal corpus documents: {dataset}/{condition}")
            for document_id, document_chunks in by_document.items():
                ordered = sorted(document_chunks, key=lambda row: int(row["ordinal"]))
                if "".join(str(row["text"]) for row in ordered) != corpus_by_id[document_id]["text"]:
                    raise ArtifactAuditError(f"Chunk round-trip mismatch: {dataset}/{condition}/{document_id}")
            traces = _jsonl(root / "retrieval" / "primary" / dataset / f"{condition}.jsonl", "retrieval", expected)
            upstream = canonical_json_hash([question_hash, corpus_hash, file_sha256(chunk_path)])
            for trace in traces:
                if trace["question_manifest_hash"] != question_hash or trace["corpus_manifest_hash"] != corpus_hash or trace["upstream_hash"] != upstream:
                    raise ArtifactAuditError(f"Broken primary retrieval provenance: {dataset}/{condition}")
                if trace["retrieval_id"] != identifier_hash(PROTOCOL_SHA256, trace["question_id"], trace["condition_id"], trace["config_hash"]):
                    raise ArtifactAuditError(f"Retrieval identifier mismatch: {dataset}/{condition}")
                if any(chunk_id not in chunk_by_condition[condition] for chunk_id in trace["top16_chunk_ids"]):
                    raise ArtifactAuditError(f"Unknown retrieved chunk: {dataset}/{condition}")
                if len(trace["top16_chunk_ids"]) != 16 or len(trace["dense_candidates"]) != 50 or len(trace["sparse_candidates"]) != 50 or len(trace["fused_candidates"]) != 50 or len(trace["reranked_candidates"]) != 50:
                    raise ArtifactAuditError(f"Retrieval depth mismatch: {dataset}/{condition}")
                retrieval_hashes.add(canonical_json_hash(trace))
            for required in (
                root / "analysis" / "retrieval" / "primary" / dataset / f"{condition}.jsonl",
                root / "audit" / "cost" / "primary" / dataset / f"{condition}.json",
                root / "audit" / "encoder-exposure" / "primary" / dataset / f"{condition}.json",
            ):
                _require(required)
            cost = read_json(root / "audit" / "cost" / "primary" / dataset / f"{condition}.json")
            if cost.get("warmup_questions") != 5 or cost.get("measured_questions") != expected or cost.get("index_vectors") != len(chunks) or cost.get("embedding_dtype") != "float32":
                raise ArtifactAuditError(f"Invalid primary operational audit: {dataset}/{condition}")
            hardware = cost.get("hardware", {})
            if not hardware.get("node") or not hardware.get("cuda_build") or not hardware.get("gpus") or any("A100" not in gpu for gpu in hardware["gpus"]):
                raise ArtifactAuditError(f"Noncanonical primary timing hardware: {dataset}/{condition}")
        for embedder in ("bge", "minilm"):
            for stack in ("dense", "hybrid", "hybrid-rerank"):
                for policy in POLICY_ORDER:
                    condition = f"{embedder}__{stack}__{policy}"
                    traces = _jsonl(root / "retrieval" / "secondary" / dataset / f"{condition}.jsonl", "retrieval", expected)
                    chunk_path = root / "chunks" / dataset / f"{policy}.jsonl"
                    upstream = canonical_json_hash([question_hash, corpus_hash, file_sha256(chunk_path)])
                    if any(row["upstream_hash"] != upstream for row in traces):
                        raise ArtifactAuditError(f"Broken E5 retrieval provenance: {dataset}/{condition}")
                    if any(row["retrieval_id"] != identifier_hash(PROTOCOL_SHA256, row["question_id"], row["condition_id"], row["config_hash"]) for row in traces):
                        raise ArtifactAuditError(f"E5 retrieval identifier mismatch: {dataset}/{condition}")
                    if any(len(row["top16_chunk_ids"]) != 16 or len(row["dense_candidates"]) != 50 for row in traces):
                        raise ArtifactAuditError(f"E5 retrieval depth mismatch: {dataset}/{condition}")
                    for required in (
                        root / "analysis" / "retrieval" / "secondary" / dataset / f"{condition}.jsonl",
                        root / "audit" / "cost" / "secondary" / dataset / f"{condition}.json",
                        root / "audit" / "encoder-exposure" / "secondary" / dataset / f"{condition}.json",
                    ):
                        _require(required)
                    cost = read_json(root / "audit" / "cost" / "secondary" / dataset / f"{condition}.json")
                    if cost.get("warmup_questions") != 5 or cost.get("measured_questions") != expected or cost.get("embedding_dtype") != "float32":
                        raise ArtifactAuditError(f"Invalid E5 operational audit: {dataset}/{condition}")
                    hardware = cost.get("hardware", {})
                    if not hardware.get("node") or not hardware.get("cuda_build") or not hardware.get("gpus") or any("A100" not in gpu for gpu in hardware["gpus"]):
                        raise ArtifactAuditError(f"Noncanonical E5 timing hardware: {dataset}/{condition}")
                _require(root / "analysis" / "retrieval" / "secondary" / dataset / f"{embedder}__{stack}__paired-effects.json")
        for role, conditions in (
            ("mistral", [*condition_ids_e2(), "gold-1024", "gold-4096"]),
            ("qwen", ([f"{policy}__matched-4096" for policy in POLICY_ORDER] + ["gold-4096"]) if dataset != "techqa" else []),
        ):
            for condition in conditions:
                generation_rows: list[dict[str, Any]] = []
                for path in sorted((root / "generation" / role / dataset / condition).glob("part-*.jsonl")):
                    generation_rows.extend(_jsonl(path, "generation"))
                if len(generation_rows) != expected:
                    raise ArtifactAuditError(f"Generation count mismatch: {role}/{dataset}/{condition}")
                if role == "mistral":
                    merged_rows = _jsonl(root / "generation" / role / dataset / f"{condition}.jsonl", "generation", expected)
                    if {row["record_hash"] for row in merged_rows} != {row["record_hash"] for row in generation_rows}:
                        raise ArtifactAuditError(f"Merged generation content mismatch: {role}/{dataset}/{condition}")
                for row in generation_rows:
                    if row["prompt_version_hash"] != prompt_template_hash(dataset):
                        raise ArtifactAuditError(f"Prompt hash mismatch: {role}/{dataset}/{condition}")
                    if row["upstream_hash"] not in retrieval_hashes | gold_hashes:
                        raise ArtifactAuditError(f"Unknown generation upstream hash: {role}/{dataset}/{condition}")
                    if row["model_repository"] != config["models"][role]["repository"] or row["model_revision"] != config["models"][role]["revision"] or row["model_snapshot_hash"] != model_snapshots[role]:
                        raise ArtifactAuditError(f"Generation model snapshot mismatch: {role}/{dataset}/{condition}")
                    if row["generation_id"] != identifier_hash(row["retrieval_or_gold_hash"], row["question_id"], row["model_snapshot_hash"], row["packing_id"], row["prompt_version_hash"]):
                        raise ArtifactAuditError(f"Generation identifier mismatch: {role}/{dataset}/{condition}")
                    generation_by_hash[row["record_hash"]] = row
                evaluations: list[dict[str, Any]] = []
                for path in sorted((root / "evaluation" / "automatic" / role / dataset / condition).glob("part-*.jsonl")):
                    evaluations.extend(_jsonl(path, "evaluation"))
                if len(evaluations) != expected or any(row["upstream_hash"] not in generation_by_hash for row in evaluations):
                    raise ArtifactAuditError(f"Automatic evaluation chain mismatch: {role}/{dataset}/{condition}")
                if any(row["evaluation_id"] != identifier_hash(row["generation_id"], row["evaluator_config_hash"]) for row in evaluations):
                    raise ArtifactAuditError(f"Evaluation identifier mismatch: {role}/{dataset}/{condition}")
        _require(root / "analysis" / "e3" / f"{dataset}-gold-gaps.json")
        if dataset != "techqa":
            _require(root / "analysis" / "e6" / f"{dataset}.json")
        report["datasets"][dataset] = {"questions": len(questions), "documents": len(corpus), "status": "valid"}
    judge_rows = []
    for path in sorted((root / "evaluation" / "judge" / "techqa").glob("**/part-*.jsonl")):
        judge_rows.extend(_jsonl(path, "evaluation"))
    if len(judge_rows) != 9_900 or any(row["upstream_hash"] not in generation_by_hash for row in judge_rows):
        raise ArtifactAuditError("TechQA judge trace count or generation links are invalid")
    judge_hash = techqa_judge_template_hash(config["models"]["qwen"])
    for row in judge_rows:
        judge = row.get("judge", {})
        if row.get("evaluator_config_hash") != judge_hash or judge.get("prompt_version") != "techqa-judge-v1" or judge.get("model_repository") != config["models"]["qwen"]["repository"] or judge.get("model_revision") != config["models"]["qwen"]["revision"] or not judge.get("messages"):
            raise ArtifactAuditError("TechQA judge provenance is incomplete or mismatched")
        if judge.get("model_snapshot_hash") != model_snapshots["qwen"]:
            raise ArtifactAuditError("TechQA judge model snapshot hash mismatch")
        if row["evaluation_id"] != identifier_hash(row["generation_id"], row["evaluator_config_hash"]):
            raise ArtifactAuditError("TechQA judge evaluation identifier mismatch")
    _require(root / "evaluation" / "human" / "techqa-package.json")
    _require(root / "evaluation" / "human" / "judge-validation.json")
    for label in ("human-labels-a.jsonl", "human-labels-b.jsonl", "human-adjudicated.jsonl"):
        _require(root / "evaluation" / "human" / label)
    report["hash_chain"] = {
        "retrieval_records": len(retrieval_hashes), "generation_records": len(generation_by_hash),
        "judge_records": len(judge_rows), "status": "valid",
    }
    return report
