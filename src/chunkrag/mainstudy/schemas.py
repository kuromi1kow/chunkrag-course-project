"""Strict artifact record validation (Specification Sections 10, 11.5, 23)."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from .constants import HASH_RE, PACKING_IDS, POLICY_ORDER, SCHEMA_VERSION


class SchemaError(ValueError):
    pass


COMMON_REQUIRED = {"schema_version"}
SCHEMAS: dict[str, set[str]] = {
    "question": {
        "question_id", "dataset", "revision", "selection_hash", "selection_rank",
        "question", "references", "gold_document_ids", "cluster_id", "eligibility",
        "gold_spans", "supporting_facts", "source_provenance",
    },
    "corpus": {
        "document_id", "dataset", "title", "text", "split", "revision", "text_sha256",
        "source_provenance",
    },
    "cluster": {"cluster_id", "dataset", "question_ids", "size"},
    "gold": {"gold_id", "question_id", "dataset", "ordered_units", "source_hash"},
    "chunk": {
        "chunk_id", "condition_id", "policy", "policy_version", "dataset", "document_id",
        "char_start", "char_end", "token_start", "token_end", "token_count", "text",
        "text_sha256", "preceding_separator", "following_separator", "ordinal",
        "final_short", "tokenizer_repository", "tokenizer_revision", "control_seed",
        "parent_chunk_count", "boundary_generation_hash",
    },
    "retrieval": {
        "retrieval_id", "question_id", "condition_id", "question_manifest_hash",
        "corpus_manifest_hash", "dense_candidates", "sparse_candidates", "fused_candidates",
        "reranked_candidates", "top16_chunk_ids", "latency", "memory", "config_hash",
        "upstream_hash",
    },
    "generation": {
        "generation_id", "question_id", "retrieval_or_gold_hash", "condition_id",
        "control_seed", "packing_id", "budget", "ranked_source_spans", "rendered_context",
        "consumed_context", "per_chunk_consumed_tokens", "prompt_version", "prompt_version_hash", "messages",
        "prompt_token_ids", "full_input_tokens", "used_input_tokens", "context_target",
        "truncation_location", "model_repository", "model_revision", "model_snapshot_hash",
        "dtype", "hardware", "raw_output", "normalized_output", "generated_tokens",
        "stopping_reason", "latency", "attempt_history", "upstream_hash",
        "record_hash",
    },
    "evaluation": {
        "evaluation_id", "generation_id", "references", "gold_document_ids", "metrics",
        "judge", "human_annotation_ids", "evaluator_config_hash", "upstream_hash",
    },
    "checkpoint": {
        "stage", "dataset", "condition_id", "shard_index", "expected_question_ids",
        "completed", "record_hashes", "protocol_sha256", "config_sha256", "environment_hash",
    },
    "run": {
        "protocol_id", "protocol_sha256", "git_commit", "dirty_worktree", "source_hash",
        "config_hash", "environment_lock_hash", "artifact_hashes", "model_snapshots",
        "hardware", "started_utc", "ended_utc", "planned_counts", "completed_counts",
        "shards", "status", "failures",
    },
}


def _require_hash(value: Any, label: str, *, allow_empty: bool = False) -> None:
    if allow_empty and value == "":
        return
    if not isinstance(value, str) or re.fullmatch(HASH_RE, value) is None:
        raise SchemaError(f"{label} must be a lowercase SHA-256")


def validate_record(schema: str, record: Mapping[str, Any], *, strict: bool = True) -> None:
    if schema not in SCHEMAS:
        raise SchemaError(f"Unknown schema: {schema}")
    if record.get("schema_version") != SCHEMA_VERSION:
        raise SchemaError(f"Invalid schema_version for {schema}: {record.get('schema_version')!r}")
    required = COMMON_REQUIRED | SCHEMAS[schema]
    missing = sorted(required - set(record))
    if missing:
        raise SchemaError(f"Missing {schema} fields: {', '.join(missing)}")
    if strict:
        extras = sorted(set(record) - required)
        if extras:
            raise SchemaError(f"Unknown {schema} fields: {', '.join(extras)}")
    for key in ("text_sha256", "selection_hash", "source_hash", "upstream_hash"):
        if key in record:
            _require_hash(record[key], f"{schema}.{key}")
    for key, value in record.items():
        if key.endswith("_hash") or key.endswith("_sha256"):
            _require_hash(value, f"{schema}.{key}")
    if schema == "chunk":
        if record["policy"] not in POLICY_ORDER:
            raise SchemaError(f"Unknown chunk policy: {record['policy']}")
        if not (0 <= record["char_start"] <= record["char_end"]):
            raise SchemaError("Invalid chunk character interval")
        if not (0 <= record["token_start"] <= record["token_end"]):
            raise SchemaError("Invalid chunk token interval")
        if record["token_count"] != record["token_end"] - record["token_start"]:
            raise SchemaError("Chunk token count disagrees with interval")
        _require_hash(record["chunk_id"], "chunk.chunk_id")
    elif schema == "retrieval":
        _require_hash(record["retrieval_id"], "retrieval.retrieval_id")
        if len(record["top16_chunk_ids"]) > 16:
            raise SchemaError("Retrieval top16 contains more than 16 chunks")
    elif schema == "generation":
        _require_hash(record["generation_id"], "generation.generation_id")
        if record["packing_id"] not in PACKING_IDS:
            raise SchemaError(f"Unknown packing ID: {record['packing_id']}")
        if record["budget"] not in (1024, 4096):
            raise SchemaError("Generation budget must be 1024 or 4096")
        from .canonical import canonical_json_hash
        payload = dict(record)
        declared = payload.pop("record_hash")
        _require_hash(declared, "generation.record_hash")
        if declared != canonical_json_hash(payload):
            raise SchemaError("Generation record hash does not match its canonical payload")
    elif schema == "evaluation":
        _require_hash(record["evaluation_id"], "evaluation.evaluation_id")
    elif schema == "checkpoint":
        if sorted(record["completed"]) != sorted(record["record_hashes"]):
            raise SchemaError("Checkpoint completed IDs and record hashes differ")


def validate_records(schema: str, records: list[Mapping[str, Any]]) -> None:
    seen: set[str] = set()
    id_field = {
        "question": "question_id", "corpus": "document_id", "cluster": "cluster_id",
        "gold": "gold_id", "chunk": "chunk_id", "retrieval": "retrieval_id",
        "generation": "generation_id", "evaluation": "evaluation_id",
    }.get(schema)
    for record in records:
        validate_record(schema, record)
        if id_field:
            identifier = str(record[id_field])
            if identifier in seen:
                raise SchemaError(f"Duplicate {schema} ID: {identifier}")
            seen.add(identifier)
