from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chunkrag.eaai_phase2.constants import (
    BASELINE_TREE_SHA256,
    CHUNKERS,
    DEVELOPMENT_SIZE,
    EXPECTED_DOCUMENTS,
    EXPECTED_ELIGIBLE_ROWS,
    HELDOUT_SIZE,
    PROTOCOL_COMMIT,
    RESERVE_SIZE,
    RUN_ID,
)
from chunkrag.eaai_phase2.io import canonical_json_bytes, sha256_bytes


EXPECTED_CHUNKERS = [
    {
        "name": "fixed_128",
        "type": "fixed",
        "chunk_size": 128,
        "chunk_overlap": 19,
        "enforce_token_limit": True,
    },
    {
        "name": "fixed_254",
        "type": "fixed",
        "chunk_size": 254,
        "chunk_overlap": 38,
        "enforce_token_limit": True,
    },
    {
        "name": "recursive_254",
        "type": "recursive",
        "chunk_size": 254,
        "chunk_overlap": 38,
        "enforce_token_limit": True,
    },
    {
        "name": "sentence_254",
        "type": "sentence",
        "chunk_size": 254,
        "chunk_overlap": 0,
        "enforce_token_limit": True,
    },
]


@dataclass(frozen=True, slots=True)
class Phase2Paths:
    repository: Path
    results_root: Path
    artifacts_root: Path
    run_results: Path
    run_artifacts: Path


def load_phase2_config(path: str | Path) -> tuple[dict[str, Any], str]:
    config_path = Path(path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_phase2_config(config)
    return config, sha256_bytes(canonical_json_bytes(config))


def validate_phase2_config(config: dict[str, Any]) -> None:
    checks = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "protocol_commit": PROTOCOL_COMMIT,
        "baseline_tree_sha256": BASELINE_TREE_SHA256,
    }
    for field, expected in checks.items():
        if config.get(field) != expected:
            raise ValueError(f"Frozen config mismatch for {field}: {config.get(field)!r} != {expected!r}")

    dataset = config.get("dataset", {})
    expected_dataset = {
        "name": "techqa",
        "source": "nvidia/TechQA-RAG-Eval",
        "split": "train",
        "revision": "0b5bbc84b7f07d6d09d063130e90b716d8d4a32a",
        "expected_eligible_rows": EXPECTED_ELIGIBLE_ROWS,
        "expected_documents": EXPECTED_DOCUMENTS,
    }
    if dataset != expected_dataset:
        raise ValueError("Dataset configuration differs from the frozen protocol")

    partition = config.get("partition", {})
    if partition != {
        "development_size": DEVELOPMENT_SIZE,
        "heldout_test_size": HELDOUT_SIZE,
        "reserve_size": RESERVE_SIZE,
    }:
        raise ValueError("Partition configuration differs from the frozen protocol")

    if config.get("chunkers") != EXPECTED_CHUNKERS:
        raise ValueError("Chunker configuration differs from the frozen protocol")
    if tuple(spec["name"] for spec in config["chunkers"]) != CHUNKERS:
        raise ValueError("Chunker order differs from the frozen protocol")

    retrieval = config.get("retrieval", {})
    expected_retrieval = {
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "embedding_model_revision": "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
        "chunking_tokenizer": "sentence-transformers/all-MiniLM-L6-v2",
        "chunking_tokenizer_revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "candidate_pool_size": 20,
        "final_top_k": 4,
        "dense_weight": 0.6,
        "bm25_weight": 0.4,
        "rrf_k": 60.0,
        "embedding_batch_size": 64,
    }
    if retrieval != expected_retrieval:
        raise ValueError("Retrieval configuration differs from the frozen protocol")

    reranker = config.get("reranker", {})
    if reranker != {
        "model": "cross-encoder/ms-marco-MiniLM-L6-v2",
        "revision": "c5ee24cb16019beea0893ab7796b1df96625c6b8",
        "batch_size": 32,
    }:
        raise ValueError("Reranker configuration differs from the frozen protocol")

    generators = config.get("generators", {})
    expected_generators = {
        "qwen": {
            "model": "Qwen/Qwen2.5-1.5B-Instruct",
            "revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
            "torch_dtype": None,
            "use_device_map": False,
            "role": "primary",
        },
        "mistral": {
            "model": "mistralai/Mistral-7B-Instruct-v0.3",
            "revision": "c170c708c41dac9275d15a8fff4eca08d52bab71",
            "torch_dtype": "float16",
            "use_device_map": True,
            "role": "secondary_replication",
        },
    }
    if generators != expected_generators:
        raise ValueError("Generator configuration differs from the frozen protocol")

    if config.get("generation") != {
        "answer_style": "complete",
        "max_input_tokens": 1536,
        "max_new_tokens": 512,
        "do_sample": False,
        "num_beams": 1,
    }:
        raise ValueError("Generation configuration differs from the frozen protocol")

    if config.get("gate") != {
        "threshold": 0.5,
        "C": 1.0,
        "penalty": "l2",
        "solver": "liblinear",
        "class_weight": "balanced",
        "max_iter": 1000,
        "random_state": 20260809,
    }:
        raise ValueError("Gate configuration differs from the frozen protocol")

    if config.get("statistics") != {
        "bootstrap_draws": 20000,
        "bootstrap_seed": 20260809,
        "randomization_draws": 100000,
        "randomization_seed": 20260810,
        "confidence": 0.95,
    }:
        raise ValueError("Statistical configuration differs from the frozen protocol")


def phase2_paths(repository: str | Path, config: dict[str, Any]) -> Phase2Paths:
    repo = Path(repository).resolve()
    results_root = (repo / "results" / "eaai_phase2").resolve()
    artifacts_root = (repo / "artifacts" / "eaai_phase2").resolve()
    return Phase2Paths(
        repository=repo,
        results_root=results_root,
        artifacts_root=artifacts_root,
        run_results=results_root / str(config["run_id"]),
        run_artifacts=artifacts_root / str(config["run_id"]),
    )
