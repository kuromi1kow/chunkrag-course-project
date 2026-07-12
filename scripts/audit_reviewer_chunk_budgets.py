#!/usr/bin/env python3
"""Verify post-decode chunk lengths against the actual embedding encoders."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from chunkrag.chunking import ChunkingContext, build_document_chunks
from chunkrag.pipeline import get_seed_values, load_dataset_bundle


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIGS = (
    REPO_ROOT / "configs" / "reviewer_robustness_retrieval_minilm.json",
    REPO_ROOT / "configs" / "reviewer_robustness_retrieval_bge.json",
    REPO_ROOT / "configs" / "reviewer_robustness_qwen.json",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def audit_config(config_path: Path) -> dict[str, Any]:
    config = load_json(config_path)
    chunk_tokenizer_name = config.get("chunking_tokenizer", config["embedding_model"])
    chunk_tokenizer_revision = config.get("chunking_tokenizer_revision")
    chunk_tokenizer = AutoTokenizer.from_pretrained(
        chunk_tokenizer_name,
        revision=chunk_tokenizer_revision,
    )
    encoder = SentenceTransformer(
        config["embedding_model"],
        revision=config.get("embedding_model_revision"),
        device="cpu",
    )
    encoder_tokenizer = encoder.tokenizer
    encoder_limit = int(encoder.max_seq_length)

    cells: list[dict[str, Any]] = []
    for seed in get_seed_values(config):
        for dataset_spec in config["datasets"]:
            documents, _ = load_dataset_bundle(dataset_spec, seed)
            for chunker_spec in config["chunkers"]:
                chunks = []
                context = ChunkingContext(tokenizer=chunk_tokenizer)
                for document in documents:
                    chunks.extend(build_document_chunks(document, chunker_spec, context))

                content_lengths = [chunk.token_count for chunk in chunks]
                encoder_lengths = [
                    len(encoder_tokenizer.encode(chunk.text, add_special_tokens=True))
                    for chunk in chunks
                ]
                target = int(chunker_spec["chunk_size"])
                content_over_target = sum(length > target for length in content_lengths)
                encoder_over_limit = sum(length > encoder_limit for length in encoder_lengths)
                cell = {
                    "config": str(config_path.relative_to(REPO_ROOT)),
                    "seed": seed,
                    "dataset": dataset_spec["name"],
                    "chunker": chunker_spec["name"],
                    "num_documents": len(documents),
                    "num_chunks": len(chunks),
                    "target_content_tokens": target,
                    "max_content_tokens": max(content_lengths, default=0),
                    "content_chunks_over_target": content_over_target,
                    "encoder_max_positions": encoder_limit,
                    "max_encoder_positions": max(encoder_lengths, default=0),
                    "chunks_over_encoder_limit": encoder_over_limit,
                }
                if chunker_spec.get("enforce_token_limit") and content_over_target:
                    raise ValueError(f"Strict chunk target violated: {cell}")
                if encoder_over_limit:
                    raise ValueError(f"Embedding encoder limit violated: {cell}")
                cells.append(cell)

    return {
        "config": str(config_path.relative_to(REPO_ROOT)),
        "embedding_model": config["embedding_model"],
        "embedding_model_revision": config.get("embedding_model_revision"),
        "chunking_tokenizer": chunk_tokenizer_name,
        "chunking_tokenizer_revision": chunk_tokenizer_revision,
        "encoder_max_positions": encoder_limit,
        "cells": cells,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "outputs" / "reviewer_robustness_chunk_budget_audit.json",
    )
    parser.add_argument("configs", nargs="*", type=Path, default=list(DEFAULT_CONFIGS))
    args = parser.parse_args()

    result = {
        "audit": "post-decode content tokens and special-token-inclusive encoder positions",
        "configs": [audit_config(path.resolve()) for path in args.configs],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
