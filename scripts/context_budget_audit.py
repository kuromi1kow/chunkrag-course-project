#!/usr/bin/env python3
"""Replay the archived top-4 contexts through the reported 1,024-token budget.

This script reconstructs the exact sampled corpora and chunks, joins them to the
ranked chunk IDs in the Mistral-v2 prediction artifacts, applies the repository's
chat-template-aware prefix-truncation algorithm, and records what the generator
could actually consume. No answers are regenerated.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from chunkrag.chunking import ChunkingContext, build_document_chunks
from chunkrag.data import load_hotpot_documents_and_examples, load_squad_documents_and_examples
from chunkrag.generation import build_openai_qa_messages
from chunkrag.text_utils import contains_normalized_answer


DATASETS = ("squad_v2", "hotpot_qa")
MAX_INPUT_TOKENS = 1_024


def chat_token_count(tokenizer, messages: list[dict[str, str]]) -> int:
    token_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    return len(token_ids)


def format_context(dataset: str, chunks: list) -> str:
    parts = []
    include_titles = dataset == "hotpot_qa"
    for index, chunk in enumerate(chunks, start=1):
        if include_titles:
            title = chunk.title.strip() if chunk.title else chunk.doc_id
            parts.append(f"[{index}] Title: {title}\nPassage: {chunk.text}")
        else:
            parts.append(f"[{index}] {chunk.text}")
    return "\n\n".join(parts)


def truncate_context(tokenizer, question: str, context: str) -> tuple[str, int, int, bool]:
    full_messages = build_openai_qa_messages(question, context)
    full_prompt_tokens = chat_token_count(tokenizer, full_messages)
    if full_prompt_tokens <= MAX_INPUT_TOKENS:
        context_tokens = len(tokenizer.encode(context, add_special_tokens=False))
        return context, full_prompt_tokens, context_tokens, False

    context_ids = tokenizer.encode(context, add_special_tokens=False)
    lo, hi = 0, len(context_ids)
    best_context = ""
    best_context_tokens = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate_context = tokenizer.decode(context_ids[:mid], skip_special_tokens=True).strip()
        candidate_messages = build_openai_qa_messages(question, candidate_context)
        if chat_token_count(tokenizer, candidate_messages) <= MAX_INPUT_TOKENS:
            best_context = candidate_context
            best_context_tokens = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best_context, full_prompt_tokens, best_context_tokens, True


def fully_consumed_chunks(tokenizer, dataset: str, chunks: list, kept_context_tokens: int) -> int:
    full = 0
    for count in range(1, len(chunks) + 1):
        prefix = format_context(dataset, chunks[:count])
        prefix_tokens = len(tokenizer.encode(prefix, add_special_tokens=False))
        if prefix_tokens <= kept_context_tokens:
            full = count
        else:
            break
    return full


def exceeds_embedding_limit(tokenizer, text: str, max_seq_length: int) -> bool:
    """Match SentenceTransformer's special-token-inclusive input length check."""
    return len(tokenizer.encode(text, add_special_tokens=True)) > max_seq_length


def load_bundle(dataset: str):
    if dataset == "squad_v2":
        return load_squad_documents_and_examples(
            split="validation",
            max_examples=60,
            candidate_pool_size=500,
            seed=42,
            answerable_only=True,
        )
    return load_hotpot_documents_and_examples(
        split="validation",
        max_examples=30,
        config_name="distractor",
        seed=42,
    )


def resolve_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def summarize(rows: list[dict]) -> dict:
    distribution = Counter(int(row["fully_consumed_chunks"]) for row in rows)
    return {
        "n": len(rows),
        "truncation_rate": mean(float(row["truncated"]) for row in rows),
        "mean_full_prompt_tokens": mean(row["full_prompt_tokens"] for row in rows),
        "mean_context_token_retention": mean(row["context_token_retention"] for row in rows),
        "mean_fully_consumed_chunks": mean(row["fully_consumed_chunks"] for row in rows),
        "all_four_chunks_fully_consumed_rate": mean(
            float(row["fully_consumed_chunks"] == 4) for row in rows
        ),
        "fully_consumed_chunk_distribution": {
            str(count): distribution[count] for count in range(5)
        },
        "mean_pretruncation_support_coverage": mean(
            row["pretruncation_support_coverage"] for row in rows
        ),
        "mean_fullchunk_support_coverage_after_truncation": mean(
            row["fullchunk_support_coverage_after_truncation"] for row in rows
        ),
        "all_support_docs_in_fully_consumed_chunks_rate": mean(
            float(row["all_support_docs_in_fully_consumed_chunks"]) for row in rows
        ),
        "gold_answer_string_visible_rate": mean(
            float(row["gold_answer_string_visible"]) for row in rows
        ),
        "questions_with_embedding_truncated_retrieved_chunk_rate": mean(
            float(row["retrieved_chunks_over_embedding_limit"] > 0) for row in rows
        ),
        "mean_retrieved_chunks_over_embedding_limit": mean(
            row["retrieved_chunks_over_embedding_limit"] for row in rows
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_config", type=Path)
    parser.add_argument("prediction_root", type=Path)
    parser.add_argument("output_json", type=Path)
    args = parser.parse_args()

    config = json.loads(args.experiment_config.read_text(encoding="utf-8"))
    retrieval_tokenizer = AutoTokenizer.from_pretrained(config["embedding_model"])
    retrieval_tokenizer.model_max_length = 1_000_000
    generator_tokenizer = AutoTokenizer.from_pretrained(config["generator_tokenizer_name"])
    semantic_encoder = SentenceTransformer(config["embedding_model"], device=resolve_device())
    embedding_max_seq_length = int(semantic_encoder.max_seq_length)
    chunking_context = ChunkingContext(
        tokenizer=retrieval_tokenizer,
        semantic_encoder=semantic_encoder,
    )

    result: dict[str, object] = {
        "max_input_tokens": MAX_INPUT_TOKENS,
        "retrieval_tokenizer": config["embedding_model"],
        "generator_tokenizer": config["generator_tokenizer_name"],
        "embedding_max_seq_length": embedding_max_seq_length,
        "embedding_length_includes_special_tokens": True,
        "truncation_policy": "keep the longest prefix whose complete chat prompt is at most 1,024 tokens",
        "datasets": {},
    }

    for dataset in DATASETS:
        documents, examples = load_bundle(dataset)
        example_map = {example.example_id: example for example in examples}
        dataset_result: dict[str, object] = {}
        for chunker_spec in config["chunkers"]:
            system = chunker_spec["name"]
            chunks = []
            for document in documents:
                chunks.extend(build_document_chunks(document, chunker_spec, chunking_context))
            chunk_map = {chunk.chunk_id: chunk for chunk in chunks}
            embedding_truncated_chunk_ids = {
                chunk.chunk_id
                for chunk in chunks
                if exceeds_embedding_limit(
                    retrieval_tokenizer,
                    chunk.text,
                    embedding_max_seq_length,
                )
            }

            prediction_path = args.prediction_root / dataset / f"{system}_predictions.json"
            predictions = json.loads(prediction_path.read_text(encoding="utf-8"))
            audit_rows = []
            for prediction in predictions:
                example = example_map[prediction["example_id"]]
                retrieved = [chunk_map[chunk_id] for chunk_id in prediction["retrieved_chunk_ids"]]
                context = format_context(dataset, retrieved)
                full_context_tokens = len(generator_tokenizer.encode(context, add_special_tokens=False))
                truncated_context, full_prompt_tokens, kept_context_tokens, was_truncated = truncate_context(
                    generator_tokenizer,
                    example.question,
                    context,
                )
                full_chunk_count = fully_consumed_chunks(
                    generator_tokenizer,
                    dataset,
                    retrieved,
                    kept_context_tokens,
                )
                fully_visible_doc_ids = {chunk.doc_id for chunk in retrieved[:full_chunk_count]}
                support_ids = set(example.relevant_doc_ids)
                fullchunk_support_hits = len(fully_visible_doc_ids & support_ids)
                fullchunk_support_coverage = (
                    fullchunk_support_hits / len(support_ids) if support_ids else 0.0
                )
                audit_rows.append(
                    {
                        "example_id": example.example_id,
                        "question": example.question,
                        "exact_match": prediction["exact_match"],
                        "f1": prediction["f1"],
                        "stored_prediction": prediction["prediction"],
                        "retrieved_chunk_ids": prediction["retrieved_chunk_ids"],
                        "retrieved_doc_ids": prediction["retrieved_doc_ids"],
                        "retrieved_chunks_over_embedding_limit": sum(
                            int(chunk.chunk_id in embedding_truncated_chunk_ids)
                            for chunk in retrieved
                        ),
                        "full_prompt_tokens": full_prompt_tokens,
                        "full_context_tokens": full_context_tokens,
                        "kept_context_tokens": kept_context_tokens,
                        "context_token_retention": (
                            kept_context_tokens / full_context_tokens if full_context_tokens else 1.0
                        ),
                        "truncated": was_truncated,
                        "fully_consumed_chunks": full_chunk_count,
                        "pretruncation_support_coverage": prediction["supporting_doc_coverage"],
                        "fullchunk_support_coverage_after_truncation": fullchunk_support_coverage,
                        "all_support_docs_in_fully_consumed_chunks": (
                            fullchunk_support_hits == len(support_ids) if support_ids else False
                        ),
                        "gold_answer_string_visible": contains_normalized_answer(
                            truncated_context,
                            example.answers,
                        ),
                    }
                )

            refusal_rows = [
                row
                for row in audit_rows
                if row["exact_match"] == 0.0
                and str(row["stored_prediction"]).strip().lower() == "unanswerable"
            ]
            dataset_result[system] = {
                "num_documents": len(documents),
                "num_chunks_reconstructed": len(chunks),
                "corpus_chunks_over_embedding_limit": len(embedding_truncated_chunk_ids),
                "corpus_chunk_over_embedding_limit_rate": (
                    len(embedding_truncated_chunk_ids) / len(chunks) if chunks else 0.0
                ),
                "summary": summarize(audit_rows),
                "refusal_diagnostic": {
                    "n": len(refusal_rows),
                    "all_support_docs_in_fully_consumed_chunks": sum(
                        int(row["all_support_docs_in_fully_consumed_chunks"])
                        for row in refusal_rows
                    ),
                    "gold_answer_string_visible": sum(
                        int(row["gold_answer_string_visible"]) for row in refusal_rows
                    ),
                },
                "per_question": audit_rows,
            }
        result["datasets"][dataset] = dataset_result

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
