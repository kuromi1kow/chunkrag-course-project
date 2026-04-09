from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

from tqdm import tqdm
from transformers import AutoTokenizer

from chunkrag.chunking import (
    chonkie_recursive_chunks,
    chonkie_semantic_chunks,
    fixed_token_chunks,
    recursive_chunks,
    semantic_chunks,
    sentence_chunks,
)
from chunkrag.data import load_hotpot_documents_and_examples, load_squad_documents_and_examples
from chunkrag.evaluation import Timer, answer_metrics, retrieval_metrics
from chunkrag.generation import QAGenerator, resolve_device
from chunkrag.retrieval import DenseRetriever
from chunkrag.schemas import Chunk, Document, QAExample


def load_experiment_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_dataset_bundle(spec: dict) -> tuple[list[Document], list[QAExample]]:
    name = spec["name"]
    if name == "squad_v2":
        return load_squad_documents_and_examples(
            split=spec.get("split", "validation"),
            max_examples=spec["max_examples"],
            candidate_pool_size=spec.get("candidate_pool_size", spec["max_examples"] * 5),
            answerable_only=spec.get("answerable_only", True),
        )
    if name == "hotpot_qa":
        return load_hotpot_documents_and_examples(
            split=spec.get("split", "validation"),
            max_examples=spec["max_examples"],
            config_name=spec.get("config", "distractor"),
        )
    raise ValueError(f"Unsupported dataset: {name}")


def build_chunks(
    documents: list[Document],
    chunker_spec: dict,
    tokenizer,
    retriever: DenseRetriever,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    chunker_type = chunker_spec["type"]
    chunker_name = chunker_spec["name"]
    for document in tqdm(documents, desc=f"chunk::{chunker_name}"):
        if chunker_type == "fixed":
            doc_chunks = fixed_token_chunks(
                document,
                tokenizer,
                chunk_size=chunker_spec["chunk_size"],
                chunk_overlap=chunker_spec.get("chunk_overlap", 0),
                chunker_name=chunker_name,
            )
        elif chunker_type == "recursive":
            doc_chunks = recursive_chunks(
                document,
                tokenizer,
                chunk_size=chunker_spec["chunk_size"],
                chunk_overlap=chunker_spec.get("chunk_overlap", 0),
                chunker_name=chunker_name,
            )
        elif chunker_type == "sentence":
            doc_chunks = sentence_chunks(
                document,
                tokenizer,
                chunk_size=chunker_spec["chunk_size"],
                chunker_name=chunker_name,
            )
        elif chunker_type == "semantic":
            doc_chunks = semantic_chunks(
                document,
                tokenizer,
                chunk_size=chunker_spec["chunk_size"],
                similarity_threshold=chunker_spec.get("similarity_threshold", 0.72),
                embedding_model=retriever.encoder,
                chunker_name=chunker_name,
            )
        elif chunker_type == "chonkie_recursive":
            doc_chunks = chonkie_recursive_chunks(
                document,
                tokenizer,
                chunk_size=chunker_spec["chunk_size"],
                chunker_name=chunker_name,
            )
        elif chunker_type == "chonkie_semantic":
            doc_chunks = chonkie_semantic_chunks(
                document,
                chunk_size=chunker_spec["chunk_size"],
                chunker_name=chunker_name,
                embedding_model_name=chunker_spec.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
                threshold=chunker_spec.get("similarity_threshold", 0.7),
                min_sentences_per_chunk=chunker_spec.get("min_sentences_per_chunk", 1),
                similarity_window=chunker_spec.get("similarity_window", 3),
                skip_window=chunker_spec.get("skip_window", 0),
            )
        else:
            raise ValueError(f"Unsupported chunker type: {chunker_type}")
        chunks.extend(doc_chunks)
    return chunks


def _save_json(path: Path, payload: dict | list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def run_dataset_experiments(config: dict, dataset_spec: dict, output_dir: Path) -> list[dict]:
    dataset_name = dataset_spec["name"]
    documents, examples = load_dataset_bundle(dataset_spec)
    dataset_dir = output_dir / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    generator = QAGenerator(
        model_name=config["generator_model"],
        device=config.get("device", "auto"),
        max_input_tokens=config.get("generation_max_input_tokens", 768),
        max_new_tokens=config.get("max_new_tokens", 32),
    )
    retrieval_tokenizer = AutoTokenizer.from_pretrained(config["embedding_model"])
    retrieval_tokenizer.model_max_length = 1_000_000
    retriever = DenseRetriever(
        model_name=config["embedding_model"],
        device=resolve_device(config.get("device", "auto")),
        batch_size=config.get("embedding_batch_size", 32),
    )

    summaries: list[dict] = []
    if config.get("run_parametric_baseline", True):
        baseline_predictions: list[dict] = []
        for example in tqdm(examples, desc=f"baseline::{dataset_name}"):
            prediction = generator.answer(example.question, context=None)
            baseline_predictions.append(
                {
                    "example_id": example.example_id,
                    "question": example.question,
                    "gold_answers": example.answers,
                    "prediction": prediction,
                    **answer_metrics(prediction, example.answers),
                }
            )
        baseline_summary = {
            "dataset": dataset_name,
            "system": "parametric_only",
            "num_examples": len(baseline_predictions),
            "exact_match": mean(row["exact_match"] for row in baseline_predictions),
            "f1": mean(row["f1"] for row in baseline_predictions),
        }
        _save_json(dataset_dir / "parametric_only_predictions.json", baseline_predictions)
        _save_json(dataset_dir / "parametric_only_summary.json", baseline_summary)
        summaries.append(baseline_summary)

    for chunker_spec in config["chunkers"]:
        chunker_name = chunker_spec["name"]
        chunks = build_chunks(documents, chunker_spec, retrieval_tokenizer, retriever)
        retriever.build(chunks)

        predictions: list[dict] = []
        retrieval_times: list[float] = []
        generation_times: list[float] = []

        for example in tqdm(examples, desc=f"eval::{dataset_name}::{chunker_name}"):
            with Timer() as retrieval_timer:
                retrieved = retriever.retrieve(example.question, config.get("retrieval_top_k", 4))
            retrieved_chunks = [chunk for chunk, _ in retrieved]
            context = "\n\n".join(
                f"[{index + 1}] {chunk.text}" for index, chunk in enumerate(retrieved_chunks)
            )
            with Timer() as generation_timer:
                prediction = generator.answer(example.question, context=context)

            retrieval_times.append(retrieval_timer.elapsed)
            generation_times.append(generation_timer.elapsed)
            predictions.append(
                {
                    "example_id": example.example_id,
                    "question": example.question,
                    "gold_answers": example.answers,
                    "prediction": prediction,
                    "retrieved_chunk_ids": [chunk.chunk_id for chunk in retrieved_chunks],
                    "retrieved_doc_ids": [chunk.doc_id for chunk in retrieved_chunks],
                    "retrieved_titles": [chunk.title for chunk in retrieved_chunks],
                    **answer_metrics(prediction, example.answers),
                    **retrieval_metrics(retrieved_chunks, example),
                }
            )

        summary = {
            "dataset": dataset_name,
            "system": chunker_name,
            "num_documents": len(documents),
            "num_examples": len(examples),
            "num_chunks": len(chunks),
            "avg_chunk_tokens": mean(chunk.token_count for chunk in chunks),
            "exact_match": mean(row["exact_match"] for row in predictions),
            "f1": mean(row["f1"] for row in predictions),
            "recall_at_k": mean(row["recall_at_k"] for row in predictions),
            "precision_at_k": mean(row["precision_at_k"] for row in predictions),
            "avg_retrieval_latency_s": mean(retrieval_times),
            "avg_generation_latency_s": mean(generation_times),
        }
        _save_json(dataset_dir / f"{chunker_name}_predictions.json", predictions)
        _save_json(dataset_dir / f"{chunker_name}_summary.json", summary)
        summaries.append(summary)

    _save_json(dataset_dir / "all_summaries.json", summaries)
    return summaries
