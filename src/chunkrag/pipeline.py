from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

from sentence_transformers import SentenceTransformer
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
from chunkrag.evaluation import Timer, answer_metrics, retrieval_metrics, summarize_metric
from chunkrag.generation import QAGenerator, resolve_device
from chunkrag.retrieval import BM25Retriever, DenseRetriever, HybridRetriever, RerankRetriever
from chunkrag.schemas import Chunk, Document, QAExample


def load_experiment_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def get_seed_values(config: dict) -> list[int]:
    if "seeds" in config:
        return [int(seed) for seed in config["seeds"]]
    return [int(config.get("seed", 42))]


def load_dataset_bundle(spec: dict, seed: int) -> tuple[list[Document], list[QAExample]]:
    name = spec["name"]
    if name == "squad_v2":
        return load_squad_documents_and_examples(
            split=spec.get("split", "validation"),
            max_examples=spec["max_examples"],
            candidate_pool_size=spec.get("candidate_pool_size", spec["max_examples"] * 5),
            seed=seed,
            answerable_only=spec.get("answerable_only", True),
        )
    if name == "hotpot_qa":
        return load_hotpot_documents_and_examples(
            split=spec.get("split", "validation"),
            max_examples=spec["max_examples"],
            config_name=spec.get("config", "distractor"),
            seed=seed,
        )
    raise ValueError(f"Unsupported dataset: {name}")


def build_chunks(
    documents: list[Document],
    chunker_spec: dict,
    tokenizer,
    semantic_encoder: SentenceTransformer,
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
                embedding_model=semantic_encoder,
                chunker_name=chunker_name,
                min_chunk_tokens=chunker_spec.get("min_chunk_tokens"),
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


def _get_retriever_specs(config: dict) -> tuple[list[dict], bool]:
    if "retrievers" in config:
        return config["retrievers"], False
    if "retriever" in config:
        return [config["retriever"]], False
    return [{"name": "dense", "type": "dense"}], True


def _get_system_name(chunker_name: str, retriever_spec: dict, legacy_names: bool) -> str:
    if legacy_names and retriever_spec.get("type", "dense") == "dense":
        return chunker_name
    retriever_name = retriever_spec.get("name", retriever_spec.get("type", "dense"))
    return f"{retriever_name}__{chunker_name}"


def _summarize_prediction_rows(
    rows: list[dict],
    *,
    dataset_name: str,
    system_name: str,
    seed: int,
    retriever_name: str,
    chunker_name: str | None,
    num_documents: int | None = None,
    num_chunks: int | None = None,
    avg_chunk_tokens: float | None = None,
    retrieval_times: list[float] | None = None,
    generation_times: list[float] | None = None,
    bootstrap_samples: int = 1_000,
    confidence: float = 0.95,
) -> dict:
    summary = {
        "dataset": dataset_name,
        "system": system_name,
        "seed": seed,
        "retriever": retriever_name,
        "chunker": chunker_name,
        "num_examples": len(rows),
    }
    if num_documents is not None:
        summary["num_documents"] = num_documents
    if num_chunks is not None:
        summary["num_chunks"] = num_chunks
    if avg_chunk_tokens is not None:
        summary["avg_chunk_tokens"] = avg_chunk_tokens
    if retrieval_times is not None:
        summary["avg_retrieval_latency_s"] = mean(retrieval_times) if retrieval_times else 0.0
    if generation_times is not None:
        summary["avg_generation_latency_s"] = mean(generation_times) if generation_times else 0.0

    for metric_name in (
        "exact_match",
        "f1",
        "recall_at_k",
        "precision_at_k",
        "supporting_doc_coverage",
        "all_supporting_docs_found",
    ):
        if rows and metric_name in rows[0]:
            summary.update(
                summarize_metric(
                    metric_name,
                    [float(row[metric_name]) for row in rows],
                    bootstrap_samples=bootstrap_samples,
                    confidence=confidence,
                    seed=seed,
                )
            )
    return summary


def _aggregate_seed_summaries(summaries: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for summary in summaries:
        key = (
            summary["dataset"],
            summary["system"],
            summary.get("retriever"),
            summary.get("chunker"),
        )
        grouped[key].append(summary)

    aggregates: list[dict] = []
    for (dataset_name, system_name, retriever_name, chunker_name), rows in grouped.items():
        aggregate = {
            "dataset": dataset_name,
            "system": system_name,
            "retriever": retriever_name,
            "chunker": chunker_name,
            "num_seeds": len(rows),
            "seed_values": sorted(int(row["seed"]) for row in rows),
        }
        numeric_fields = sorted(
            {
                field
                for row in rows
                for field, value in row.items()
                if isinstance(value, (int, float))
                and field != "seed"
                and not field.endswith("_ci_low")
                and not field.endswith("_ci_high")
            }
        )
        for field in numeric_fields:
            values = [float(row[field]) for row in rows if field in row]
            aggregate[f"{field}_mean"] = mean(values)
            aggregate[f"{field}_std"] = stdev(values) if len(values) > 1 else 0.0
            aggregate[f"{field}_min"] = min(values)
            aggregate[f"{field}_max"] = max(values)
        aggregates.append(aggregate)
    return sorted(aggregates, key=lambda row: (row["dataset"], row["system"]))


def run_dataset_experiments(config: dict, dataset_spec: dict, output_dir: Path, seed: int) -> list[dict]:
    dataset_name = dataset_spec["name"]
    documents, examples = load_dataset_bundle(dataset_spec, seed=seed)
    dataset_dir = output_dir / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(config.get("device", "auto"))
    generator = QAGenerator(
        model_name=config["generator_model"],
        device=config.get("device", "auto"),
        max_input_tokens=config.get("generation_max_input_tokens", 768),
        max_new_tokens=config.get("max_new_tokens", 32),
        torch_dtype=config.get("generator_torch_dtype"),
        use_device_map=config.get("generator_use_device_map", False),
    )
    retrieval_tokenizer = AutoTokenizer.from_pretrained(config["embedding_model"])
    retrieval_tokenizer.model_max_length = 1_000_000
    semantic_encoder = SentenceTransformer(config["embedding_model"], device=device)
    retriever_specs, legacy_system_names = _get_retriever_specs(config)
    bootstrap_samples = int(config.get("bootstrap_samples", 1_000))
    confidence = float(config.get("confidence_level", 0.95))

    summaries: list[dict] = []
    if config.get("run_parametric_baseline", True):
        baseline_predictions: list[dict] = []
        for example in tqdm(examples, desc=f"baseline::{dataset_name}::seed_{seed}"):
            prediction = generator.answer(example.question, context=None)
            baseline_predictions.append(
                {
                    "seed": seed,
                    "example_id": example.example_id,
                    "question": example.question,
                    "gold_answers": example.answers,
                    "prediction": prediction,
                    **answer_metrics(prediction, example.answers),
                }
            )
        baseline_summary = _summarize_prediction_rows(
            baseline_predictions,
            dataset_name=dataset_name,
            system_name="parametric_only",
            seed=seed,
            retriever_name="parametric_only",
            chunker_name=None,
            bootstrap_samples=bootstrap_samples,
            confidence=confidence,
        )
        _save_json(dataset_dir / "parametric_only_predictions.json", baseline_predictions)
        _save_json(dataset_dir / "parametric_only_summary.json", baseline_summary)
        summaries.append(baseline_summary)

    for chunker_spec in config["chunkers"]:
        chunker_name = chunker_spec["name"]
        chunks = build_chunks(documents, chunker_spec, retrieval_tokenizer, semantic_encoder)
        dense_retriever: DenseRetriever | None = None
        sparse_retriever: BM25Retriever | None = None

        def get_dense_retriever() -> DenseRetriever:
            nonlocal dense_retriever
            if dense_retriever is None:
                dense_retriever = DenseRetriever(
                    encoder=semantic_encoder,
                    device=device,
                    batch_size=config.get("embedding_batch_size", 32),
                )
                dense_retriever.build(chunks)
            return dense_retriever

        def get_sparse_retriever() -> BM25Retriever:
            nonlocal sparse_retriever
            if sparse_retriever is None:
                sparse_retriever = BM25Retriever()
                sparse_retriever.build(chunks)
            return sparse_retriever

        def materialize_retriever(spec: dict):
            retriever_type = spec.get("type", "dense")
            if retriever_type == "dense":
                return get_dense_retriever()
            if retriever_type == "bm25":
                return get_sparse_retriever()
            if retriever_type == "hybrid":
                return HybridRetriever(
                    dense_retriever=get_dense_retriever(),
                    sparse_retriever=get_sparse_retriever(),
                    candidate_pool_size=spec.get("candidate_pool_size", max(config.get("retrieval_top_k", 4) * 5, 20)),
                    dense_weight=spec.get("dense_weight", 0.5),
                    sparse_weight=spec.get("sparse_weight", 0.5),
                    rrf_k=spec.get("rrf_k", 60.0),
                )
            if retriever_type == "rerank":
                base_retriever_spec = dict(spec.get("base_retriever", {"type": "dense"}))
                base_retriever = materialize_retriever(base_retriever_spec)
                return RerankRetriever(
                    base_retriever=base_retriever,
                    model_name=spec.get("model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
                    device=device,
                    candidate_pool_size=spec.get("candidate_pool_size", max(config.get("retrieval_top_k", 4) * 5, 20)),
                    batch_size=spec.get("batch_size", 16),
                )
            raise ValueError(f"Unsupported retriever type: {retriever_type}")

        for retriever_spec in retriever_specs:
            system_name = _get_system_name(chunker_name, retriever_spec, legacy_system_names)
            retriever_name = retriever_spec.get("name", retriever_spec.get("type", "dense"))
            retriever = materialize_retriever(retriever_spec)

            predictions: list[dict] = []
            retrieval_times: list[float] = []
            generation_times: list[float] = []

            for example in tqdm(examples, desc=f"eval::{dataset_name}::{system_name}::seed_{seed}"):
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
                        "seed": seed,
                        "retriever": retriever_name,
                        "chunker": chunker_name,
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

            summary = _summarize_prediction_rows(
                predictions,
                dataset_name=dataset_name,
                system_name=system_name,
                seed=seed,
                retriever_name=retriever_name,
                chunker_name=chunker_name,
                num_documents=len(documents),
                num_chunks=len(chunks),
                avg_chunk_tokens=mean(chunk.token_count for chunk in chunks) if chunks else 0.0,
                retrieval_times=retrieval_times,
                generation_times=generation_times,
                bootstrap_samples=bootstrap_samples,
                confidence=confidence,
            )
            _save_json(dataset_dir / f"{system_name}_predictions.json", predictions)
            _save_json(dataset_dir / f"{system_name}_summary.json", summary)
            summaries.append(summary)

    _save_json(dataset_dir / "all_summaries.json", summaries)
    return summaries


def run_experiment_suite(config: dict, output_dir: Path) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = get_seed_values(config)
    use_seed_subdirs = len(seeds) > 1

    all_summaries: list[dict] = []
    for seed in seeds:
        run_output_dir = output_dir / f"seed_{seed}" if use_seed_subdirs else output_dir
        for dataset_spec in config["datasets"]:
            all_summaries.extend(run_dataset_experiments(config, dataset_spec, run_output_dir, seed=seed))

    _save_json(output_dir / "experiment_config.json", config)
    _save_json(output_dir / "all_results.json", all_summaries)
    _save_json(output_dir / "aggregate_results.json", _aggregate_seed_summaries(all_summaries))
    return all_summaries
