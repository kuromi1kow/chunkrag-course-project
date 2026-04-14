from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from chunkrag.chunking import ChunkingContext, build_document_chunks
from chunkrag.data import load_hotpot_documents_and_examples, load_squad_documents_and_examples
from chunkrag.evaluation import (
    Timer,
    answer_metrics,
    bootstrap_confidence_interval,
    retrieval_metrics,
)
from chunkrag.generation import Generator, OpenAICompatibleGenerator, QAGenerator, resolve_device
from chunkrag.retrieval import Retriever, RetrieverFactory, RetrieverFactoryContext
from chunkrag.schemas import (
    AggregateMetricSummary,
    AggregateSummaryRow,
    Chunk,
    Document,
    MetricSummary,
    PredictionRecord,
    QAExample,
    SummaryRow,
)


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def get_seed_values(config: dict[str, Any]) -> list[int]:
    if "seeds" in config:
        return [int(seed) for seed in config["seeds"]]
    return [int(config.get("seed", 42))]


def load_dataset_bundle(spec: dict[str, Any], seed: int) -> tuple[list[Document], list[QAExample]]:
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
    chunker_spec: dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    semantic_encoder: SentenceTransformer | None,
) -> list[Chunk]:
    context = ChunkingContext(tokenizer=tokenizer, semantic_encoder=semantic_encoder)
    chunks: list[Chunk] = []
    for document in tqdm(documents, desc=f"chunk::{chunker_spec['name']}"):
        chunks.extend(build_document_chunks(document, chunker_spec, context))
    return chunks


class ArtifactWriter:
    def write_json(self, path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self._serialize(payload), handle, indent=2, ensure_ascii=False)

    def _serialize(self, payload: Any) -> Any:
        if isinstance(payload, SummaryRow):
            return payload.to_flat_dict()
        if isinstance(payload, AggregateSummaryRow):
            return payload.to_flat_dict()
        if isinstance(payload, PredictionRecord):
            return payload.to_dict()
        if isinstance(payload, list):
            return [self._serialize(item) for item in payload]
        if isinstance(payload, dict):
            return {key: self._serialize(value) for key, value in payload.items()}
        return payload


@dataclass(slots=True)
class SharedExperimentResources:
    device: str
    generator: Generator
    retrieval_tokenizer: PreTrainedTokenizerBase
    semantic_encoder: SentenceTransformer
    embedding_model: str
    retrieval_cache_dir: Path

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> SharedExperimentResources:
        device = resolve_device(config.get("device", "auto"))
        if config.get("generator_base_url"):
            generator = OpenAICompatibleGenerator(
                model_name=config["generator_model"],
                base_url=config["generator_base_url"],
                api_key=config.get("generator_api_key", "chunkrag-demo-key"),
                tokenizer_name=config.get("generator_tokenizer_name"),
                max_input_tokens=config.get("generation_max_input_tokens", 768),
                max_new_tokens=config.get("max_new_tokens", 32),
                temperature=float(config.get("generator_temperature", 0.0)),
            )
        else:
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
        retrieval_cache_dir = Path(config.get("retrieval_cache_dir", ".cache/chunkrag/retrieval"))
        return cls(
            device=device,
            generator=generator,
            retrieval_tokenizer=retrieval_tokenizer,
            semantic_encoder=semantic_encoder,
            embedding_model=config["embedding_model"],
            retrieval_cache_dir=retrieval_cache_dir,
        )


@dataclass(slots=True)
class SystemRunOutput:
    predictions: list[PredictionRecord]
    retrieval_times: list[float]
    generation_times: list[float]


@dataclass(slots=True)
class SystemRunner:
    dataset_name: str
    system_name: str
    seed: int
    retriever_name: str
    chunker_name: str | None
    retriever: Retriever
    generator: Generator
    examples: list[QAExample]
    retrieval_top_k: int

    def run(self) -> SystemRunOutput:
        predictions: list[PredictionRecord] = []
        retrieval_times: list[float] = []
        generation_times: list[float] = []

        for example in tqdm(self.examples, desc=f"eval::{self.dataset_name}::{self.system_name}::seed_{self.seed}"):
            with Timer() as retrieval_timer:
                retrieved = self.retriever.retrieve(example.question, self.retrieval_top_k)
            retrieved_chunks = [chunk for chunk, _ in retrieved]
            context = "\n\n".join(f"[{index + 1}] {chunk.text}" for index, chunk in enumerate(retrieved_chunks))
            with Timer() as generation_timer:
                prediction = self.generator.answer(example.question, context=context)

            retrieval_times.append(retrieval_timer.elapsed)
            generation_times.append(generation_timer.elapsed)
            answer_scores = answer_metrics(prediction, example.answers)
            retrieval_scores = retrieval_metrics(retrieved_chunks, example)
            predictions.append(
                PredictionRecord(
                    seed=self.seed,
                    retriever=self.retriever_name,
                    chunker=self.chunker_name,
                    example_id=example.example_id,
                    question=example.question,
                    gold_answers=example.answers,
                    prediction=prediction,
                    retrieved_chunk_ids=[chunk.chunk_id for chunk in retrieved_chunks],
                    retrieved_doc_ids=[chunk.doc_id for chunk in retrieved_chunks],
                    retrieved_titles=[chunk.title for chunk in retrieved_chunks],
                    exact_match=answer_scores["exact_match"],
                    f1=answer_scores["f1"],
                    recall_at_k=retrieval_scores["recall_at_k"],
                    precision_at_k=retrieval_scores["precision_at_k"],
                    supporting_doc_coverage=retrieval_scores["supporting_doc_coverage"],
                    all_supporting_docs_found=retrieval_scores["all_supporting_docs_found"],
                )
            )

        return SystemRunOutput(
            predictions=predictions,
            retrieval_times=retrieval_times,
            generation_times=generation_times,
        )


def _metric_summary(
    values: list[float],
    *,
    bootstrap_samples: int,
    confidence: float,
    seed: int,
) -> MetricSummary:
    metric_mean = mean(values) if values else 0.0
    ci_low, ci_high = bootstrap_confidence_interval(
        values,
        num_bootstrap_samples=bootstrap_samples,
        confidence=confidence,
        seed=seed,
    )
    return MetricSummary(value=metric_mean, ci_low=ci_low, ci_high=ci_high)


def _get_retriever_specs(config: dict[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    if "retrievers" in config:
        return config["retrievers"], False
    if "retriever" in config:
        return [config["retriever"]], False
    return [{"name": "dense", "type": "dense"}], True


def _get_system_name(chunker_name: str, retriever_spec: dict[str, Any], legacy_names: bool) -> str:
    if legacy_names and retriever_spec.get("type", "dense") == "dense":
        return chunker_name
    retriever_name = retriever_spec.get("name", retriever_spec.get("type", "dense"))
    return f"{retriever_name}__{chunker_name}"


def _summarize_prediction_rows(
    rows: list[PredictionRecord],
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
) -> SummaryRow:
    metrics: dict[str, MetricSummary] = {}
    for metric_name in (
        "exact_match",
        "f1",
        "recall_at_k",
        "precision_at_k",
        "supporting_doc_coverage",
        "all_supporting_docs_found",
    ):
        values = [float(getattr(row, metric_name)) for row in rows]
        metrics[metric_name] = _metric_summary(
            values,
            bootstrap_samples=bootstrap_samples,
            confidence=confidence,
            seed=seed,
        )

    return SummaryRow(
        dataset=dataset_name,
        system=system_name,
        seed=seed,
        retriever=retriever_name,
        chunker=chunker_name,
        num_examples=len(rows),
        metrics=metrics,
        num_documents=num_documents,
        num_chunks=num_chunks,
        avg_chunk_tokens=avg_chunk_tokens,
        avg_retrieval_latency_s=mean(retrieval_times) if retrieval_times else 0.0,
        avg_generation_latency_s=mean(generation_times) if generation_times else 0.0,
    )


def _aggregate_seed_summaries(summaries: list[SummaryRow]) -> list[AggregateSummaryRow]:
    grouped: dict[tuple[str, str, str | None, str | None], list[SummaryRow]] = defaultdict(list)
    for summary in summaries:
        grouped[(summary.dataset, summary.system, summary.retriever, summary.chunker)].append(summary)

    aggregates: list[AggregateSummaryRow] = []
    for (dataset_name, system_name, retriever_name, chunker_name), rows in grouped.items():
        snapshots = [row.numeric_fields() for row in rows]
        field_names = sorted({field for snapshot in snapshots for field in snapshot})
        metric_aggregates: dict[str, AggregateMetricSummary] = {}
        for field_name in field_names:
            values = [snapshot[field_name] for snapshot in snapshots if field_name in snapshot]
            metric_aggregates[field_name] = AggregateMetricSummary(
                mean=mean(values),
                std=stdev(values) if len(values) > 1 else 0.0,
                min=min(values),
                max=max(values),
            )
        aggregates.append(
            AggregateSummaryRow(
                dataset=dataset_name,
                system=system_name,
                retriever=retriever_name,
                chunker=chunker_name,
                num_seeds=len(rows),
                seed_values=sorted(int(row.seed) for row in rows),
                aggregates=metric_aggregates,
            )
        )
    return sorted(aggregates, key=lambda row: (row.dataset, row.system))


@dataclass(slots=True)
class DatasetExperimentRunner:
    config: dict[str, Any]
    dataset_spec: dict[str, Any]
    output_dir: Path
    seed: int
    resources: SharedExperimentResources
    writer: ArtifactWriter

    def run(self) -> list[SummaryRow]:
        dataset_name = self.dataset_spec["name"]
        documents, examples = load_dataset_bundle(self.dataset_spec, seed=self.seed)
        dataset_dir = self.output_dir / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        retriever_specs, legacy_system_names = _get_retriever_specs(self.config)
        bootstrap_samples = int(self.config.get("bootstrap_samples", 1_000))
        confidence = float(self.config.get("confidence_level", 0.95))
        retrieval_top_k = int(self.config.get("retrieval_top_k", 4))

        summaries: list[SummaryRow] = []
        if self.config.get("run_parametric_baseline", True):
            summaries.append(
                self._run_parametric_baseline(
                    dataset_name=dataset_name,
                    dataset_dir=dataset_dir,
                    examples=examples,
                    bootstrap_samples=bootstrap_samples,
                    confidence=confidence,
                )
            )

        for chunker_spec in self.config["chunkers"]:
            chunker_name = chunker_spec["name"]
            chunks = build_chunks(
                documents,
                chunker_spec,
                self.resources.retrieval_tokenizer,
                self.resources.semantic_encoder,
            )
            retriever_factory = RetrieverFactory(
                chunks,
                RetrieverFactoryContext(
                    encoder=self.resources.semantic_encoder,
                    encoder_identifier=self.resources.embedding_model,
                    device=self.resources.device,
                    embedding_batch_size=int(self.config.get("embedding_batch_size", 32)),
                    retrieval_top_k=retrieval_top_k,
                    cache_dir=self.resources.retrieval_cache_dir,
                    cache_namespace=f"{dataset_name}/{chunker_name}",
                ),
            )

            for retriever_spec in retriever_specs:
                system_name = _get_system_name(chunker_name, retriever_spec, legacy_system_names)
                retriever_name = retriever_spec.get("name", retriever_spec.get("type", "dense"))
                retriever = retriever_factory.create(retriever_spec)
                system_output = SystemRunner(
                    dataset_name=dataset_name,
                    system_name=system_name,
                    seed=self.seed,
                    retriever_name=retriever_name,
                    chunker_name=chunker_name,
                    retriever=retriever,
                    generator=self.resources.generator,
                    examples=examples,
                    retrieval_top_k=retrieval_top_k,
                ).run()

                summary = _summarize_prediction_rows(
                    system_output.predictions,
                    dataset_name=dataset_name,
                    system_name=system_name,
                    seed=self.seed,
                    retriever_name=retriever_name,
                    chunker_name=chunker_name,
                    num_documents=len(documents),
                    num_chunks=len(chunks),
                    avg_chunk_tokens=mean(chunk.token_count for chunk in chunks) if chunks else 0.0,
                    retrieval_times=system_output.retrieval_times,
                    generation_times=system_output.generation_times,
                    bootstrap_samples=bootstrap_samples,
                    confidence=confidence,
                )
                self.writer.write_json(dataset_dir / f"{system_name}_predictions.json", system_output.predictions)
                self.writer.write_json(dataset_dir / f"{system_name}_summary.json", summary)
                summaries.append(summary)

        self.writer.write_json(dataset_dir / "all_summaries.json", summaries)
        return summaries

    def _run_parametric_baseline(
        self,
        *,
        dataset_name: str,
        dataset_dir: Path,
        examples: list[QAExample],
        bootstrap_samples: int,
        confidence: float,
    ) -> SummaryRow:
        baseline_predictions: list[PredictionRecord] = []
        for example in tqdm(examples, desc=f"baseline::{dataset_name}::seed_{self.seed}"):
            prediction = self.resources.generator.answer(example.question, context=None)
            answer_scores = answer_metrics(prediction, example.answers)
            baseline_predictions.append(
                PredictionRecord(
                    seed=self.seed,
                    retriever="parametric_only",
                    chunker=None,
                    example_id=example.example_id,
                    question=example.question,
                    gold_answers=example.answers,
                    prediction=prediction,
                    exact_match=answer_scores["exact_match"],
                    f1=answer_scores["f1"],
                )
            )

        summary = _summarize_prediction_rows(
            baseline_predictions,
            dataset_name=dataset_name,
            system_name="parametric_only",
            seed=self.seed,
            retriever_name="parametric_only",
            chunker_name=None,
            bootstrap_samples=bootstrap_samples,
            confidence=confidence,
        )
        self.writer.write_json(dataset_dir / "parametric_only_predictions.json", baseline_predictions)
        self.writer.write_json(dataset_dir / "parametric_only_summary.json", summary)
        return summary


class ExperimentRunner:
    def __init__(self, config: dict[str, Any], output_dir: Path) -> None:
        self.config = config
        self.output_dir = output_dir
        self.writer = ArtifactWriter()
        self.resources = SharedExperimentResources.from_config(config)

    def run(self) -> list[SummaryRow]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        seeds = get_seed_values(self.config)
        use_seed_subdirs = len(seeds) > 1

        all_summaries: list[SummaryRow] = []
        for seed in seeds:
            run_output_dir = self.output_dir / f"seed_{seed}" if use_seed_subdirs else self.output_dir
            for dataset_spec in self.config["datasets"]:
                dataset_runner = DatasetExperimentRunner(
                    config=self.config,
                    dataset_spec=dataset_spec,
                    output_dir=run_output_dir,
                    seed=seed,
                    resources=self.resources,
                    writer=self.writer,
                )
                all_summaries.extend(dataset_runner.run())

        self.writer.write_json(self.output_dir / "experiment_config.json", self.config)
        self.writer.write_json(self.output_dir / "all_results.json", all_summaries)
        self.writer.write_json(self.output_dir / "aggregate_results.json", _aggregate_seed_summaries(all_summaries))
        return all_summaries


def run_dataset_experiments(
    config: dict[str, Any],
    dataset_spec: dict[str, Any],
    output_dir: Path,
    seed: int,
) -> list[dict[str, Any]]:
    runner = DatasetExperimentRunner(
        config=config,
        dataset_spec=dataset_spec,
        output_dir=output_dir,
        seed=seed,
        resources=SharedExperimentResources.from_config(config),
        writer=ArtifactWriter(),
    )
    return [summary.to_flat_dict() for summary in runner.run()]


def run_experiment_suite(config: dict[str, Any], output_dir: Path) -> list[dict[str, Any]]:
    summaries = ExperimentRunner(config, output_dir).run()
    return [summary.to_flat_dict() for summary in summaries]
