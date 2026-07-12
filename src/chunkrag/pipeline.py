from __future__ import annotations

import json
import hashlib
import importlib.metadata
import platform
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from chunkrag.chunking import ChunkingContext, build_document_chunks
from chunkrag.data import (
    load_hotpot_documents_and_examples,
    load_squad_documents_and_examples,
    load_techqa_documents_and_examples,
)
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


def canonical_config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def source_tree_hash() -> str:
    root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    paths = sorted((root / "src").rglob("*.py")) + [root / "scripts" / "run_experiments.py"]
    for path in paths:
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def runtime_manifest(
    config: dict[str, Any],
    device: str,
    num_summaries: int,
    source_sha256: str,
    generator_dtype: str | None,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        git_dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        git_commit = None
        git_dirty = None

    packages = {}
    for package in (
        "datasets",
        "faiss-cpu",
        "langchain-text-splitters",
        "numpy",
        "rank-bm25",
        "sentence-transformers",
        "spacy",
        "torch",
        "transformers",
    ):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    return {
        "status": "complete",
        "config_sha256": canonical_config_hash(config),
        "source_tree_sha256": source_sha256,
        "git_commit": git_commit,
        "git_worktree_dirty_at_run": git_dirty,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "device": device,
        "generator_dtype": generator_dtype,
        "packages": packages,
        "num_summary_rows": num_summaries,
    }


def load_dataset_bundle(spec: dict[str, Any], seed: int) -> tuple[list[Document], list[QAExample]]:
    name = spec["name"]
    if name == "squad_v2":
        return load_squad_documents_and_examples(
            split=spec.get("split", "validation"),
            max_examples=spec["max_examples"],
            candidate_pool_size=spec.get("candidate_pool_size", spec["max_examples"] * 5),
            seed=seed,
            answerable_only=spec.get("answerable_only", True),
            revision=spec.get("revision"),
        )
    if name == "hotpot_qa":
        return load_hotpot_documents_and_examples(
            split=spec.get("split", "validation"),
            max_examples=spec["max_examples"],
            config_name=spec.get("config", "distractor"),
            seed=seed,
            revision=spec.get("revision"),
        )
    if name == "techqa":
        return load_techqa_documents_and_examples(
            split=spec.get("split", "train"),
            max_examples=spec["max_examples"],
            seed=seed,
            revision=spec.get("revision"),
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
    generator: Generator | None
    retrieval_tokenizer: PreTrainedTokenizerBase
    semantic_encoder: SentenceTransformer
    embedding_model: str
    retrieval_cache_dir: Path

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> SharedExperimentResources:
        device = resolve_device(config.get("device", "auto"))
        generator: Generator | None = None
        if config.get("run_generation", True):
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
                    revision=config.get("generator_model_revision"),
                    device=config.get("device", "auto"),
                    max_input_tokens=config.get("generation_max_input_tokens", 768),
                    max_new_tokens=config.get("max_new_tokens", 32),
                    torch_dtype=config.get("generator_torch_dtype"),
                    use_device_map=config.get("generator_use_device_map", False),
                )
        embedding_revision = config.get("embedding_model_revision")
        chunking_tokenizer = config.get("chunking_tokenizer", config["embedding_model"])
        chunking_tokenizer_revision = config.get(
            "chunking_tokenizer_revision",
            embedding_revision if chunking_tokenizer == config["embedding_model"] else None,
        )
        retrieval_tokenizer = AutoTokenizer.from_pretrained(
            chunking_tokenizer,
            revision=chunking_tokenizer_revision,
        )
        retrieval_tokenizer.model_max_length = 1_000_000
        encoder_kwargs = {"device": device}
        if embedding_revision is not None:
            encoder_kwargs["revision"] = embedding_revision
        semantic_encoder = SentenceTransformer(config["embedding_model"], **encoder_kwargs)
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
    generator: Generator | None
    examples: list[QAExample]
    retrieval_top_k: int
    answer_style: str = "extractive"
    max_new_tokens: int | None = None

    def _format_context(self, retrieved_chunks: list[Chunk]) -> str:
        parts: list[str] = []
        include_titles = self.dataset_name == "hotpot_qa"
        for index, chunk in enumerate(retrieved_chunks, start=1):
            title = chunk.title.strip() if chunk.title else chunk.doc_id
            if include_titles:
                parts.append(f"[{index}] Title: {title}\nPassage: {chunk.text}")
            else:
                parts.append(f"[{index}] {chunk.text}")
        return "\n\n".join(parts)

    def run(self) -> SystemRunOutput:
        predictions: list[PredictionRecord] = []
        retrieval_times: list[float] = []
        generation_times: list[float] = []

        for example in tqdm(self.examples, desc=f"eval::{self.dataset_name}::{self.system_name}::seed_{self.seed}"):
            with Timer() as retrieval_timer:
                retrieved = self.retriever.retrieve(example.question, self.retrieval_top_k)
            retrieved_chunks = [chunk for chunk, _ in retrieved]
            context = self._format_context(retrieved_chunks)
            if self.generator is None:
                prediction = ""
                generation_elapsed = 0.0
                generation_trace: dict[str, object] = {}
            else:
                with Timer() as generation_timer:
                    answer_with_style = getattr(self.generator, "answer_with_style", None)
                    if answer_with_style is None:
                        if self.answer_style != "extractive":
                            raise TypeError(
                                f"{type(self.generator).__name__} does not support "
                                f"answer_style={self.answer_style!r}"
                            )
                        prediction = self.generator.answer(example.question, context=context)
                    else:
                        prediction = answer_with_style(
                            example.question,
                            context=context,
                            answer_style=self.answer_style,
                            max_new_tokens=self.max_new_tokens,
                        )
                generation_elapsed = generation_timer.elapsed
                generation_trace = dict(getattr(self.generator, "last_trace", {}))

            retrieval_times.append(retrieval_timer.elapsed)
            generation_times.append(generation_elapsed)
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
                    answer_string_visible_at_k=retrieval_scores["answer_string_visible_at_k"],
                    raw_prediction=generation_trace.get("raw_prediction"),
                    full_prompt_tokens=generation_trace.get("full_prompt_tokens"),
                    used_prompt_tokens=generation_trace.get("used_prompt_tokens"),
                    context_truncated=generation_trace.get("context_truncated"),
                    refinement_applied=generation_trace.get("refinement_applied"),
                    generated_tokens=generation_trace.get("generated_tokens"),
                    generation_max_new_tokens=generation_trace.get("generation_max_new_tokens"),
                    generation_length_capped=generation_trace.get("generation_length_capped"),
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
        "answer_string_visible_at_k",
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
        if self.config.get("run_parametric_baseline", True) and self.resources.generator is not None:
            summaries.append(
                self._run_parametric_baseline(
                    dataset_name=dataset_name,
                    dataset_dir=dataset_dir,
                    examples=examples,
                    bootstrap_samples=bootstrap_samples,
                    confidence=confidence,
                    answer_style=str(self.dataset_spec.get("answer_style", "extractive")),
                    max_new_tokens=self.dataset_spec.get("max_new_tokens"),
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
                    encoder_identifier=(
                        f"{self.resources.embedding_model}@{self.config['embedding_model_revision']}"
                        if self.config.get("embedding_model_revision")
                        else self.resources.embedding_model
                    ),
                    device=self.resources.device,
                    embedding_batch_size=int(self.config.get("embedding_batch_size", 32)),
                    retrieval_top_k=retrieval_top_k,
                    cache_dir=self.resources.retrieval_cache_dir,
                    cache_namespace=f"{dataset_name}/{chunker_name}",
                    query_prefix=str(self.config.get("retrieval_query_prefix", "")),
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
                    answer_style=str(self.dataset_spec.get("answer_style", "extractive")),
                    max_new_tokens=self.dataset_spec.get("max_new_tokens"),
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
        answer_style: str,
        max_new_tokens: int | None,
    ) -> SummaryRow:
        if self.resources.generator is None:
            raise RuntimeError("Parametric baseline requires generation to be enabled")
        baseline_predictions: list[PredictionRecord] = []
        for example in tqdm(examples, desc=f"baseline::{dataset_name}::seed_{self.seed}"):
            answer_with_style = getattr(self.resources.generator, "answer_with_style", None)
            if answer_with_style is None:
                if answer_style != "extractive":
                    raise TypeError(
                        f"{type(self.resources.generator).__name__} does not support "
                        f"answer_style={answer_style!r}"
                    )
                prediction = self.resources.generator.answer(example.question, context=None)
            else:
                prediction = answer_with_style(
                    example.question,
                    context=None,
                    answer_style=answer_style,
                    max_new_tokens=max_new_tokens,
                )
            generation_trace = dict(getattr(self.resources.generator, "last_trace", {}))
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
                    raw_prediction=generation_trace.get("raw_prediction"),
                    full_prompt_tokens=generation_trace.get("full_prompt_tokens"),
                    used_prompt_tokens=generation_trace.get("used_prompt_tokens"),
                    context_truncated=generation_trace.get("context_truncated"),
                    refinement_applied=generation_trace.get("refinement_applied"),
                    generated_tokens=generation_trace.get("generated_tokens"),
                    generation_max_new_tokens=generation_trace.get("generation_max_new_tokens"),
                    generation_length_capped=generation_trace.get("generation_length_capped"),
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
        self.source_tree_sha256 = source_tree_hash()
        self.resources = SharedExperimentResources.from_config(config)
        if isinstance(self.resources.generator, QAGenerator):
            self.generator_dtype = str(next(self.resources.generator.model.parameters()).dtype)
        else:
            self.generator_dtype = None

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
        self.writer.write_json(
            self.output_dir / "run_manifest.json",
            runtime_manifest(
                self.config,
                self.resources.device,
                len(all_summaries),
                self.source_tree_sha256,
                self.generator_dtype,
            ),
        )
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
