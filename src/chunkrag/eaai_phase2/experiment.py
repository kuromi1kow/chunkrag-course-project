from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

from chunkrag.eaai_phase2.config import Phase2Paths, load_phase2_config, phase2_paths
from chunkrag.eaai_phase2.constants import (
    BASELINE_TREE_SHA256,
    CHUNKERS,
    CONDITIONS,
    EXPECTED_DOCUMENTS,
    EXPECTED_ELIGIBLE_ROWS,
    PROTOCOL_COMMIT,
)
from chunkrag.eaai_phase2.gate import fit_gate, save_gate
from chunkrag.eaai_phase2.integrity import (
    repository_root,
    require_within,
    verify_baseline,
    verify_clean_paths,
    verify_protocol_commit,
)
from chunkrag.eaai_phase2.io import (
    add_row_hash,
    canonical_json_bytes,
    iter_jsonl,
    read_json,
    sha256_bytes,
    sha256_file,
    validate_row_hash,
    write_immutable_json,
    write_immutable_jsonl,
)
from chunkrag.eaai_phase2.partition import (
    FrozenPartition,
    make_frozen_partition,
    public_partition_summary,
)


SCIENTIFIC_PATHS = (
    "reports/eaai_phase2_protocol.md",
    "configs/eaai_phase2/techqa_adaptive_v1.json",
    "requirements-eaai-phase2.txt",
    "src/chunkrag/eaai_phase2",
    "scripts/run_eaai_phase2.py",
    "scripts/analyze_eaai_phase2.py",
    "scripts/prepare_eaai_phase2_colab_bundle.py",
)


def _git_head(repo: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _runtime_metadata(device: str) -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for package in (
        "datasets",
        "faiss-cpu",
        "joblib",
        "numpy",
        "pandas",
        "rank-bm25",
        "scikit-learn",
        "sentence-transformers",
        "spacy",
        "torch",
        "transformers",
    ):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    metadata: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "device": device,
        "packages": packages,
    }
    try:
        import torch

        if torch.cuda.is_available():
            metadata["cuda_device_name"] = torch.cuda.get_device_name(0)
            metadata["cuda_version"] = torch.version.cuda
    except Exception:
        pass
    return metadata


def _release_accelerators() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _private_key(question_id: str) -> str:
    return hashlib.sha256(question_id.encode("utf-8")).hexdigest()[:24]


def _condition_order(question_id: str, chunker: str) -> tuple[str, str]:
    digest = hashlib.sha256(f"{question_id}\0{chunker}".encode("utf-8")).digest()
    return CONDITIONS if digest[0] % 2 == 0 else tuple(reversed(CONDITIONS))


def _format_techqa_context(chunks: Iterable[Any]) -> str:
    return "\n\n".join(f"[{index}] {chunk.text}" for index, chunk in enumerate(chunks, start=1))


class Phase2Experiment:
    def __init__(
        self,
        config_path: str | Path,
        *,
        repository: str | Path | None = None,
    ) -> None:
        self.repo = Path(repository).resolve() if repository else repository_root()
        config_file = Path(config_path)
        if not config_file.is_absolute():
            config_file = self.repo / config_file
        self.config_path = config_file.resolve()
        self.config, self.config_sha256 = load_phase2_config(self.config_path)
        self.paths: Phase2Paths = phase2_paths(self.repo, self.config)
        require_within(self.paths.run_results, self.paths.results_root)
        require_within(self.paths.run_artifacts, self.paths.artifacts_root)

    def preflight(self, *, require_committed_implementation: bool) -> dict[str, Any]:
        baseline = verify_baseline(self.repo)
        protocol_sha256 = verify_protocol_commit(self.repo)
        if require_committed_implementation:
            verify_clean_paths(SCIENTIFIC_PATHS, self.repo)
        return {
            "status": "passed",
            "baseline_files_verified": baseline.verified_files,
            "baseline_tree_sha256": baseline.tree_sha256,
            "protocol_commit": PROTOCOL_COMMIT,
            "protocol_sha256": protocol_sha256,
            "config_sha256": self.config_sha256,
            "implementation_commit": _git_head(self.repo),
        }

    def _load_bundle(self) -> tuple[list[Any], dict[str, Any], FrozenPartition]:
        from chunkrag.data import load_techqa_documents_and_examples

        dataset = self.config["dataset"]
        documents, examples = load_techqa_documents_and_examples(
            split=dataset["split"],
            max_examples=EXPECTED_ELIGIBLE_ROWS,
            seed=0,
            revision=dataset["revision"],
        )
        if len(documents) != EXPECTED_DOCUMENTS:
            raise RuntimeError(f"Expected {EXPECTED_DOCUMENTS} TechQA documents, found {len(documents)}")
        if len(examples) != EXPECTED_ELIGIBLE_ROWS:
            raise RuntimeError(
                f"Expected {EXPECTED_ELIGIBLE_ROWS} eligible TechQA rows, found {len(examples)}"
            )
        example_index = {str(example.example_id): example for example in examples}
        if len(example_index) != len(examples):
            raise RuntimeError("TechQA eligible question IDs are not unique")
        partition = make_frozen_partition(example_index)
        return documents, example_index, partition

    @property
    def private_partition_path(self) -> Path:
        return self.paths.run_artifacts / "partitions" / "private_ids.json"

    @property
    def public_partition_path(self) -> Path:
        return self.paths.run_results / "partition_summary.json"

    @property
    def gate_manifest_path(self) -> Path:
        return self.paths.run_artifacts / "gate" / "gate_manifest.json"

    @property
    def gate_model_path(self) -> Path:
        return self.paths.run_artifacts / "gate" / "gate.joblib"

    def prepare_partition(self, *, require_committed: bool = True) -> FrozenPartition:
        preflight = self.preflight(require_committed_implementation=require_committed)
        _, _, partition = self._load_bundle()
        private_payload = partition.as_dict()
        private_payload.update(
            {
                "study": "eaai_phase2",
                "dataset_revision": self.config["dataset"]["revision"],
                "baseline_tree_sha256": BASELINE_TREE_SHA256,
                "protocol_commit": PROTOCOL_COMMIT,
                "config_sha256": self.config_sha256,
                "implementation_commit": preflight["implementation_commit"],
            }
        )
        write_immutable_json(self.private_partition_path, private_payload)
        public_payload = public_partition_summary(partition)
        public_payload.update(
            {
                "study": "eaai_phase2",
                "dataset_revision": self.config["dataset"]["revision"],
                "baseline_tree_sha256": BASELINE_TREE_SHA256,
                "protocol_commit": PROTOCOL_COMMIT,
                "config_sha256": self.config_sha256,
                "preflight": preflight,
            }
        )
        write_immutable_json(self.public_partition_path, public_payload)
        return partition

    def _load_partition(self) -> FrozenPartition:
        if not self.private_partition_path.is_file():
            return self.prepare_partition()
        payload = read_json(self.private_partition_path)
        current_commit = _git_head(self.repo)
        if payload.get("implementation_commit") != current_commit:
            raise RuntimeError(
                "Phase 2 partition is bound to implementation commit "
                f"{payload.get('implementation_commit')}, not current HEAD {current_commit}"
            )
        if payload.get("config_sha256") != self.config_sha256:
            raise RuntimeError("Stored partition config hash differs from the current frozen config")
        partition = FrozenPartition(
            tuple(payload["development"]),
            tuple(payload["heldout_test"]),
            tuple(payload["reserve"]),
            str(payload["partition_sha256"]),
        )
        expected = make_frozen_partition(
            [*partition.development, *partition.heldout_test, *partition.reserve]
        )
        if partition != expected:
            raise RuntimeError("Stored private partition differs from the frozen partition algorithm")
        return partition

    def _retrieval_row_dir(self, split: str, chunker: str) -> Path:
        return self.paths.run_artifacts / "retrieval_rows" / split / chunker

    def _generation_row_dir(self, generator: str, split: str, chunker: str) -> Path:
        return self.paths.run_artifacts / "generation_rows" / generator / split / chunker

    def retrieval_jsonl_path(self, split: str) -> Path:
        return self.paths.run_results / f"retrieval_{split}.jsonl"

    def generation_jsonl_path(self, generator: str, split: str) -> Path:
        return self.paths.run_results / f"generation_{generator}_{split}.jsonl"

    def _require_gate_frozen(self) -> dict[str, Any]:
        if not self.gate_manifest_path.is_file() or not self.gate_model_path.is_file():
            raise RuntimeError("The development-trained gate must be frozen before held-out work")
        manifest = read_json(self.gate_manifest_path)
        if sha256_file(self.gate_model_path) != manifest.get("model_sha256"):
            raise RuntimeError("Frozen gate model hash does not match its manifest")
        return manifest

    def run_retrieval(self, split: str, *, device: str = "auto") -> Path:
        if split not in {"development", "heldout_test"}:
            raise ValueError("Retrieval is limited to development or heldout_test")
        preflight = self.preflight(require_committed_implementation=True)
        partition = self._load_partition()
        gate_manifest = self._require_gate_frozen() if split == "heldout_test" else None
        documents, examples, current_partition = self._load_bundle()
        if current_partition.partition_sha256 != partition.partition_sha256:
            raise RuntimeError("Reloaded partition differs from the frozen artifact")

        from sentence_transformers import CrossEncoder, SentenceTransformer
        from transformers import AutoTokenizer

        from chunkrag.eaai_phase2.retrieval import (
            PairedRetrievalEngine,
            serialize_ranking,
        )
        from chunkrag.evaluation import retrieval_metrics
        from chunkrag.generation import resolve_device
        from chunkrag.pipeline import build_chunks
        from chunkrag.retrieval import BM25Retriever, DenseRetriever

        resolved_device = resolve_device(device)
        retrieval = self.config["retrieval"]
        tokenizer = AutoTokenizer.from_pretrained(
            retrieval["chunking_tokenizer"],
            revision=retrieval["chunking_tokenizer_revision"],
        )
        tokenizer.model_max_length = 1_000_000
        encoder = SentenceTransformer(
            retrieval["embedding_model"],
            revision=retrieval["embedding_model_revision"],
            device=resolved_device,
        )
        reranker = self.config["reranker"]
        cross_encoder = CrossEncoder(
            reranker["model"],
            revision=reranker["revision"],
            device=resolved_device,
        )

        split_ids = partition.ids_for(split)
        expected_rows = len(split_ids) * len(CHUNKERS)
        row_paths: list[Path] = []
        for chunker_spec in self.config["chunkers"]:
            chunker = chunker_spec["name"]
            chunks = build_chunks(documents, chunker_spec, tokenizer, None)
            dense = DenseRetriever(
                encoder=encoder,
                encoder_identifier=(
                    f"{retrieval['embedding_model']}@{retrieval['embedding_model_revision']}"
                ),
                device=resolved_device,
                batch_size=retrieval["embedding_batch_size"],
                cache_dir=self.paths.run_artifacts / "embedding_cache",
                cache_namespace=f"techqa/{chunker}",
                query_prefix=retrieval["query_prefix"],
            )
            dense.build(chunks)
            bm25 = BM25Retriever()
            bm25.build(chunks)
            engine = PairedRetrievalEngine(
                dense_retriever=dense,
                bm25_retriever=bm25,
                cross_encoder=cross_encoder,
                chunker=chunker,
                candidate_pool_size=retrieval["candidate_pool_size"],
                final_top_k=retrieval["final_top_k"],
                dense_weight=retrieval["dense_weight"],
                bm25_weight=retrieval["bm25_weight"],
                rrf_k=retrieval["rrf_k"],
                reranker_batch_size=reranker["batch_size"],
            )
            row_dir = self._retrieval_row_dir(split, chunker)
            for question_id in split_ids:
                row_path = row_dir / f"{_private_key(question_id)}.json"
                row_paths.append(row_path)
                if row_path.is_file():
                    existing = read_json(row_path)
                    validate_row_hash(existing)
                    if (
                        existing.get("question_id") != question_id
                        or existing.get("chunker") != chunker
                        or existing.get("split") != split
                        or existing.get("config_sha256") != self.config_sha256
                    ):
                        raise RuntimeError(f"Checkpoint identity mismatch: {row_path}")
                    continue
                example = examples[question_id]
                paired = engine.retrieve_pair(example.question)
                hybrid_chunks = [chunk for chunk, _ in paired.hybrid_top_k]
                reranked_chunks = [chunk for chunk, _ in paired.reranked_top_k]
                payload = {
                    "schema_version": 1,
                    "study_stage": "eaai_phase2_retrieval",
                    "dataset": "techqa",
                    "dataset_revision": self.config["dataset"]["revision"],
                    "split": split,
                    "question_id": question_id,
                    "question": example.question,
                    "reference_answers": list(example.answers),
                    "relevant_document_ids": list(example.relevant_doc_ids),
                    "chunker": chunker,
                    "partition_sha256": partition.partition_sha256,
                    "baseline_tree_sha256": BASELINE_TREE_SHA256,
                    "protocol_commit": PROTOCOL_COMMIT,
                    "config_sha256": self.config_sha256,
                    "implementation_commit": preflight["implementation_commit"],
                    "gate_sha256": None if gate_manifest is None else gate_manifest["model_sha256"],
                    "features": paired.features,
                    "dense_candidates": serialize_ranking(paired.dense),
                    "bm25_candidates": serialize_ranking(paired.bm25),
                    "fused_candidates": serialize_ranking(paired.fused_candidates),
                    "reranked_candidates": serialize_ranking(paired.reranked_candidates),
                    "hybrid": {
                        "top_k": serialize_ranking(paired.hybrid_top_k),
                        "metrics": retrieval_metrics(hybrid_chunks, example),
                    },
                    "reranked": {
                        "top_k": serialize_ranking(paired.reranked_top_k),
                        "metrics": retrieval_metrics(reranked_chunks, example),
                    },
                    "timing_seconds": {
                        "dense": paired.dense_latency_s,
                        "bm25": paired.bm25_latency_s,
                        "fusion": paired.fusion_latency_s,
                        "hybrid_total": paired.hybrid_retrieval_latency_s,
                        "reranker_only": paired.reranker_latency_s,
                        "hybrid_plus_reranker": (
                            paired.hybrid_retrieval_latency_s + paired.reranker_latency_s
                        ),
                    },
                }
                write_immutable_json(row_path, add_row_hash(payload))

        if len(row_paths) != expected_rows or any(not path.is_file() for path in row_paths):
            raise RuntimeError(f"Incomplete retrieval stage: expected {expected_rows} rows")
        rows = [read_json(path) for path in sorted(row_paths)]
        for row in rows:
            validate_row_hash(row)
        write_immutable_jsonl(self.retrieval_jsonl_path(split), rows)
        manifest = {
            "schema_version": 1,
            "stage": f"retrieval_{split}",
            "status": "complete",
            "row_count": len(rows),
            "output_sha256": sha256_file(self.retrieval_jsonl_path(split)),
            "partition_sha256": partition.partition_sha256,
            "gate_sha256": None if gate_manifest is None else gate_manifest["model_sha256"],
            "baseline_tree_sha256": BASELINE_TREE_SHA256,
            "protocol_commit": PROTOCOL_COMMIT,
            "config_sha256": self.config_sha256,
            "implementation_commit": preflight["implementation_commit"],
            "runtime": _runtime_metadata(resolved_device),
        }
        write_immutable_json(self.paths.run_artifacts / "manifests" / f"retrieval_{split}.json", manifest)
        del cross_encoder, encoder
        _release_accelerators()
        return self.retrieval_jsonl_path(split)

    def _load_retrieval_rows(self, split: str) -> list[dict[str, Any]]:
        path = self.retrieval_jsonl_path(split)
        if not path.is_file():
            raise FileNotFoundError(f"Retrieval stage is missing: {path}")
        rows = list(iter_jsonl(path))
        expected = (200 if split in {"development", "heldout_test"} else 0) * len(CHUNKERS)
        if len(rows) != expected:
            raise RuntimeError(f"Expected {expected} retrieval rows for {split}, found {len(rows)}")
        for row in rows:
            validate_row_hash(row)
        return rows

    def run_generation(
        self,
        split: str,
        generator_name: str,
        *,
        device: str = "auto",
    ) -> Path:
        if split not in {"development", "heldout_test"}:
            raise ValueError("Generation is limited to development or heldout_test")
        if generator_name not in self.config["generators"]:
            raise ValueError(f"Unknown generator: {generator_name}")
        if generator_name == "mistral" and split != "heldout_test":
            raise ValueError("The frozen protocol runs Mistral only on heldout_test")
        preflight = self.preflight(require_committed_implementation=True)
        gate_manifest = self._require_gate_frozen() if split == "heldout_test" else None
        retrieval_rows = self._load_retrieval_rows(split)
        documents, examples, partition = self._load_bundle()

        from transformers import AutoTokenizer

        from chunkrag.evaluation import Timer, answer_metrics, retrieval_metrics
        from chunkrag.generation import QAGenerator
        from chunkrag.pipeline import build_chunks

        retrieval = self.config["retrieval"]
        tokenizer = AutoTokenizer.from_pretrained(
            retrieval["chunking_tokenizer"],
            revision=retrieval["chunking_tokenizer_revision"],
        )
        tokenizer.model_max_length = 1_000_000
        chunk_maps: dict[str, dict[str, Any]] = {}
        for chunker_spec in self.config["chunkers"]:
            chunks = build_chunks(documents, chunker_spec, tokenizer, None)
            chunk_maps[chunker_spec["name"]] = {chunk.chunk_id: chunk for chunk in chunks}

        generator_spec = self.config["generators"][generator_name]
        generation_spec = self.config["generation"]
        generator = QAGenerator(
            model_name=generator_spec["model"],
            revision=generator_spec["revision"],
            device=device,
            max_input_tokens=generation_spec["max_input_tokens"],
            max_new_tokens=generation_spec["max_new_tokens"],
            torch_dtype=generator_spec["torch_dtype"],
            use_device_map=generator_spec["use_device_map"],
        )
        row_paths: list[Path] = []
        for retrieval_row in retrieval_rows:
            question_id = str(retrieval_row["question_id"])
            chunker = str(retrieval_row["chunker"])
            example = examples[question_id]
            chunk_map = chunk_maps[chunker]
            for condition in _condition_order(question_id, chunker):
                row_dir = self._generation_row_dir(generator_name, split, chunker)
                row_path = row_dir / f"{_private_key(question_id)}__{condition}.json"
                row_paths.append(row_path)
                if row_path.is_file():
                    existing = read_json(row_path)
                    validate_row_hash(existing)
                    if (
                        existing.get("question_id") != question_id
                        or existing.get("condition") != condition
                        or existing.get("generator") != generator_name
                    ):
                        raise RuntimeError(f"Generation checkpoint identity mismatch: {row_path}")
                    continue
                selected = retrieval_row[condition]["top_k"]
                selected_chunks = [chunk_map[str(item["chunk_id"])] for item in selected]
                context = _format_techqa_context(selected_chunks)
                with Timer() as generation_timer:
                    prediction = generator.answer_with_style(
                        example.question,
                        context=context,
                        answer_style=generation_spec["answer_style"],
                        max_new_tokens=generation_spec["max_new_tokens"],
                    )
                answer_scores = answer_metrics(prediction, example.answers)
                recalculated_retrieval = retrieval_metrics(selected_chunks, example)
                stored_retrieval = retrieval_row[condition]["metrics"]
                for metric, value in recalculated_retrieval.items():
                    if abs(float(value) - float(stored_retrieval[metric])) > 1e-12:
                        raise RuntimeError(
                            f"Retrieval metric mismatch for {question_id}/{chunker}/{condition}/{metric}"
                        )
                trace = dict(generator.last_trace)
                payload = {
                    "schema_version": 1,
                    "study_stage": "eaai_phase2_generation",
                    "dataset": "techqa",
                    "dataset_revision": self.config["dataset"]["revision"],
                    "split": split,
                    "generator": generator_name,
                    "generator_model": generator_spec["model"],
                    "generator_revision": generator_spec["revision"],
                    "generator_role": generator_spec["role"],
                    "question_id": question_id,
                    "question": example.question,
                    "reference_answers": list(example.answers),
                    "chunker": chunker,
                    "condition": condition,
                    "condition_order": list(_condition_order(question_id, chunker)),
                    "partition_sha256": partition.partition_sha256,
                    "baseline_tree_sha256": BASELINE_TREE_SHA256,
                    "protocol_commit": PROTOCOL_COMMIT,
                    "config_sha256": self.config_sha256,
                    "implementation_commit": preflight["implementation_commit"],
                    "gate_sha256": None if gate_manifest is None else gate_manifest["model_sha256"],
                    "features": retrieval_row["features"],
                    "retrieved_chunk_ids": [chunk.chunk_id for chunk in selected_chunks],
                    "retrieved_document_ids": [chunk.doc_id for chunk in selected_chunks],
                    "retrieval_metrics": recalculated_retrieval,
                    "raw_prediction": trace.get("raw_prediction"),
                    "prediction": prediction,
                    "exact_match": float(answer_scores["exact_match"]),
                    "f1": float(answer_scores["f1"]),
                    "full_prompt_tokens": trace.get("full_prompt_tokens"),
                    "used_prompt_tokens": trace.get("used_prompt_tokens"),
                    "context_truncated": trace.get("context_truncated"),
                    "generated_tokens": trace.get("generated_tokens"),
                    "generation_max_new_tokens": trace.get("generation_max_new_tokens"),
                    "generation_length_capped": trace.get("generation_length_capped"),
                    "timing_seconds": {
                        "hybrid_retrieval": retrieval_row["timing_seconds"]["hybrid_total"],
                        "reranker_only": (
                            0.0
                            if condition == "hybrid"
                            else retrieval_row["timing_seconds"]["reranker_only"]
                        ),
                        "generation": generation_timer.elapsed,
                        "end_to_end_component_sum": (
                            retrieval_row["timing_seconds"]["hybrid_total"]
                            + (
                                0.0
                                if condition == "hybrid"
                                else retrieval_row["timing_seconds"]["reranker_only"]
                            )
                            + generation_timer.elapsed
                        ),
                    },
                }
                write_immutable_json(row_path, add_row_hash(payload))

        expected_rows = len(retrieval_rows) * len(CONDITIONS)
        if len(row_paths) != expected_rows or any(not path.is_file() for path in row_paths):
            raise RuntimeError(f"Incomplete generation stage: expected {expected_rows} rows")
        rows = [read_json(path) for path in sorted(row_paths)]
        for row in rows:
            validate_row_hash(row)
        output_path = self.generation_jsonl_path(generator_name, split)
        write_immutable_jsonl(output_path, rows)
        model_dtype = str(next(generator.model.parameters()).dtype)
        manifest = {
            "schema_version": 1,
            "stage": f"generation_{generator_name}_{split}",
            "status": "complete",
            "row_count": len(rows),
            "output_sha256": sha256_file(output_path),
            "partition_sha256": partition.partition_sha256,
            "gate_sha256": None if gate_manifest is None else gate_manifest["model_sha256"],
            "generator": generator_spec,
            "model_dtype": model_dtype,
            "baseline_tree_sha256": BASELINE_TREE_SHA256,
            "protocol_commit": PROTOCOL_COMMIT,
            "config_sha256": self.config_sha256,
            "implementation_commit": preflight["implementation_commit"],
            "runtime": _runtime_metadata(str(generator.device)),
        }
        write_immutable_json(
            self.paths.run_artifacts / "manifests" / f"generation_{generator_name}_{split}.json",
            manifest,
        )
        del generator
        _release_accelerators()
        return output_path

    def _load_generation_rows(self, generator: str, split: str) -> list[dict[str, Any]]:
        path = self.generation_jsonl_path(generator, split)
        if not path.is_file():
            raise FileNotFoundError(f"Generation stage is missing: {path}")
        rows = list(iter_jsonl(path))
        expected = 200 * len(CHUNKERS) * len(CONDITIONS)
        if len(rows) != expected:
            raise RuntimeError(f"Expected {expected} generation rows, found {len(rows)}")
        for row in rows:
            validate_row_hash(row)
        return rows

    def fit_development_gate(self) -> Path:
        preflight = self.preflight(require_committed_implementation=True)
        if self.gate_manifest_path.exists() or self.gate_model_path.exists():
            manifest = self._require_gate_frozen()
            return self.gate_manifest_path
        heldout_generation_dir = self.paths.run_artifacts / "generation_rows" / "qwen" / "heldout_test"
        heldout_retrieval_dir = self.paths.run_artifacts / "retrieval_rows" / "heldout_test"
        if (
            self.generation_jsonl_path("qwen", "heldout_test").exists()
            or self.retrieval_jsonl_path("heldout_test").exists()
            or (heldout_generation_dir.exists() and any(heldout_generation_dir.rglob("*.json")))
            or (heldout_retrieval_dir.exists() and any(heldout_retrieval_dir.rglob("*.json")))
        ):
            raise RuntimeError("Cannot fit a new gate after any held-out artifact exists")
        retrieval_rows = self._load_retrieval_rows("development")
        generation_rows = self._load_generation_rows("qwen", "development")
        generation_index = {
            (str(row["question_id"]), str(row["chunker"]), str(row["condition"])): row
            for row in generation_rows
        }
        if len(generation_index) != len(generation_rows):
            raise RuntimeError("Duplicate development generation rows")
        features: list[dict[str, Any]] = []
        labels: list[int] = []
        training_keys: list[str] = []
        for row in retrieval_rows:
            key = (str(row["question_id"]), str(row["chunker"]))
            hybrid = generation_index[(*key, "hybrid")]
            reranked = generation_index[(*key, "reranked")]
            features.append(dict(row["features"]))
            labels.append(int(float(reranked["f1"]) > float(hybrid["f1"])))
            training_keys.append(f"{key[0]}\0{key[1]}")
        model, metadata = fit_gate(features, labels)
        model_sha256 = save_gate(model, self.gate_model_path)
        partition = self._load_partition()
        manifest = {
            "schema_version": 1,
            "stage": "fit_gate",
            "status": "complete",
            "trained_split": "development",
            "heldout_outputs_absent_at_fit": True,
            "model_path": str(self.gate_model_path.relative_to(self.repo)),
            "model_sha256": model_sha256,
            "training_key_sha256": sha256_bytes(canonical_json_bytes(sorted(training_keys))),
            "partition_sha256": partition.partition_sha256,
            "baseline_tree_sha256": BASELINE_TREE_SHA256,
            "protocol_commit": PROTOCOL_COMMIT,
            "config_sha256": self.config_sha256,
            "implementation_commit": preflight["implementation_commit"],
            "metadata": metadata,
        }
        write_immutable_json(self.gate_manifest_path, manifest)
        return self.gate_manifest_path

    def run_qwen(self, *, device: str = "auto") -> dict[str, str]:
        self.prepare_partition()
        self.run_retrieval("development", device=device)
        self.run_generation("development", "qwen", device=device)
        self.fit_development_gate()
        self.run_retrieval("heldout_test", device=device)
        self.run_generation("heldout_test", "qwen", device=device)
        return {
            "gate_manifest": str(self.gate_manifest_path),
            "heldout_generation": str(self.generation_jsonl_path("qwen", "heldout_test")),
        }

    def run_mistral(self, *, device: str = "auto") -> Path:
        self._require_gate_frozen()
        if not self.retrieval_jsonl_path("heldout_test").is_file():
            raise RuntimeError("Held-out retrieval must be complete before Mistral replication")
        return self.run_generation("heldout_test", "mistral", device=device)
