#!/usr/bin/env python3
"""Summarize the reviewer-driven retrieval and local-generator robustness experiments.

This script only reads archived experiment artifacts. It does not rerun data
loading, retrieval, or generation. All paired tests use aligned questions as
the sampling unit and deterministic pseudo-random streams.
"""

from __future__ import annotations

import json
import hashlib
import math
import random
import re
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Callable, Hashable

from chunkrag.evaluation import bootstrap_confidence_interval
from chunkrag.text_utils import best_exact_match, best_f1, normalize_answer


REPO_ROOT = Path(__file__).resolve().parents[1]
RETRIEVAL_ROOTS = {
    "MiniLM": REPO_ROOT / "outputs" / "reviewer_robustness_retrieval_minilm",
    "BGE-small": REPO_ROOT / "outputs" / "reviewer_robustness_retrieval_bge",
}
RETRIEVAL_CONFIGS = {
    "MiniLM": REPO_ROOT / "configs" / "reviewer_robustness_retrieval_minilm.json",
    "BGE-small": REPO_ROOT / "configs" / "reviewer_robustness_retrieval_bge.json",
}
QWEN_ROOT = REPO_ROOT / "outputs" / "reviewer_robustness_qwen"
QWEN_CONFIG = REPO_ROOT / "configs" / "reviewer_robustness_qwen.json"
MISTRAL_ROOT = REPO_ROOT / "outputs" / "reviewer_robustness_mistral"
MISTRAL_CONFIG = REPO_ROOT / "configs" / "reviewer_robustness_mistral.json"
GENERATION_SPECS = (
    ("Qwen2.5-1.5B", QWEN_ROOT, QWEN_CONFIG),
    ("Mistral-7B", MISTRAL_ROOT, MISTRAL_CONFIG),
)
OUTPUT_JSON = REPO_ROOT / "outputs" / "reviewer_robustness_analysis.json"
OUTPUT_MARKDOWN = REPO_ROOT / "outputs" / "reviewer_robustness_analysis.md"
RETRIEVAL_TEX = REPO_ROOT / "reports" / "generated" / "table_robustness_retrieval.tex"
GENERATION_TEX = REPO_ROOT / "reports" / "generated" / "table_robustness_generation.tex"

DATASETS = ("squad_v2", "hotpot_qa", "techqa")
DATASET_LABELS = {
    "squad_v2": "SQuAD 2.0",
    "hotpot_qa": "HotpotQA",
    "techqa": "TechQA",
}
RETRIEVERS = ("dense", "bm25", "hybrid")
CHUNKERS = ("fixed_128", "fixed_254", "recursive_254", "sentence_254")
GENERATION_SYSTEMS = (
    "parametric_only",
    "hybrid__fixed_128",
    "hybrid__fixed_254",
    "hybrid__recursive_254",
    "hybrid__sentence_254",
)
COMPARATORS = ("fixed_128", "fixed_254", "sentence_254")
BOOTSTRAP_DRAWS = 20_000
RANDOMIZATION_DRAWS = 100_000
RANDOM_SEED = 8677


def read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Required experiment artifact is missing: {path}")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_completed_root(root: Path, expected_config_path: Path) -> dict[str, Any]:
    expected = read_json(expected_config_path)
    actual = read_json(root / "experiment_config.json")
    if actual != expected:
        raise ValueError(f"Archived config does not match {expected_config_path}: {root}")
    manifest = read_json(root / "run_manifest.json")
    if manifest.get("status") != "complete":
        raise ValueError(f"Run is not marked complete: {root}")
    expected_hash = config_hash(actual)
    if manifest.get("config_sha256") != expected_hash:
        raise ValueError(f"Config hash mismatch in {root / 'run_manifest.json'}")
    source_hash = manifest.get("source_tree_sha256")
    if not isinstance(source_hash, str) or re.fullmatch(r"[0-9a-f]{64}", source_hash) is None:
        raise ValueError(f"Missing or invalid source hash in {root / 'run_manifest.json'}")
    summary_rows = read_json(root / "all_results.json")
    if not isinstance(summary_rows, list):
        raise TypeError(f"Expected a list in {root / 'all_results.json'}")
    manifest_row_count = manifest.get("num_summary_rows")
    if isinstance(manifest_row_count, bool) or not isinstance(manifest_row_count, int):
        raise ValueError(f"Missing or invalid summary-row count in {root / 'run_manifest.json'}")
    if manifest_row_count != len(summary_rows):
        raise ValueError(
            f"Manifest summary-row count mismatch in {root}: "
            f"{manifest_row_count} != {len(summary_rows)}"
        )
    return actual


def manifest_source_hash(root: Path) -> str:
    manifest = read_json(root / "run_manifest.json")
    source_hash = manifest.get("source_tree_sha256")
    if not isinstance(source_hash, str):
        raise ValueError(f"Missing source hash in {root / 'run_manifest.json'}")
    return source_hash


def unique_index(
    rows: list[dict[str, Any]],
    key_fn: Callable[[dict[str, Any]], Hashable],
    *,
    label: str,
) -> dict[Any, dict[str, Any]]:
    index: dict[Any, dict[str, Any]] = {}
    for row in rows:
        key = key_fn(row)
        if key in index:
            raise ValueError(f"Duplicate {label}: {key}")
        index[key] = row
    return index


def assert_close(actual: float, expected: float, label: str, tolerance: float = 1e-9) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
        raise ValueError(f"Metric mismatch for {label}: {actual} != {expected}")


def rounded(value: float, digits: int = 8) -> float:
    return round(float(value), digits)


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile requires at least one value")
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def paired_bootstrap_ci(
    differences: list[float], *, draws: int, rng: random.Random
) -> tuple[float, float]:
    n = len(differences)
    samples = [mean(differences[rng.randrange(n)] for _ in range(n)) for _ in range(draws)]
    return percentile(samples, 0.025), percentile(samples, 0.975)


def paired_randomization_p(
    differences: list[float], *, draws: int, rng: random.Random
) -> float:
    observed = abs(mean(differences))
    if observed == 0.0:
        return 1.0
    extreme = 0
    for _ in range(draws):
        randomized = mean(value if rng.random() < 0.5 else -value for value in differences)
        if abs(randomized) >= observed - 1e-15:
            extreme += 1
    return (extreme + 1) / (draws + 1)


def holm_adjust(raw: dict[str, float]) -> dict[str, float]:
    ordered = sorted(raw, key=lambda key: (raw[key], key))
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, key in enumerate(ordered):
        candidate = min(1.0, (total - rank) * raw[key])
        running = max(running, candidate)
        adjusted[key] = running
    return adjusted


def answer_visibility_is_applicable(dataset: str, gold_answers: list[str]) -> bool:
    """Exclude literal yes/no matching, which is not an evidence diagnostic."""
    if dataset != "hotpot_qa":
        return True
    normalized = {normalize_answer(str(answer)) for answer in gold_answers}
    return not bool(normalized & {"yes", "no"})


def applicable_answer_visibility_values(
    dataset: str,
    rows: list[dict[str, Any]],
) -> list[float]:
    return [
        float(row["answer_string_visible_at_k"])
        for row in rows
        if answer_visibility_is_applicable(
            dataset,
            [str(answer) for answer in row["gold_answers"]],
        )
    ]


def recompute_answer_vectors(
    rows: list[dict[str, Any]],
    *,
    label: str,
) -> tuple[list[float], list[float]]:
    exact_match_values: list[float] = []
    f1_values: list[float] = []
    for row in rows:
        gold_answers = [str(answer) for answer in row["gold_answers"]]
        prediction = str(row["prediction"])
        recomputed_em = best_exact_match(prediction, gold_answers)
        recomputed_f1 = best_f1(prediction, gold_answers)
        assert_close(
            float(row["exact_match"]),
            recomputed_em,
            f"{label}/{row['example_id']}/stored EM",
        )
        assert_close(
            float(row["f1"]),
            recomputed_f1,
            f"{label}/{row['example_id']}/stored F1",
        )
        exact_match_values.append(recomputed_em)
        f1_values.append(recomputed_f1)
    return exact_match_values, f1_values


def _require_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Expected integer {label}; found {value!r}")
    return value


def _require_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"Expected boolean {label}; found {value!r}")
    return value


def validate_generation_trace(
    row: dict[str, Any],
    *,
    expected_input_limit: int,
    expected_output_limit: int,
    label: str,
) -> None:
    full_tokens = _require_int(row.get("full_prompt_tokens"), f"{label}/full_prompt_tokens")
    used_tokens = _require_int(row.get("used_prompt_tokens"), f"{label}/used_prompt_tokens")
    generated_tokens = _require_int(row.get("generated_tokens"), f"{label}/generated_tokens")
    output_limit = _require_int(
        row.get("generation_max_new_tokens"),
        f"{label}/generation_max_new_tokens",
    )
    context_truncated = _require_bool(
        row.get("context_truncated"),
        f"{label}/context_truncated",
    )
    _require_bool(row.get("refinement_applied"), f"{label}/refinement_applied")
    length_capped = _require_bool(
        row.get("generation_length_capped"),
        f"{label}/generation_length_capped",
    )

    if min(full_tokens, used_tokens, generated_tokens, output_limit) < 0:
        raise ValueError(f"Negative token count in {label}")
    if used_tokens > full_tokens:
        raise ValueError(f"Used prompt exceeds full prompt in {label}")
    if used_tokens > expected_input_limit:
        raise ValueError(f"Used prompt exceeds configured input limit in {label}")
    if output_limit != expected_output_limit:
        raise ValueError(
            f"Unexpected output limit in {label}: {output_limit} != {expected_output_limit}"
        )
    if generated_tokens > output_limit:
        raise ValueError(f"Generated tokens exceed configured output limit in {label}")
    if context_truncated and full_tokens <= used_tokens:
        raise ValueError(f"Truncation flag is inconsistent with prompt lengths in {label}")
    if not context_truncated and full_tokens != used_tokens:
        raise ValueError(f"Prompt lengths imply unreported truncation in {label}")
    if length_capped and generated_tokens != output_limit:
        raise ValueError(f"Length-cap flag is inconsistent with generated length in {label}")


def apply_global_holm_family(
    generations: dict[str, dict[str, Any]],
    raw_p_by_model: dict[str, dict[str, float]],
) -> dict[str, Any]:
    if set(generations) != set(raw_p_by_model):
        raise ValueError("Generation results and raw-p mappings have different model labels")
    expected_per_model = len(DATASETS) * len(COMPARATORS)
    for model_label, model_values in raw_p_by_model.items():
        if len(model_values) != expected_per_model:
            raise ValueError(
                f"Expected {expected_per_model} contrasts for {model_label}; "
                f"found {len(model_values)}"
            )
    combined = {
        f"{model_label}::{contrast}": p_value
        for model_label, model_values in raw_p_by_model.items()
        for contrast, p_value in model_values.items()
    }
    adjusted = holm_adjust(combined)
    family_size = len(combined)
    for key, adjusted_value in adjusted.items():
        model_label, dataset, comparator = key.split("::", 2)
        generations[model_label]["paired_f1_against_recursive_254"]["datasets"][dataset][
            comparator
        ]["randomization_p_holm_global"] = rounded(adjusted_value)
    generator_labels = list(generations)
    for generation in generations.values():
        paired_metadata = generation["paired_f1_against_recursive_254"]
        paired_metadata["primary_holm_family_size"] = family_size
        paired_metadata["primary_holm_family"] = (
            f"all {family_size} post-hoc chunker contrasts across "
            f"{len(generator_labels)} generator(s) and three datasets"
        )
    return {
        "method": "Holm step-down adjustment",
        "family_size": family_size,
        "generator_labels": generator_labels,
        "definition": (
            f"all {family_size} reported post-hoc recursive-versus-comparator "
            "contrasts across the included generators and datasets"
        ),
    }


def generation_prediction_path(root: Path, dataset: str, system: str) -> Path:
    return root / dataset / f"{system}_predictions.json"


def load_aligned_generation_predictions(
    root: Path,
    label: str,
    dataset: str,
) -> dict[str, list[dict[str, Any]]]:
    rows_by_system: dict[str, list[dict[str, Any]]] = {}
    reference_signatures: list[tuple[Any, ...]] | None = None
    for system in GENERATION_SYSTEMS:
        rows = read_json(generation_prediction_path(root, dataset, system))
        if not isinstance(rows, list):
            raise TypeError(f"Expected a prediction list for {label}/{dataset}/{system}")
        ids = [str(row["example_id"]) for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate example IDs in {dataset}/{system}")
        signatures = [
            (
                str(row["example_id"]),
                str(row["question"]),
                tuple(str(value) for value in row["gold_answers"]),
            )
            for row in rows
        ]
        if reference_signatures is None:
            reference_signatures = signatures
        elif signatures != reference_signatures:
            raise ValueError(f"{label} questions/golds are not aligned for {dataset}/{system}")
        rows_by_system[system] = rows
    return rows_by_system


def retrieval_analysis() -> dict[str, Any]:
    source_rows: dict[str, list[dict[str, Any]]] = {}
    configs: dict[str, dict[str, Any]] = {}
    source_hashes: dict[str, str] = {}
    for embedder, root in RETRIEVAL_ROOTS.items():
        configs[embedder] = validate_completed_root(root, RETRIEVAL_CONFIGS[embedder])
        rows = read_json(root / "all_results.json")
        if not isinstance(rows, list):
            raise TypeError(f"Expected a list in {root / 'all_results.json'}")
        source_rows[embedder] = rows
        source_hashes[embedder] = manifest_source_hash(root)
    if len(set(source_hashes.values())) != 1:
        raise ValueError(f"Retrieval runs used different source trees: {source_hashes}")

    cells: list[dict[str, Any]] = []
    best: list[dict[str, Any]] = []
    best_answer_visibility: list[dict[str, Any]] = []
    bm25_by_embedder: dict[str, dict[tuple[str, str], tuple[tuple[float, ...], tuple[float, ...]]]] = {}
    bm25_predictions_by_embedder: dict[
        str,
        dict[tuple[str, str, int], list[tuple[Any, ...]]],
    ] = {}
    question_signatures: dict[tuple[str, int], list[tuple[Any, ...]]] = {}

    for embedder, rows in source_rows.items():
        expected_sizes = {
            str(spec["name"]): int(spec["max_examples"])
            for spec in configs[embedder]["datasets"]
        }
        index: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        for row in rows:
            key = (str(row["dataset"]), str(row["retriever"]), str(row["chunker"]))
            index.setdefault(key, []).append(row)

        expected_keys = {
            (dataset, retriever, chunker)
            for dataset in DATASETS
            for retriever in RETRIEVERS
            for chunker in CHUNKERS
        }
        if set(index) != expected_keys:
            missing = sorted(expected_keys - set(index))
            extra = sorted(set(index) - expected_keys)
            raise ValueError(f"Unexpected retrieval cells for {embedder}; missing={missing}, extra={extra}")

        bm25_by_embedder[embedder] = {}
        bm25_predictions_by_embedder[embedder] = {}
        for dataset in DATASETS:
            for retriever in RETRIEVERS:
                candidates: list[dict[str, Any]] = []
                for chunker in CHUNKERS:
                    group = sorted(index[(dataset, retriever, chunker)], key=lambda row: int(row["seed"]))
                    seeds = [int(row["seed"]) for row in group]
                    if seeds != [13, 21, 34]:
                        raise ValueError(
                            f"Expected seeds 13, 21, 34 for {embedder}/{dataset}/{retriever}/{chunker}; "
                            f"found {seeds}"
                        )
                    values: list[float] = []
                    answer_visibility_values: list[float] = []
                    answer_visibility_ns: list[int] = []
                    for row in group:
                        seed = int(row["seed"])
                        prediction_path = (
                            RETRIEVAL_ROOTS[embedder]
                            / f"seed_{seed}"
                            / dataset
                            / f"{retriever}__{chunker}_predictions.json"
                        )
                        predictions = sorted(
                            read_json(prediction_path),
                            key=lambda item: str(item["example_id"]),
                        )
                        if len(predictions) != expected_sizes[dataset]:
                            raise ValueError(
                                f"Unexpected prediction count in {prediction_path}: {len(predictions)}"
                            )
                        prediction_ids = [str(item["example_id"]) for item in predictions]
                        if len(prediction_ids) != len(set(prediction_ids)):
                            raise ValueError(f"Duplicate example IDs in {prediction_path}")
                        signatures = [
                            (
                                str(item["example_id"]),
                                str(item["question"]),
                                tuple(str(value) for value in item["gold_answers"]),
                            )
                            for item in predictions
                        ]
                        signature_key = (dataset, seed)
                        previous = question_signatures.setdefault(signature_key, signatures)
                        if signatures != previous:
                            raise ValueError(f"Question/gold mismatch in {prediction_path}")
                        if retriever == "bm25":
                            bm25_predictions_by_embedder[embedder][
                                (dataset, chunker, seed)
                            ] = [
                                (
                                    str(item["example_id"]),
                                    tuple(str(value) for value in item["retrieved_chunk_ids"]),
                                    float(item["all_supporting_docs_found"]),
                                    float(item["answer_string_visible_at_k"]),
                                )
                                for item in predictions
                            ]

                        all_hit = mean(float(item["all_supporting_docs_found"]) for item in predictions)
                        raw_answer_visible = mean(
                            float(item["answer_string_visible_at_k"]) for item in predictions
                        )
                        applicable_visibility = applicable_answer_visibility_values(dataset, predictions)
                        if not applicable_visibility:
                            raise ValueError(f"No applicable AnsVis examples in {prediction_path}")
                        answer_visible = mean(applicable_visibility)
                        assert_close(
                            all_hit,
                            float(row["all_supporting_docs_found"]),
                            f"{embedder}/{dataset}/{retriever}/{chunker}/seed_{seed}/AllHit",
                        )
                        assert_close(
                            raw_answer_visible,
                            float(row["answer_string_visible_at_k"]),
                            f"{embedder}/{dataset}/{retriever}/{chunker}/seed_{seed}/raw AnsVis",
                        )
                        values.append(all_hit)
                        answer_visibility_values.append(answer_visible)
                        answer_visibility_ns.append(len(applicable_visibility))
                    cell = {
                        "embedder": embedder,
                        "dataset": dataset,
                        "retriever": retriever,
                        "chunker": chunker,
                        "seed_values": [rounded(value) for value in values],
                        "mean": rounded(mean(values)),
                        "sample_std": rounded(stdev(values)),
                        "answer_visibility_seed_values": [
                            rounded(value) for value in answer_visibility_values
                        ],
                        "answer_visibility_applicable_n_seed_values": answer_visibility_ns,
                        "answer_visibility_mean": rounded(mean(answer_visibility_values)),
                        "answer_visibility_sample_std": rounded(stdev(answer_visibility_values)),
                    }
                    cells.append(cell)
                    candidates.append(cell)
                    if retriever == "bm25":
                        bm25_by_embedder[embedder][(dataset, chunker)] = (
                            tuple(values),
                            tuple(answer_visibility_values),
                        )

                if retriever in {"dense", "hybrid"}:
                    best_mean = max(float(cell["mean"]) for cell in candidates)
                    winners = [
                        str(cell["chunker"])
                        for cell in candidates
                        if math.isclose(float(cell["mean"]), best_mean, rel_tol=0.0, abs_tol=1e-12)
                    ]
                    winner_details = [
                        cell for cell in candidates if str(cell["chunker"]) in winners
                    ]
                    exemplar = winner_details[0]
                    best.append(
                        {
                            "embedder": embedder,
                            "dataset": dataset,
                            "retriever": retriever,
                            "chunkers": winners,
                            "mean": exemplar["mean"],
                            "sample_std": exemplar["sample_std"],
                            "winner_details": winner_details,
                        }
                    )
                    best_visibility_mean = max(
                        float(cell["answer_visibility_mean"]) for cell in candidates
                    )
                    visibility_winners = [
                        str(cell["chunker"])
                        for cell in candidates
                        if math.isclose(
                            float(cell["answer_visibility_mean"]),
                            best_visibility_mean,
                            rel_tol=0.0,
                            abs_tol=1e-12,
                        )
                    ]
                    visibility_exemplar = next(
                        cell for cell in candidates if cell["chunker"] == visibility_winners[0]
                    )
                    visibility_winner_details = [
                        cell for cell in candidates if str(cell["chunker"]) in visibility_winners
                    ]
                    best_answer_visibility.append(
                        {
                            "embedder": embedder,
                            "dataset": dataset,
                            "retriever": retriever,
                            "chunkers": visibility_winners,
                            "mean": visibility_exemplar["answer_visibility_mean"],
                            "sample_std": visibility_exemplar["answer_visibility_sample_std"],
                            "winner_details": visibility_winner_details,
                        }
                    )

    first, second = RETRIEVAL_ROOTS
    bm25_identical = bm25_by_embedder[first] == bm25_by_embedder[second]
    bm25_predictions_identical = (
        bm25_predictions_by_embedder[first] == bm25_predictions_by_embedder[second]
    )
    if not bm25_identical or not bm25_predictions_identical:
        raise ValueError("BM25 results differ across embedding-model configurations")

    bm25_cells = [
        cell
        for cell in cells
        if cell["embedder"] == first and cell["retriever"] == "bm25"
    ]
    dense_hybrid_cells = [cell for cell in cells if cell["retriever"] != "bm25"]
    return {
        "selection_metric": "all_supporting_docs_found_at_4",
        "secondary_metric": (
            "normalized_gold_answer_token_sequence_visible_in_any_retrieved_chunk_at_4; "
            "HotpotQA yes/no examples excluded"
        ),
        "answer_visibility_exclusion": (
            "HotpotQA examples whose normalized reference answer is yes or no are excluded "
            "because literal label presence is not an evidence diagnostic"
        ),
        "seeds": [int(seed) for seed in configs[first]["seeds"]],
        "source_tree_sha256": source_hashes[first],
        "summary_statistic": (
            "mean and sample standard deviation across configured data-sampling seeds"
        ),
        "dense_and_hybrid_cells": dense_hybrid_cells,
        "best_dense_and_hybrid": best,
        "best_answer_visibility_dense_and_hybrid": best_answer_visibility,
        "bm25": {
            "scoring_uses_embeddings": False,
            "shared_chunking_tokenizer": configs[first]["chunking_tokenizer"],
            "identical_across_embedding_configurations": (
                bm25_identical and bm25_predictions_identical
            ),
            "cells": bm25_cells,
        },
    }


def generation_analysis(
    root: Path,
    config_path: Path,
    label: str,
    rng_offset: int = 0,
) -> tuple[dict[str, Any], dict[str, float]]:
    config = validate_completed_root(root, config_path)
    summary_rows = read_json(root / "all_results.json")
    if not isinstance(summary_rows, list):
        raise TypeError(f"Expected a list in {root / 'all_results.json'}")
    summary_index = unique_index(
        summary_rows,
        lambda row: (str(row["dataset"]), str(row["system"])),
        label=f"{label} summary cell",
    )
    expected = {(dataset, system) for dataset in DATASETS for system in GENERATION_SYSTEMS}
    if set(summary_index) != expected:
        missing = sorted(expected - set(summary_index))
        extra = sorted(set(summary_index) - expected)
        raise ValueError(f"Unexpected {label} summary cells; missing={missing}, extra={extra}")

    marginal: dict[str, dict[str, Any]] = {}
    paired: dict[str, dict[str, Any]] = {}
    prompt_audit: dict[str, dict[str, Any]] = {}
    model_raw_p: dict[str, float] = {}
    expected_sizes = {
        str(spec["name"]): int(spec["max_examples"])
        for spec in config["datasets"]
    }
    dataset_specs = {str(spec["name"]): spec for spec in config["datasets"]}
    configured_seeds = [int(seed) for seed in config.get("seeds", [config.get("seed", 42)])]
    if len(configured_seeds) != 1:
        raise ValueError(f"Generation analysis expects one configured seed for {label}")
    expected_seed = configured_seeds[0]
    input_limit = int(config["generation_max_input_tokens"])
    bootstrap_draws = int(config["bootstrap_samples"])
    confidence = float(config["confidence_level"])

    for dataset_index, dataset in enumerate(DATASETS):
        rows_by_system = load_aligned_generation_predictions(root, label, dataset)
        marginal[dataset] = {}
        prompt_audit[dataset] = {}

        for system in GENERATION_SYSTEMS:
            row = summary_index[(dataset, system)]
            prediction_rows = rows_by_system[system]
            if len(prediction_rows) != expected_sizes[dataset]:
                raise ValueError(f"Unexpected {label} prediction count for {dataset}/{system}")
            if int(row["seed"]) != expected_seed:
                raise ValueError(f"Unexpected {label} summary seed for {dataset}/{system}")
            if int(row["num_examples"]) != len(prediction_rows):
                raise ValueError(f"Unexpected {label} summary n for {dataset}/{system}")
            if any(int(item["seed"]) != expected_seed for item in prediction_rows):
                raise ValueError(f"Unexpected {label} prediction seed for {dataset}/{system}")
            exact_match_values, f1_values = recompute_answer_vectors(
                prediction_rows,
                label=f"{label}/{dataset}/{system}",
            )
            assert_close(
                mean(f1_values),
                float(row["f1"]),
                f"{label}/{dataset}/{system}/F1",
            )
            assert_close(
                mean(exact_match_values),
                float(row["exact_match"]),
                f"{label}/{dataset}/{system}/EM",
            )
            f1_ci_low, f1_ci_high = bootstrap_confidence_interval(
                f1_values,
                num_bootstrap_samples=bootstrap_draws,
                confidence=confidence,
                seed=expected_seed,
            )
            em_ci_low, em_ci_high = bootstrap_confidence_interval(
                exact_match_values,
                num_bootstrap_samples=bootstrap_draws,
                confidence=confidence,
                seed=expected_seed,
            )
            for metric_name, actual, expected_value in (
                ("F1 CI low", float(row["f1_ci_low"]), f1_ci_low),
                ("F1 CI high", float(row["f1_ci_high"]), f1_ci_high),
                ("EM CI low", float(row["exact_match_ci_low"]), em_ci_low),
                ("EM CI high", float(row["exact_match_ci_high"]), em_ci_high),
            ):
                assert_close(actual, expected_value, f"{label}/{dataset}/{system}/{metric_name}")
            if system != "parametric_only":
                assert_close(
                    mean(float(item["all_supporting_docs_found"]) for item in prediction_rows),
                    float(row["all_supporting_docs_found"]),
                    f"{label}/{dataset}/{system}/AllHit",
                )
                assert_close(
                    mean(float(item["answer_string_visible_at_k"]) for item in prediction_rows),
                    float(row["answer_string_visible_at_k"]),
                    f"{label}/{dataset}/{system}/raw AnsVis",
                )
            applicable_visibility = (
                []
                if system == "parametric_only"
                else applicable_answer_visibility_values(dataset, prediction_rows)
            )
            if system != "parametric_only" and not applicable_visibility:
                raise ValueError(f"No applicable AnsVis examples for {label}/{dataset}/{system}")
            marginal[dataset][system] = {
                "n": len(prediction_rows),
                "f1": rounded(mean(f1_values)),
                "f1_ci_low": rounded(f1_ci_low),
                "f1_ci_high": rounded(f1_ci_high),
                "exact_match": rounded(mean(exact_match_values)),
                "exact_match_ci_low": rounded(em_ci_low),
                "exact_match_ci_high": rounded(em_ci_high),
                "all_supporting_docs_found_at_4": (
                    None
                    if system == "parametric_only"
                    else rounded(row["all_supporting_docs_found"])
                ),
                "answer_string_visible_at_4": (
                    None
                    if system == "parametric_only"
                    else rounded(mean(applicable_visibility))
                ),
                "answer_string_visible_at_4_applicable_n": (
                    None if system == "parametric_only" else len(applicable_visibility)
                ),
            }

            expected_output_limit = int(
                dataset_specs[dataset].get("max_new_tokens", config["max_new_tokens"])
            )
            for item in prediction_rows:
                validate_generation_trace(
                    item,
                    expected_input_limit=input_limit,
                    expected_output_limit=expected_output_limit,
                    label=f"{label}/{dataset}/{system}/{item['example_id']}",
                )
            full_tokens = [int(item["full_prompt_tokens"]) for item in prediction_rows]
            used_tokens = [int(item["used_prompt_tokens"]) for item in prediction_rows]
            generated_tokens = [int(item["generated_tokens"]) for item in prediction_rows]
            truncated_count = sum(bool(item["context_truncated"]) for item in prediction_rows)
            refinement_count = sum(bool(item["refinement_applied"]) for item in prediction_rows)
            length_capped_count = sum(
                bool(item["generation_length_capped"]) for item in prediction_rows
            )
            prompt_audit[dataset][system] = {
                "n": len(prediction_rows),
                "context_truncated_count": truncated_count,
                "context_truncated_rate": rounded(truncated_count / len(prediction_rows)),
                "full_prompt_tokens_mean": rounded(mean(full_tokens), 4),
                "full_prompt_tokens_min": min(full_tokens),
                "full_prompt_tokens_max": max(full_tokens),
                "used_prompt_tokens_mean": rounded(mean(used_tokens), 4),
                "tokens_removed_total": sum(full - used for full, used in zip(full_tokens, used_tokens)),
                "refinement_applied_count": refinement_count,
                "generated_tokens_mean": rounded(mean(generated_tokens), 4),
                "generated_tokens_max": max(generated_tokens),
                "generation_max_new_tokens": sorted(
                    {int(item["generation_max_new_tokens"]) for item in prediction_rows}
                ),
                "generation_length_capped_count": length_capped_count,
                "generation_length_capped_rate": rounded(
                    length_capped_count / len(prediction_rows)
                ),
            }

        reference = rows_by_system["hybrid__recursive_254"]
        comparison_results: dict[str, Any] = {}
        raw_p: dict[str, float] = {}
        for comparator_index, comparator in enumerate(COMPARATORS):
            candidate = rows_by_system[f"hybrid__{comparator}"]
            differences = [
                float(reference_row["f1"]) - float(candidate_row["f1"])
                for reference_row, candidate_row in zip(reference, candidate)
            ]
            ci_rng = random.Random(
                RANDOM_SEED + rng_offset + 100_000 + dataset_index * 10_000 + comparator_index * 100
            )
            test_rng = random.Random(
                RANDOM_SEED + rng_offset + 200_000 + dataset_index * 10_000 + comparator_index * 100
            )
            low, high = paired_bootstrap_ci(differences, draws=BOOTSTRAP_DRAWS, rng=ci_rng)
            p_value = paired_randomization_p(
                differences, draws=RANDOMIZATION_DRAWS, rng=test_rng
            )
            comparison_results[comparator] = {
                "n": len(differences),
                "mean_f1_difference": rounded(mean(differences)),
                "ci_low": rounded(low),
                "ci_high": rounded(high),
                "randomization_p_raw": rounded(p_value),
            }
            raw_p[comparator] = p_value
            model_raw_p[f"{dataset}::{comparator}"] = p_value

        adjusted = holm_adjust(raw_p)
        for comparator, adjusted_value in adjusted.items():
            comparison_results[comparator]["randomization_p_holm_within_dataset"] = rounded(
                adjusted_value
            )
        paired[dataset] = comparison_results

    result = {
        "label": label,
        "model": config["generator_model"],
        "model_revision": config.get("generator_model_revision"),
        "source_tree_sha256": manifest_source_hash(root),
        "retriever": "hybrid",
        "generation_max_input_tokens": input_limit,
        "dataset_max_new_tokens": {
            dataset: int(dataset_specs[dataset].get("max_new_tokens", config["max_new_tokens"]))
            for dataset in DATASETS
        },
        "dataset_sizes": expected_sizes,
        "confidence_level": confidence,
        "marginal_question_bootstrap_draws": bootstrap_draws,
        "marginal": marginal,
        "paired_f1_against_recursive_254": {
            "sampling_unit": "question",
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "randomization_draws": RANDOMIZATION_DRAWS,
            "seed": RANDOM_SEED,
            "within_dataset_holm_values_reported_for_transparency": True,
            "datasets": paired,
        },
        "prompt_audit": prompt_audit,
    }
    return result, model_raw_p


def validate_cross_model_alignment(
    active_specs: list[tuple[str, Path, Path]],
) -> None:
    if len(active_specs) <= 1:
        return
    reference_label: str | None = None
    reference_signatures: dict[tuple[str, str], list[tuple[Any, ...]]] | None = None
    reference_source_hash: str | None = None
    for label, root, _ in active_specs:
        source_hash = manifest_source_hash(root)
        if reference_source_hash is None:
            reference_source_hash = source_hash
        elif source_hash != reference_source_hash:
            raise ValueError(
                f"Generation source hashes differ: {reference_label}={reference_source_hash}, "
                f"{label}={source_hash}"
            )
        signatures: dict[tuple[str, str], list[tuple[Any, ...]]] = {}
        for dataset in DATASETS:
            rows_by_system = load_aligned_generation_predictions(root, label, dataset)
            for system, rows in rows_by_system.items():
                signatures[(dataset, system)] = [
                    (
                        str(row["example_id"]),
                        str(row["question"]),
                        tuple(str(value) for value in row["gold_answers"]),
                        tuple(str(value) for value in row["retrieved_chunk_ids"]),
                    )
                    for row in rows
                ]
        if reference_signatures is None:
            reference_label = label
            reference_signatures = signatures
        elif signatures != reference_signatures:
            raise ValueError(
                f"Generation questions/golds/retrieved chunks differ between "
                f"{reference_label} and {label}"
            )


def best_index(
    best_rows: list[dict[str, Any]], embedder: str, dataset: str, retriever: str
) -> dict[str, Any]:
    matches = [
        row
        for row in best_rows
        if row["embedder"] == embedder
        and row["dataset"] == dataset
        and row["retriever"] == retriever
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one best row for {embedder}/{dataset}/{retriever}")
    return matches[0]


def chunker_markdown(chunkers: list[str]) -> str:
    return "/".join(f"`{chunker}`" for chunker in chunkers)


def chunker_tex(chunkers: list[str]) -> str:
    escaped = [chunker.replace("_", "\\_") for chunker in chunkers]
    return "/".join(f"\\texttt{{{chunker}}}" for chunker in escaped)


def winner_score_strings(
    row: dict[str, Any],
    *,
    answer_visibility: bool,
    tex: bool,
) -> list[str]:
    mean_key = "answer_visibility_mean" if answer_visibility else "mean"
    std_key = "answer_visibility_sample_std" if answer_visibility else "sample_std"
    separator = " $\\pm$ " if tex else " ± "
    return [
        f"{100.0 * float(detail[mean_key]):.1f}{separator}{100.0 * float(detail[std_key]):.1f}"
        for detail in row["winner_details"]
    ]


def winner_scores(
    row: dict[str, Any],
    *,
    answer_visibility: bool,
    tex: bool,
) -> str:
    return "/".join(
        winner_score_strings(
            row,
            answer_visibility=answer_visibility,
            tex=tex,
        )
    )


def render_retrieval_tex(retrieval: dict[str, Any]) -> str:
    lines = [
        "% Generated by scripts/analyze_reviewer_robustness.py; do not edit by hand.",
        "",
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.4pt}",
        r"\begin{tabular}{llllrlr}",
        r"\toprule",
        r"Dataset & Embedder & Retriever & Best AllHit & Score & Best AnsVis & Score \\",
        r"\midrule",
    ]
    allhit_rows = retrieval["best_dense_and_hybrid"]
    visibility_rows = retrieval["best_answer_visibility_dense_and_hybrid"]
    for dataset in DATASETS:
        for embedder in RETRIEVAL_ROOTS:
            for retriever in ("dense", "hybrid"):
                allhit = best_index(allhit_rows, embedder, dataset, retriever)
                visibility = best_index(visibility_rows, embedder, dataset, retriever)
                lines.append(
                    f"{DATASET_LABELS[dataset]} & {embedder} & {retriever} & "
                    f"{chunker_tex(allhit['chunkers'])} & "
                    f"{winner_scores(allhit, answer_visibility=False, tex=True)} & "
                    f"{chunker_tex(visibility['chunkers'])} & "
                    f"{winner_scores(visibility, answer_visibility=True, tex=True)} \\\\"
                )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\caption{Best observed chunker under document AllHit@4 and normalized "
                r"answer-token-sequence visibility (AnsVis@4) for each dense and hybrid setting. Scores "
                "are mean $\\pm$ sample SD (percentage points) across seeds "
                + ", ".join(str(seed) for seed in retrieval["seeds"])
                + r"; ties are retained and list one score per tied chunker. HotpotQA yes/no "
                r"questions are excluded from AnsVis because literal label presence is not an "
                r"evidence diagnostic. AnsVis remains conservative for paraphrases and long answers. BM25 "
                r"uses no embeddings and is reported once in the supplemental analysis.}"
            ),
            r"\label{tab:robustness-retrieval}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def generation_cell(row: dict[str, Any], *, bold: bool) -> str:
    rendered = (
        f"{100.0 * float(row['f1']):.1f} "
        f"[{100.0 * float(row['f1_ci_low']):.1f}, {100.0 * float(row['f1_ci_high']):.1f}]"
    )
    return f"\\textbf{{{rendered}}}" if bold else rendered


def render_generation_tex(generations: dict[str, dict[str, Any]]) -> str:
    if not generations:
        raise ValueError("At least one generation result is required")
    first_generation = next(iter(generations.values()))
    bootstrap_draws = int(first_generation["marginal_question_bootstrap_draws"])
    confidence = float(first_generation["confidence_level"])
    dataset_sizes = {
        dataset: int(first_generation["dataset_sizes"][dataset]) for dataset in DATASETS
    }
    for generation in generations.values():
        if int(generation["marginal_question_bootstrap_draws"]) != bootstrap_draws:
            raise ValueError("Generation runs use different marginal bootstrap draw counts")
        if not math.isclose(
            float(generation["confidence_level"]),
            confidence,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("Generation runs use different confidence levels")
        if {
            dataset: int(generation["dataset_sizes"][dataset]) for dataset in DATASETS
        } != dataset_sizes:
            raise ValueError("Generation runs use different dataset sizes")
    confidence_percent = 100.0 * confidence
    sample_size_text = "/".join(str(dataset_sizes[dataset]) for dataset in DATASETS)
    columns = (
        ("parametric_only", "No context"),
        ("hybrid__fixed_128", r"\texttt{fixed\_128}"),
        ("hybrid__fixed_254", r"\texttt{fixed\_254}"),
        ("hybrid__recursive_254", r"\texttt{recursive\_254}"),
        ("hybrid__sentence_254", r"\texttt{sentence\_254}"),
    )
    lines = [
        "% Generated by scripts/analyze_reviewer_robustness.py; do not edit by hand.",
        "",
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.0pt}",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        "Dataset & Model & " + " & ".join(label for _, label in columns) + r" \\",
        r"\midrule",
    ]
    for model_label, generation in generations.items():
        for dataset in DATASETS:
            rows = generation["marginal"][dataset]
            rag_systems = [system for system, _ in columns if system != "parametric_only"]
            best_rag = max(float(rows[system]["f1"]) for system in rag_systems)
            rendered_cells = []
            for system, _ in columns:
                is_best = system != "parametric_only" and math.isclose(
                    float(rows[system]["f1"]), best_rag, rel_tol=0.0, abs_tol=1e-12
                )
                rendered_cells.append(generation_cell(rows[system], bold=is_best))
            lines.append(
                f"{DATASET_LABELS[dataset]} & {model_label} & "
                + " & ".join(rendered_cells)
                + r" \\"
            )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\caption{Generator F1 with no retrieved context and with BGE-small hybrid "
                f"retrieval. Cells show F1 [marginal {bootstrap_draws:,}-draw question-bootstrap "
                f"{confidence_percent:g}\\% CI] in percentage points; "
                f"$n={sample_size_text}$ for SQuAD/HotpotQA/TechQA. "
                r"Bold marks the best observed RAG "
                r"cell within each dataset and does not by itself imply statistical significance.}"
            ),
            r"\label{tab:robustness-generation}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def render_markdown(analysis: dict[str, Any]) -> str:
    retrieval = analysis["retrieval"]
    generations = analysis["generation_models"]
    first_generation = next(iter(generations.values()))
    marginal_draws = int(first_generation["marginal_question_bootstrap_draws"])
    confidence_percent = 100.0 * float(first_generation["confidence_level"])
    paired_draws = int(first_generation["paired_f1_against_recursive_254"]["bootstrap_draws"])
    retrieval_seed_text = ", ".join(str(seed) for seed in retrieval["seeds"])
    holm_family_size = int(analysis["multiple_testing"]["family_size"])
    lines = [
        "# Reviewer-driven robustness analysis",
        "",
        "This file is generated from archived prediction and summary artifacts. Retrieval values "
        f"are document-level AllHit@4 means ± sample SD over seeds {retrieval_seed_text}. Generator "
        f"intervals are recomputed {marginal_draws:,}-draw question-level percentile-bootstrap "
        f"{confidence_percent:g}% intervals; paired contrasts use {paired_draws:,} bootstrap draws.",
        "",
        "## Retrieval robustness",
        "",
        "AnsVis@4 excludes HotpotQA questions whose normalized reference is `yes` or `no`, "
        "because literal label presence is not evidence that the supporting passage was retrieved. "
        "Applicable per-seed sample sizes are retained in the JSON output.",
        "",
        "| Dataset | Embedder | Retriever | Best AllHit chunker | AllHit@4 (%) | Best AnsVis chunker | AnsVis@4 (%) |",
        "|---|---|---|---|---:|---|---:|",
    ]
    allhit_rows = retrieval["best_dense_and_hybrid"]
    visibility_rows = retrieval["best_answer_visibility_dense_and_hybrid"]
    for dataset in DATASETS:
        for embedder in RETRIEVAL_ROOTS:
            for retriever in ("dense", "hybrid"):
                allhit = best_index(allhit_rows, embedder, dataset, retriever)
                visibility = best_index(visibility_rows, embedder, dataset, retriever)
                lines.append(
                    f"| {DATASET_LABELS[dataset]} | {embedder} | {retriever} | "
                    f"{chunker_markdown(allhit['chunkers'])} | "
                    f"{winner_scores(allhit, answer_visibility=False, tex=False)} | "
                    f"{chunker_markdown(visibility['chunkers'])} | "
                    f"{winner_scores(visibility, answer_visibility=True, tex=False)} |"
                )

    lines.extend(
        [
            "",
            "BM25 scoring uses no embeddings, and both configurations use the same pinned MiniLM "
            "chunk tokenizer. They produced exactly identical per-seed results, so the best BM25 "
            "cell for each dataset is shown once below.",
            "",
            "| Dataset | Best AllHit chunker | AllHit@4 (%) | Best AnsVis chunker | AnsVis@4 (%) |",
            "|---|---|---:|---|---:|",
        ]
    )
    bm25_cells = retrieval["bm25"]["cells"]
    for dataset in DATASETS:
        candidates = [cell for cell in bm25_cells if cell["dataset"] == dataset]
        best_mean = max(float(cell["mean"]) for cell in candidates)
        winners = [cell for cell in candidates if math.isclose(float(cell["mean"]), best_mean, abs_tol=1e-12)]
        chunkers = [str(cell["chunker"]) for cell in winners]
        best_visibility = max(float(cell["answer_visibility_mean"]) for cell in candidates)
        visibility_winner_rows = [
            cell
            for cell in candidates
            if math.isclose(
                float(cell["answer_visibility_mean"]),
                best_visibility,
                abs_tol=1e-12,
            )
        ]
        visibility_winners = [str(cell["chunker"]) for cell in visibility_winner_rows]
        lines.append(
            f"| {DATASET_LABELS[dataset]} | {chunker_markdown(chunkers)} | "
            + "/".join(
                f"{100.0 * float(cell['mean']):.1f} ± {100.0 * float(cell['sample_std']):.1f}"
                for cell in winners
            )
            + f" | {chunker_markdown(visibility_winners)} | "
            + "/".join(
                f"{100.0 * float(cell['answer_visibility_mean']):.1f} ± "
                f"{100.0 * float(cell['answer_visibility_sample_std']):.1f}"
                for cell in visibility_winner_rows
            )
            + " |"
        )

    for model_label, generation in generations.items():
        lines.extend(
            [
                "",
                f"## {model_label} generation robustness",
                "",
                f"Cells show F1 percentage points and marginal "
                f"{int(generation['marginal_question_bootstrap_draws']):,}-draw "
                f"{100.0 * float(generation['confidence_level']):g}% confidence intervals.",
                "",
                "| Dataset | No context | `fixed_128` | `fixed_254` | `recursive_254` | `sentence_254` |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for dataset in DATASETS:
            cells = []
            for system in GENERATION_SYSTEMS:
                row = generation["marginal"][dataset][system]
                cells.append(
                    f"{100.0 * row['f1']:.1f} [{100.0 * row['f1_ci_low']:.1f}, "
                    f"{100.0 * row['f1_ci_high']:.1f}]"
                )
            lines.append(f"| {DATASET_LABELS[dataset]} | " + " | ".join(cells) + " |")

        lines.extend(
            [
                "",
                "### Paired F1 contrasts",
                "",
                "Differences are `recursive_254` minus the comparator. CIs use 20,000 paired "
                "bootstrap draws; two-sided sign-flip tests use 100,000 draws. The primary Holm "
                f"correction covers all {holm_family_size} contrasts across the included generators.",
                "",
                "| Dataset | Comparator | ΔF1 (pp) | 95% CI (pp) | Raw p | Global Holm p |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        paired = generation["paired_f1_against_recursive_254"]["datasets"]
        for dataset in DATASETS:
            for comparator in COMPARATORS:
                row = paired[dataset][comparator]
                lines.append(
                    f"| {DATASET_LABELS[dataset]} | `{comparator}` | "
                    f"{100.0 * row['mean_f1_difference']:+.2f} | "
                    f"[{100.0 * row['ci_low']:+.2f}, {100.0 * row['ci_high']:+.2f}] | "
                    f"{row['randomization_p_raw']:.5f} | "
                    f"{row['randomization_p_holm_global']:.5f} |"
                )

        lines.extend(
            [
                "",
                f"### {model_label} token-budget audit",
                "",
                "| Dataset | System | Input limit | Output limit | Context cut | Length capped | Mean full | Mean used | Mean generated | Max generated |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for dataset in DATASETS:
            for system in GENERATION_SYSTEMS:
                row = generation["prompt_audit"][dataset][system]
                system_label = system.removeprefix("hybrid__")
                lines.append(
                    f"| {DATASET_LABELS[dataset]} | `{system_label}` | "
                    f"{generation['generation_max_input_tokens']} | "
                    f"{row['generation_max_new_tokens'][0]} | "
                    f"{row['context_truncated_count']}/{row['n']} | "
                    f"{row['generation_length_capped_count']}/{row['n']} | "
                    f"{row['full_prompt_tokens_mean']:.1f} | "
                    f"{row['used_prompt_tokens_mean']:.1f} | "
                    f"{row['generated_tokens_mean']:.1f} | "
                    f"{row['generated_tokens_max']} |"
                )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    retrieval = retrieval_analysis()
    generation_models: dict[str, dict[str, Any]] = {}
    raw_p_by_model: dict[str, dict[str, float]] = {}
    active_specs: list[tuple[str, Path, Path]] = []
    for model_index, (label, root, config_path) in enumerate(GENERATION_SPECS):
        if label != "Qwen2.5-1.5B" and not (root / "run_manifest.json").is_file():
            continue
        generation, raw_p = generation_analysis(
            root,
            config_path,
            label,
            rng_offset=model_index * 1_000_000,
        )
        generation_models[label] = generation
        raw_p_by_model[label] = raw_p
        active_specs.append((label, root, config_path))
    validate_cross_model_alignment(active_specs)
    source_hashes = {
        "retrieval": str(retrieval["source_tree_sha256"]),
        **{
            model_label: str(generation["source_tree_sha256"])
            for model_label, generation in generation_models.items()
        },
    }
    if len(set(source_hashes.values())) != 1:
        raise ValueError(f"Experiment runs used different source trees: {source_hashes}")
    multiple_testing = apply_global_holm_family(generation_models, raw_p_by_model)
    analysis = {
        "analysis_version": 3,
        "sources": {
            "retrieval_minilm": str(RETRIEVAL_ROOTS["MiniLM"].relative_to(REPO_ROOT)),
            "retrieval_bge": str(RETRIEVAL_ROOTS["BGE-small"].relative_to(REPO_ROOT)),
            "generation_qwen": str(QWEN_ROOT.relative_to(REPO_ROOT)),
            **(
                {"generation_mistral": str(MISTRAL_ROOT.relative_to(REPO_ROOT))}
                if "Mistral-7B" in generation_models
                else {}
            ),
        },
        "experiment_source_tree_sha256": next(iter(source_hashes.values())),
        "multiple_testing": multiple_testing,
        "retrieval": retrieval,
        "generation": generation_models["Qwen2.5-1.5B"],
        "generation_models": generation_models,
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    RETRIEVAL_TEX.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_JSON.open("w", encoding="utf-8") as handle:
        json.dump(analysis, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    OUTPUT_MARKDOWN.write_text(render_markdown(analysis), encoding="utf-8")
    RETRIEVAL_TEX.write_text(render_retrieval_tex(retrieval), encoding="utf-8")
    GENERATION_TEX.write_text(render_generation_tex(generation_models), encoding="utf-8")

    print(f"Wrote {OUTPUT_JSON.relative_to(REPO_ROOT)}")
    print(f"Wrote {OUTPUT_MARKDOWN.relative_to(REPO_ROOT)}")
    print(f"Wrote {RETRIEVAL_TEX.relative_to(REPO_ROOT)}")
    print(f"Wrote {GENERATION_TEX.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
