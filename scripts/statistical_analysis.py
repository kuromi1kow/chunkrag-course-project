#!/usr/bin/env python3
"""Paired uncertainty analysis for the archived Mistral-v2 predictions.

The script never reruns the retriever or generator. It treats questions as the
sampling unit, reports percentile bootstrap intervals, and uses a paired
sign-flip randomization test for chunker comparisons. The randomization tests
measure uncertainty over the sampled questions, not run-to-run model variance.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from statistics import mean


DATASETS = ("squad_v2", "hotpot_qa")
SYSTEMS = (
    "parametric_only",
    "fixed_128",
    "fixed_256",
    "fixed_512",
    "recursive_256",
    "sentence_256",
    "semantic_256",
)
CHUNKER_COMPARATORS = (
    "fixed_128",
    "fixed_256",
    "fixed_512",
    "sentence_256",
    "semantic_256",
)
METRICS = ("exact_match", "f1", "recall_at_k", "precision_at_k")


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


def bootstrap_mean_ci(
    values: list[float], *, draws: int, rng: random.Random
) -> tuple[float, float]:
    n = len(values)
    samples = [mean(values[rng.randrange(n)] for _ in range(n)) for _ in range(draws)]
    return percentile(samples, 0.025), percentile(samples, 0.975)


def paired_bootstrap_ci(
    differences: list[float], *, draws: int, rng: random.Random
) -> tuple[float, float]:
    return bootstrap_mean_ci(differences, draws=draws, rng=rng)


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
    ordered = sorted(raw, key=raw.get)
    total = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, key in enumerate(ordered):
        candidate = min(1.0, (total - rank) * raw[key])
        running = max(running, candidate)
        adjusted[key] = running
    return adjusted


def load_predictions(root: Path, dataset: str, system: str) -> list[dict]:
    path = root / dataset / f"{system}_predictions.json"
    with path.open(encoding="utf-8") as handle:
        rows = json.load(handle)
    return sorted(rows, key=lambda row: row["example_id"])


def ensure_alignment(rows_by_system: dict[str, list[dict]]) -> list[str]:
    ids = [row["example_id"] for row in rows_by_system[SYSTEMS[0]]]
    for system, rows in rows_by_system.items():
        other = [row["example_id"] for row in rows]
        if other != ids:
            raise ValueError(f"Prediction rows are not aligned for {system}")
    return ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prediction_root", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--bootstrap-draws", type=int, default=20_000)
    parser.add_argument("--randomization-draws", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=8677)
    args = parser.parse_args()

    result: dict[str, object] = {
        "sampling_unit": "question",
        "bootstrap_draws": args.bootstrap_draws,
        "randomization_draws": args.randomization_draws,
        "seed": args.seed,
        "datasets": {},
    }

    for dataset_index, dataset in enumerate(DATASETS):
        rows_by_system = {system: load_predictions(args.prediction_root, dataset, system) for system in SYSTEMS}
        ids = ensure_alignment(rows_by_system)
        dataset_result: dict[str, object] = {
            "n": len(ids),
            "systems": {},
            "paired_against_recursive_256": {},
        }

        for system_index, system in enumerate(SYSTEMS):
            system_result: dict[str, object] = {}
            rows = rows_by_system[system]
            for metric_index, metric in enumerate(METRICS):
                values = [float(row[metric]) for row in rows]
                rng = random.Random(args.seed + dataset_index * 10_000 + system_index * 100 + metric_index)
                low, high = bootstrap_mean_ci(values, draws=args.bootstrap_draws, rng=rng)
                system_result[metric] = {"mean": mean(values), "ci_low": low, "ci_high": high}
            dataset_result["systems"][system] = system_result

        reference = rows_by_system["recursive_256"]
        raw_p_by_metric: dict[str, dict[str, float]] = {metric: {} for metric in ("exact_match", "f1")}
        for comparator_index, comparator in enumerate(CHUNKER_COMPARATORS):
            comparison: dict[str, object] = {}
            candidate = rows_by_system[comparator]
            for metric_index, metric in enumerate(("exact_match", "f1")):
                differences = [
                    float(ref_row[metric]) - float(candidate_row[metric])
                    for ref_row, candidate_row in zip(reference, candidate)
                ]
                rng_ci = random.Random(
                    args.seed + 100_000 + dataset_index * 10_000 + comparator_index * 100 + metric_index
                )
                rng_test = random.Random(
                    args.seed + 200_000 + dataset_index * 10_000 + comparator_index * 100 + metric_index
                )
                low, high = paired_bootstrap_ci(differences, draws=args.bootstrap_draws, rng=rng_ci)
                p_value = paired_randomization_p(
                    differences, draws=args.randomization_draws, rng=rng_test
                )
                comparison[metric] = {
                    "mean_difference": mean(differences),
                    "ci_low": low,
                    "ci_high": high,
                    "randomization_p_raw": p_value,
                }
                raw_p_by_metric[metric][comparator] = p_value
            dataset_result["paired_against_recursive_256"][comparator] = comparison

        for metric, raw_values in raw_p_by_metric.items():
            adjusted = holm_adjust(raw_values)
            for comparator, adjusted_value in adjusted.items():
                dataset_result["paired_against_recursive_256"][comparator][metric][
                    "randomization_p_holm"
                ] = adjusted_value

        if dataset == "squad_v2":
            fixed_hits = {
                row["example_id"]
                for row in rows_by_system["fixed_128"]
                if float(row["recall_at_k"]) == 1.0
            }
            semantic_hits = {
                row["example_id"]
                for row in rows_by_system["semantic_256"]
                if float(row["recall_at_k"]) == 1.0
            }
            all_ids = set(ids)
            dataset_result["fixed_128_vs_semantic_256_retrieval_overlap"] = {
                "both_hit": len(fixed_hits & semantic_hits),
                "fixed_only_hit": len(fixed_hits - semantic_hits),
                "semantic_only_hit": len(semantic_hits - fixed_hits),
                "both_miss": len(all_ids - (fixed_hits | semantic_hits)),
                "fixed_missed_example_ids": sorted(all_ids - fixed_hits),
                "semantic_missed_example_ids": sorted(all_ids - semantic_hits),
            }

        result["datasets"][dataset] = dataset_result

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)


if __name__ == "__main__":
    main()
