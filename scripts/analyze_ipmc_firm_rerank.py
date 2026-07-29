#!/usr/bin/env python3
"""Compare paired hybrid and hybrid-rerank retrieval artifacts for IP&MC FIRM."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any


DATASETS = ("squad_v2", "hotpot_qa", "techqa")
DATASET_LABELS = {
    "squad_v2": "SQuAD 2.0",
    "hotpot_qa": "HotpotQA",
    "techqa": "TechQA",
}
CHUNKERS = ("fixed_128", "fixed_254", "recursive_254", "sentence_254")
RETRIEVERS = ("hybrid", "hybrid_rerank")
SEEDS = (13, 21, 34)
METRICS = ("all_supporting_docs_found", "answer_string_visible_at_k")


def read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Required artifact is missing: {path}")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def sample_sd(values: list[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def prediction_path(
    root: Path,
    seed: int,
    dataset: str,
    retriever: str,
    chunker: str,
) -> Path:
    return (
        root
        / f"seed_{seed}"
        / dataset
        / f"{retriever}__{chunker}_predictions.json"
    )


def prediction_index(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_json(path)
    if not isinstance(rows, list):
        raise TypeError(f"Expected a prediction list: {path}")
    index = {str(row["example_id"]): row for row in rows}
    if len(index) != len(rows):
        raise ValueError(f"Duplicate example IDs: {path}")
    return index


def validate_summary_rows(root: Path) -> dict[tuple[str, str, str, int], dict[str, Any]]:
    rows = read_json(root / "all_results.json")
    if not isinstance(rows, list):
        raise TypeError("all_results.json must contain a list")
    expected = {
        (dataset, retriever, chunker, seed)
        for dataset in DATASETS
        for retriever in RETRIEVERS
        for chunker in CHUNKERS
        for seed in SEEDS
    }
    index = {
        (
            str(row["dataset"]),
            str(row["retriever"]),
            str(row["chunker"]),
            int(row["seed"]),
        ): row
        for row in rows
    }
    if len(index) != len(rows):
        raise ValueError("Duplicate summary cells")
    if set(index) != expected:
        raise ValueError(
            f"Unexpected summary matrix; missing={sorted(expected - set(index))}, "
            f"extra={sorted(set(index) - expected)}"
        )
    return index


def analyze(root: Path) -> dict[str, Any]:
    manifest = read_json(root / "run_manifest.json")
    if manifest.get("status") != "complete":
        raise ValueError(f"Run is not complete: {root}")
    summaries = validate_summary_rows(root)

    cells: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for chunker in CHUNKERS:
            metric_seed_values: dict[str, dict[str, list[float]]] = {
                metric: {retriever: [] for retriever in RETRIEVERS}
                for metric in METRICS
            }
            latency_values: dict[str, list[float]] = {
                retriever: [] for retriever in RETRIEVERS
            }
            flips = defaultdict(int)

            for seed in SEEDS:
                hybrid = prediction_index(
                    prediction_path(root, seed, dataset, "hybrid", chunker)
                )
                reranked = prediction_index(
                    prediction_path(root, seed, dataset, "hybrid_rerank", chunker)
                )
                if set(hybrid) != set(reranked):
                    raise ValueError(
                        f"Question mismatch for {dataset}/{chunker}/seed_{seed}"
                    )

                for metric in METRICS:
                    for retriever, rows in (
                        ("hybrid", hybrid),
                        ("hybrid_rerank", reranked),
                    ):
                        metric_seed_values[metric][retriever].append(
                            mean(float(row[metric]) for row in rows.values())
                        )

                for example_id in sorted(hybrid):
                    before = hybrid[example_id]
                    after = reranked[example_id]
                    for metric, label in (
                        ("all_supporting_docs_found", "allhit"),
                        ("answer_string_visible_at_k", "ansvis"),
                    ):
                        left = float(before[metric])
                        right = float(after[metric])
                        if left == 0.0 and right == 1.0:
                            flips[f"{label}_gain"] += 1
                        elif left == 1.0 and right == 0.0:
                            flips[f"{label}_loss"] += 1

                for retriever in RETRIEVERS:
                    summary = summaries[(dataset, retriever, chunker, seed)]
                    latency_values[retriever].append(
                        float(summary["avg_retrieval_latency_s"])
                    )

            cell: dict[str, Any] = {
                "dataset": dataset,
                "chunker": chunker,
                "flips": dict(sorted(flips.items())),
            }
            for metric in METRICS:
                before = metric_seed_values[metric]["hybrid"]
                after = metric_seed_values[metric]["hybrid_rerank"]
                deltas = [right - left for left, right in zip(before, after, strict=True)]
                cell[metric] = {
                    "hybrid_mean": mean(before),
                    "hybrid_sd": sample_sd(before),
                    "rerank_mean": mean(after),
                    "rerank_sd": sample_sd(after),
                    "paired_delta_mean": mean(deltas),
                    "paired_delta_sd": sample_sd(deltas),
                }
            hybrid_latency = latency_values["hybrid"]
            rerank_latency = latency_values["hybrid_rerank"]
            cell["latency"] = {
                "hybrid_seconds": mean(hybrid_latency),
                "rerank_seconds": mean(rerank_latency),
                "ratio": mean(rerank_latency) / mean(hybrid_latency),
            }
            cells.append(cell)

    return {
        "run_root": str(root),
        "source_tree_sha256": manifest.get("source_tree_sha256"),
        "seeds": list(SEEDS),
        "cells": cells,
    }


def score(value: float) -> str:
    return f"{100.0 * value:.1f}"


def render_tex(report: dict[str, Any]) -> str:
    rows = []
    for cell in report["cells"]:
        allhit = cell["all_supporting_docs_found"]
        ansvis = cell["answer_string_visible_at_k"]
        rows.append(
            "{} & \\texttt{{{}}} & {} & {} & {:+.1f} & {} & {} & {:+.1f} & {:.1f}$\\times$ \\\\".format(
                DATASET_LABELS[cell["dataset"]],
                cell["chunker"].replace("_", r"\_"),
                score(allhit["hybrid_mean"]),
                score(allhit["rerank_mean"]),
                100.0 * allhit["paired_delta_mean"],
                score(ansvis["hybrid_mean"]),
                score(ansvis["rerank_mean"]),
                100.0 * ansvis["paired_delta_mean"],
                cell["latency"]["ratio"],
            )
        )
    return "\n".join(
        [
            "% Generated by scripts/analyze_ipmc_firm_rerank.py; do not edit.",
            r"\begin{table*}[t]",
            r"\centering",
            r"\scriptsize",
            r"\setlength{\tabcolsep}{3.0pt}",
            r"\begin{tabular}{llrrrrrrr}",
            r"\toprule",
            r"Dataset & Chunker & \multicolumn{3}{c}{AllHit@4} & \multicolumn{3}{c}{AnsVis@4} & Latency \\",
            r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
            r" & & Hybrid & +Rerank & $\Delta$ & Hybrid & +Rerank & $\Delta$ & Ratio \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\caption{Paired BGE hybrid retrieval with and without cross-encoder "
                r"reranking. Quality values and deltas are percentage points averaged "
                r"over seeds 13, 21, and 34. Latency is the ratio of mean end-to-end "
                r"retrieval time.}"
            ),
            r"\label{tab:firm-rerank}",
            r"\end{table*}",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-root",
        default="outputs/ipmc_firm_rerank_bge",
        type=Path,
    )
    parser.add_argument(
        "--json-output",
        default="outputs/ipmc_firm_rerank_analysis.json",
        type=Path,
    )
    parser.add_argument(
        "--tex-output",
        default="generated/table_ipmc_firm_rerank.tex",
        type=Path,
    )
    args = parser.parse_args()

    report = analyze(args.run_root)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.tex_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.tex_output.write_text(render_tex(report), encoding="utf-8")
    print(f"Wrote {args.json_output}")
    print(f"Wrote {args.tex_output}")


if __name__ == "__main__":
    main()
