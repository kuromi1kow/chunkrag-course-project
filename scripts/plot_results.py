from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_rows(path: str | Path) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as handle:
        rows = json.load(handle)
    return pd.DataFrame(rows)


def metric_column(frame: pd.DataFrame, metric_name: str) -> str:
    aggregate_name = f"{metric_name}_mean"
    if aggregate_name in frame.columns:
        return aggregate_name
    return metric_name


def parse_chunk_size(chunker_name: str | None) -> int | None:
    if not chunker_name:
        return None
    match = re.search(r"_(\d+)$", chunker_name)
    if match:
        return int(match.group(1))
    return None


def make_fixed_chunk_plot(frame: pd.DataFrame, output_dir: Path) -> None:
    squad = frame[frame["dataset"] == "squad_v2"].copy()
    if squad.empty or "chunker" not in squad.columns:
        return
    squad["chunk_size"] = squad["chunker"].map(parse_chunk_size)
    squad = squad[squad["chunker"].fillna("").str.startswith("fixed_")]
    if squad.empty:
        return

    f1_col = metric_column(squad, "f1")
    retriever_col = "retriever" if "retriever" in squad.columns else None
    plt.figure(figsize=(7, 4.5))
    if retriever_col:
        for retriever_name, group in squad.groupby(retriever_col):
            ordered = group.sort_values("chunk_size")
            plt.plot(ordered["chunk_size"], ordered[f1_col], marker="o", label=retriever_name)
    else:
        ordered = squad.sort_values("chunk_size")
        plt.plot(ordered["chunk_size"], ordered[f1_col], marker="o")
    plt.xlabel("Chunk Size")
    plt.ylabel("SQuAD F1")
    plt.title("SQuAD F1 vs Fixed Chunk Size")
    if retriever_col:
        plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "squad_f1_vs_chunk_size.png", dpi=200)
    plt.close()


def make_latency_plot(frame: pd.DataFrame, output_dir: Path) -> None:
    latency_col = metric_column(frame, "avg_generation_latency_s")
    chunk_col = metric_column(frame, "num_chunks")
    if latency_col not in frame.columns or chunk_col not in frame.columns:
        return

    plot_frame = frame[frame["system"] != "parametric_only"].copy()
    if plot_frame.empty:
        return

    plt.figure(figsize=(7, 4.5))
    for dataset_name, group in plot_frame.groupby("dataset"):
        plt.scatter(group[chunk_col], group[latency_col], label=dataset_name, alpha=0.8)
        for _, row in group.iterrows():
            plt.annotate(row["system"], (row[chunk_col], row[latency_col]), fontsize=7, alpha=0.8)
    plt.xlabel("Number of Chunks")
    plt.ylabel("Average Generation Latency (s)")
    plt.title("Latency vs Chunk Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "latency_vs_chunks.png", dpi=200)
    plt.close()


def make_retrieval_quality_plot(frame: pd.DataFrame, output_dir: Path) -> None:
    f1_col = metric_column(frame, "f1")
    coverage_col = None
    for candidate in ("supporting_doc_coverage_mean", "supporting_doc_coverage", "recall_at_k_mean", "recall_at_k"):
        if candidate in frame.columns:
            coverage_col = candidate
            break
    if coverage_col is None or f1_col not in frame.columns:
        return

    plot_frame = frame[frame["system"] != "parametric_only"].copy()
    if plot_frame.empty:
        return

    plt.figure(figsize=(7, 4.5))
    for dataset_name, group in plot_frame.groupby("dataset"):
        plt.scatter(group[coverage_col], group[f1_col], label=dataset_name, alpha=0.8)
        for _, row in group.iterrows():
            plt.annotate(row["system"], (row[coverage_col], row[f1_col]), fontsize=7, alpha=0.8)
    plt.xlabel("Retrieval Coverage")
    plt.ylabel("Answer F1")
    plt.title("Retrieval Quality vs Answer Quality")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "retrieval_vs_answer_quality.png", dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="Path to all_results.json or aggregate_results.json")
    parser.add_argument("--output-dir", required=True, help="Directory to write figures into")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = load_rows(args.results)
    make_fixed_chunk_plot(frame, output_dir)
    make_latency_plot(frame, output_dir)
    make_retrieval_quality_plot(frame, output_dir)
    print(f"Saved figures to {output_dir}")


if __name__ == "__main__":
    main()
