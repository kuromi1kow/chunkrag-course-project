#!/usr/bin/env python3
"""Build figures introduced by the reviewer-driven manuscript revision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SYSTEMS = (
    "fixed_128",
    "fixed_256",
    "fixed_512",
    "recursive_256",
    "sentence_256",
    "semantic_256",
)
LABELS = ("Fixed 128", "Fixed 256", "Fixed 512", "Recursive 256", "Sentence 256", "Semantic 256")
COLORS = {"squad_v2": "#2F6690", "hotpot_qa": "#D18F00"}


def values(payload: dict, dataset: str, key: str) -> list[float]:
    return [payload["datasets"][dataset][system]["summary"][key] for system in SYSTEMS]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("context_audit", type=Path)
    parser.add_argument("output_prefix", type=Path)
    args = parser.parse_args()

    payload = json.loads(args.context_audit.read_text(encoding="utf-8"))
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    x = np.arange(len(SYSTEMS))
    width = 0.36
    figure, axes = plt.subplots(1, 2, figsize=(7.15, 2.55), constrained_layout=True)

    for offset, dataset, label in ((-width / 2, "squad_v2", "SQuAD 2.0"), (width / 2, "hotpot_qa", "HotpotQA")):
        truncation = [100.0 * value for value in values(payload, dataset, "truncation_rate")]
        axes[0].bar(x + offset, truncation, width, label=label, color=COLORS[dataset])
        consumed = values(payload, dataset, "mean_fully_consumed_chunks")
        axes[1].bar(x + offset, consumed, width, label=label, color=COLORS[dataset])

    axes[0].set_title("(a) Prefix-truncated prompts")
    axes[0].set_ylabel("Questions (%)")
    axes[0].set_ylim(0, 105)
    axes[0].set_yticks((0, 25, 50, 75, 100))

    axes[1].set_title("(b) Retrieved chunks fully consumed")
    axes[1].set_ylabel("Mean number (max. 4)")
    axes[1].set_ylim(0, 4.2)
    axes[1].set_yticks((0, 1, 2, 3, 4))

    for axis in axes:
        axis.set_xticks(x)
        axis.set_xticklabels(LABELS, rotation=28, ha="right")
        axis.grid(axis="y", color="#D7D7D7", linewidth=0.55)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, legend_labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.17))

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(args.output_prefix.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
