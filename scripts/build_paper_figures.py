"""Build publication-quality figures for the final ACL report.

Figures produced (numbered to match the discussion in the report):

    fig1_chunker_summary       Grouped bar chart of EM and F1 by chunker, faceted by dataset.
    fig2_error_breakdown       Stacked bar chart of coarse error categories per chunker per dataset.
    fig4_retrieval_vs_answer   Scatter of answer F1 vs Recall@4 per chunker per dataset.
    fig5_fixed_size_curve      F1 (and EM) vs fixed chunk size for the fixed_* family on SQuAD.

Note: the system architecture diagram (Figure 1 in the final report) is rendered
with TikZ inline in `reports/final_report_acl.tex`, not by this script.

Inputs:
    outputs/midway_mistral_endpoint_v2/aggregate_results.json
    outputs/revision_audit/failure_reanalysis.json

The aggregate file above is the one whose numbers match the tables in
`reports/final_report_acl.tex`.

Outputs:
    reports/figures/<stem>.pdf
    reports/figures/<stem>.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CHUNKER_ORDER = [
    "fixed_128",
    "fixed_256",
    "fixed_512",
    "recursive_256",
    "sentence_256",
    "semantic_256",
    "parametric_only",
]

CHUNKER_PRETTY = {
    "fixed_128": "fixed_128",
    "fixed_256": "fixed_256",
    "fixed_512": "fixed_512",
    "recursive_256": "recursive_256",
    "sentence_256": "sentence_256",
    "semantic_256": "semantic_256",
    "parametric_only": "parametric_only",
}

DATASET_PRETTY = {"squad_v2": "SQuAD v2", "hotpot_qa": "HotpotQA"}

PALETTE = {
    "fixed_128": "#9ecae1",
    "fixed_256": "#6baed6",
    "fixed_512": "#3182bd",
    "recursive_256": "#e6550d",
    "sentence_256": "#74c476",
    "semantic_256": "#9e9ac8",
    "parametric_only": "#bdbdbd",
}

ERROR_PALETTE = {
    "Evidence limited": "#d62728",
    "Form/refusal candidate": "#1f77b4",
    "Content mismatch": "#7f7f7f",
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
            "grid.linewidth": 0.5,
            "savefig.bbox": "tight",
            "savefig.dpi": 200,
            "pdf.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.pdf")
    fig.savefig(out_dir / f"{stem}.png")
    plt.close(fig)


def load_aggregate(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)
    frame = pd.DataFrame(rows)
    frame["chunker_label"] = frame["system"].fillna("parametric_only")
    return frame


def load_error_summary(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = []
    for dataset, systems in payload["datasets"].items():
        for chunker, values in systems.items():
            coarse = values["coarse"]
            rows.append(
                {
                    "dataset": dataset,
                    "chunker": chunker,
                    "em_zero": values["em_zero"],
                    "evidence_limited_pct": coarse["evidence_limited"]["percentage"],
                    "response_form_candidate_pct": coarse["response_form_candidate"]["percentage"],
                    "answer_content_error_pct": coarse["answer_content_error"]["percentage"],
                }
            )
    return pd.DataFrame(rows)


def fig1_chunker_summary(agg: pd.DataFrame, out_dir: Path) -> None:
    """Grouped bar chart: EM and F1 per chunker, side-by-side panels for SQuAD and HotpotQA."""
    datasets = ["squad_v2", "hotpot_qa"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), sharey=False)

    for ax, dataset in zip(axes, datasets):
        sub = agg[agg["dataset"] == dataset].copy()
        sub["chunker_label"] = pd.Categorical(
            sub["chunker_label"], categories=CHUNKER_ORDER, ordered=True
        )
        sub = sub.sort_values("chunker_label").reset_index(drop=True)

        chunkers = sub["chunker_label"].astype(str).tolist()
        em_vals = sub["exact_match_mean"].fillna(0).to_numpy() * 100.0
        f1_vals = sub["f1_mean"].fillna(0).to_numpy() * 100.0

        x = np.arange(len(chunkers))
        bar_width = 0.38

        bars_em = ax.bar(
            x - bar_width / 2,
            em_vals,
            bar_width,
            label="EM",
            color="#bdbdbd",
            edgecolor="black",
            linewidth=0.4,
        )
        bars_f1 = ax.bar(
            x + bar_width / 2,
            f1_vals,
            bar_width,
            label="F1",
            color=[PALETTE.get(c, "#cccccc") for c in chunkers],
            edgecolor="black",
            linewidth=0.4,
        )

        for bars in (bars_em, bars_f1):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(
                    f"{height:.1f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                )

        ax.set_xticks(x)
        ax.set_xticklabels(chunkers, rotation=30, ha="right")
        ax.set_ylim(0, max(f1_vals.max(), em_vals.max()) * 1.18)
        ax.set_title(DATASET_PRETTY[dataset])
        ax.set_ylabel("Score (%)")

        best_idx = int(np.argmax(f1_vals))
        bars_f1[best_idx].set_edgecolor("black")
        bars_f1[best_idx].set_linewidth(1.6)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#bdbdbd", edgecolor="black"),
        plt.Rectangle((0, 0), 1, 1, color="#3182bd", edgecolor="black"),
    ]
    fig.legend(
        handles,
        ["Exact Match (EM)", "Token-level F1 (color = chunker)"],
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.02),
        frameon=False,
    )
    fig.suptitle("End-to-end QA quality by chunker", y=1.08, fontsize=12)
    fig.tight_layout()
    save_figure(fig, out_dir, "fig1_chunker_summary")


def fig2_error_breakdown(err: pd.DataFrame, out_dir: Path) -> None:
    """Stacked bar of coarse error categories per chunker per dataset."""
    err = err.copy()
    err = err[err["chunker"] != "parametric_only"]
    err["chunker"] = pd.Categorical(
        err["chunker"], categories=[c for c in CHUNKER_ORDER if c != "parametric_only"], ordered=True
    )
    err = err.sort_values(["dataset", "chunker"])

    datasets = ["squad_v2", "hotpot_qa"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)

    for ax, dataset in zip(axes, datasets):
        sub = err[err["dataset"] == dataset]
        chunkers = sub["chunker"].astype(str).tolist()
        x = np.arange(len(chunkers))

        retrieval = sub["evidence_limited_pct"].to_numpy()
        form_candidates = sub["response_form_candidate_pct"].to_numpy()
        model_err = sub["answer_content_error_pct"].to_numpy()

        bottom = np.zeros(len(chunkers))
        for label, values in [
            ("Evidence limited", retrieval),
            ("Form/refusal candidate", form_candidates),
            ("Content mismatch", model_err),
        ]:
            bars = ax.bar(
                x,
                values,
                bottom=bottom,
                label=label,
                color=ERROR_PALETTE[label],
                edgecolor="black",
                linewidth=0.4,
                width=0.65,
            )
            for bar, val in zip(bars, values):
                if val >= 5.0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_y() + bar.get_height() / 2,
                        f"{val:.0f}%",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="white",
                    )
            bottom = bottom + values

        for xi, total in zip(x, sub["em_zero"].to_numpy()):
            ax.text(xi, 102, f"n={int(total)}", ha="center", va="bottom", fontsize=7.5, color="#444444")

        ax.set_xticks(x)
        ax.set_xticklabels(chunkers, rotation=30, ha="right")
        ax.set_ylim(0, 112)
        ax.set_ylabel("Share of EM=0 failures (%)")
        ax.set_title(DATASET_PRETTY[dataset])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02), frameon=False
    )
    fig.suptitle("Automatic diagnostic breakdown of EM=0 predictions", y=1.08, fontsize=12)
    fig.tight_layout()
    save_figure(fig, out_dir, "fig2_error_breakdown")


def fig4_retrieval_vs_answer(agg: pd.DataFrame, out_dir: Path) -> None:
    """Scatter of F1 vs Recall@4, two panels (one per dataset)."""
    sub = agg[agg["chunker_label"] != "parametric_only"].copy()

    datasets = ["squad_v2", "hotpot_qa"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    label_offsets = {
        "squad_v2": {
            "fixed_128": (-72, 6),
            "fixed_256": (-72, 6),
            "fixed_512": (8, -3),
            "recursive_256": (8, 6),
            "sentence_256": (8, -3),
            "semantic_256": (-78, 6),
        },
        "hotpot_qa": {
            "fixed_128": (-72, 6),
            "fixed_256": (-72, -12),
            "fixed_512": (8, -12),
            "recursive_256": (8, 6),
            "sentence_256": (-78, -3),
            "semantic_256": (8, -3),
        },
    }

    for ax, dataset in zip(axes, datasets):
        group = sub[sub["dataset"] == dataset]
        for _, row in group.iterrows():
            chunker = row["chunker_label"]
            x = row["recall_at_k_mean"] * 100.0
            y = row["f1_mean"] * 100.0
            ax.scatter(
                x,
                y,
                s=160,
                marker="o",
                facecolor=PALETTE.get(chunker, "#cccccc"),
                edgecolor="black",
                linewidth=0.7,
                alpha=0.95,
                zorder=3,
            )
            offset = label_offsets[dataset].get(chunker, (8, 5))
            ax.annotate(
                chunker,
                (x, y),
                xytext=offset,
                textcoords="offset points",
                fontsize=8.5,
                zorder=4,
            )

        ax.set_xlabel("Recall@4 (%)")
        ax.set_ylabel("Token-level F1 (%)")
        ax.set_title(DATASET_PRETTY[dataset])

        recall_vals = group["recall_at_k_mean"].to_numpy() * 100
        f1_vals = group["f1_mean"].to_numpy() * 100
        x_pad = max(2.5, (recall_vals.max() - recall_vals.min()) * 0.6)
        y_pad = max(3.0, (f1_vals.max() - f1_vals.min()) * 0.4)
        ax.set_xlim(recall_vals.min() - x_pad, recall_vals.max() + x_pad)
        ax.set_ylim(f1_vals.min() - y_pad, f1_vals.max() + y_pad)

    chunker_handles = [
        plt.Line2D(
            [0], [0],
            marker="o",
            color="white",
            markerfacecolor=PALETTE[c],
            markeredgecolor="black",
            markersize=10,
            label=c,
        )
        for c in CHUNKER_ORDER
        if c != "parametric_only"
    ]
    fig.legend(
        handles=chunker_handles,
        loc="upper center",
        ncol=6,
        bbox_to_anchor=(0.5, 1.02),
        frameon=False,
        fontsize=9,
    )
    fig.suptitle(
        "Retrieval coverage vs answer quality\n(coverage and F1 do not move together: high Recall@4 chunkers can have lower F1)",
        y=1.13,
        fontsize=12,
    )
    fig.tight_layout()
    save_figure(fig, out_dir, "fig4_retrieval_vs_answer")


def fig5_fixed_size_curve(agg: pd.DataFrame, out_dir: Path) -> None:
    """F1 and EM as a function of fixed chunk size, for both datasets."""
    sub = agg[agg["chunker_label"].str.startswith("fixed_")].copy()
    sub["chunk_size"] = sub["chunker_label"].str.extract(r"_(\d+)$").astype(int)
    sub = sub.sort_values(["dataset", "chunk_size"])

    fig, ax = plt.subplots(figsize=(6.8, 4.4))

    dataset_styles = {
        "squad_v2": {"color": "#3182bd", "marker": "o", "linestyle": "-"},
        "hotpot_qa": {"color": "#e6550d", "marker": "s", "linestyle": "-"},
    }

    for dataset, group in sub.groupby("dataset"):
        style = dataset_styles[dataset]
        ax.plot(
            group["chunk_size"],
            group["f1_mean"] * 100.0,
            marker=style["marker"],
            linestyle="-",
            color=style["color"],
            linewidth=2,
            markersize=8,
            label=f"{DATASET_PRETTY[dataset]} F1",
        )
        ax.plot(
            group["chunk_size"],
            group["exact_match_mean"] * 100.0,
            marker=style["marker"],
            linestyle="--",
            color=style["color"],
            linewidth=1.4,
            markersize=6,
            alpha=0.7,
            label=f"{DATASET_PRETTY[dataset]} EM",
        )
        for _, row in group.iterrows():
            ax.annotate(
                f"{row['f1_mean']*100:.1f}",
                (row["chunk_size"], row["f1_mean"] * 100.0),
                xytext=(0, 9),
                textcoords="offset points",
                ha="center",
                fontsize=8.5,
                color=style["color"],
                fontweight="bold",
            )

    ax.set_xticks([128, 256, 512])
    ax.set_xlabel("Fixed chunk size (target tokens)")
    ax.set_ylabel("Score (%)")
    ax.set_ylim(15, 64)
    ax.set_title("Fixed chunk size vs QA quality\n(SQuAD F1 peaks at 256; HotpotQA is roughly flat)")
    ax.legend(loc="upper right", ncol=2, frameon=True, fontsize=8)
    fig.tight_layout()
    save_figure(fig, out_dir, "fig5_fixed_size_curve")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--aggregate",
        type=Path,
        default=Path("outputs/midway_mistral_endpoint_v2/aggregate_results.json"),
        help="Aggregate results JSON used in the paper tables.",
    )
    parser.add_argument(
        "--errors",
        type=Path,
        default=Path("outputs/revision_audit/failure_reanalysis.json"),
        help="Corrected failure-analysis JSON.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reports/figures"),
        help="Where to write figure files.",
    )
    args = parser.parse_args()

    setup_style()

    agg = load_aggregate(args.aggregate)
    err = load_error_summary(args.errors)

    fig1_chunker_summary(agg, args.out_dir)
    fig2_error_breakdown(err, args.out_dir)
    fig4_retrieval_vs_answer(agg, args.out_dir)
    fig5_fixed_size_curve(agg, args.out_dir)

    print(f"Wrote figures to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
