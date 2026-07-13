"""Frozen paper-output assignment and regeneration (Specification Section 30)."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .canonical import atomic_write_json
from .constants import MAIN_FIGURES, MAIN_TABLES, PROTOCOL_ID


def output_assignment_manifest() -> dict[str, Any]:
    return {
        "schema_version": PROTOCOL_ID,
        "main_figures": {name: list(experiments) for name, experiments in MAIN_FIGURES.items()},
        "main_tables": {name: list(experiments) for name, experiments in MAIN_TABLES.items()},
        "appendix_figures": {
            "A1": ["archive"], "A2": ["E1"], "A3": ["E1", "E5"], "A4": ["E4"],
            "A5": ["E7"], "A6": ["E4"], "A7": ["E1"], "A8": ["E6"],
        },
        "appendix_tables": {
            "B1": ["E1"], "B2": ["E1"], "B3": ["E2", "E3"], "B4": ["E2"],
            "B5": ["E4"], "B6": ["E5"], "B7": ["E6"], "B8": ["E7"],
            "B9": ["E7"], "B10": ["archive"], "B11": ["protocol", "E4"], "B12": ["E0"],
        },
    }


def write_output_assignment(path: Path) -> str:
    return atomic_write_json(path, output_assignment_manifest())


def validate_output_assignments() -> None:
    manifest = output_assignment_manifest()
    if set(manifest["main_figures"]) != {"figure1", "figure2", "figure3"}:
        raise ValueError("Exactly three main figures are required")
    if set(manifest["main_tables"]) != {"table1", "table2", "table3"}:
        raise ValueError("Exactly three main tables are required")


def _escape(value: Any) -> str:
    return str(value).replace("_", "\\_").replace("%", "\\%")


def regenerate_tables(analysis: Mapping[str, Any], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    primary = analysis.get("primary", [])
    lines = ["\\begin{tabular}{lrrrrrr}", "Test & $\\Delta$ & 95\\% low & 95\\% high & $r_{rb}$ & $p$ & Holm $p$ \\\\", "\\hline"]
    for row in primary:
        lines.append(f"{_escape(row['test_id'])} & {row['mean_difference']:.3f} & {row['ci95_low']:.3f} & {row['ci95_high']:.3f} & {row['rank_biserial']:.3f} & {row['raw_p']:.4g} & {row['holm_p']:.4g} \\\\")
    lines.extend(["\\end{tabular}", ""])
    table2 = output_dir / "table2_primary_contrasts.tex"
    table2.write_text("\n".join(lines), encoding="utf-8")
    dataset_lines = ["\\begin{tabular}{lrrrll}", "Dataset & Questions & Documents & Clusters & Question hash & Corpus hash \\\\", "\\hline"]
    for row in analysis.get("dataset_summary", []):
        dataset_lines.append(f"{_escape(row['dataset'])} & {row['questions']} & {row['documents']} & {row['clusters']} & \\texttt{{{row['question_hash'][:8]}}} & \\texttt{{{row['corpus_hash'][:8]}}} \\\\")
    dataset_lines.extend(["\\end{tabular}", ""])
    table1 = output_dir / "table1_dataset_manifests.tex"
    table1.write_text("\n".join(dataset_lines), encoding="utf-8")
    table3 = output_dir / "table3_gold_techqa.tex"
    table3.write_text("\\begin{tabular}{lll}\nCondition & Metric & Value \\\\\n\\hline\nGold/TechQA rows are generated from E3/E4 locked summaries.\\\n\n\\end{tabular}\n", encoding="utf-8")
    manifest_path = output_dir / "paper-output-assignment.json"
    atomic_write_json(manifest_path, output_assignment_manifest(), overwrite=True)
    return [table1, table2, table3, manifest_path]


def regenerate_figures(analysis: Mapping[str, Any], output_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    figures: list[Path] = []
    fig, axis = plt.subplots(figsize=(7.2, 2.3))
    axis.axis("off")
    axis.text(0.02, 0.65, "Boundary policy", bbox={"boxstyle": "round", "fc": "#e8f1fa"})
    axis.text(0.38, 0.65, "Frozen retrieval trace", bbox={"boxstyle": "round", "fc": "#eef7e8"})
    axis.text(0.72, 0.65, "Matched evidence", bbox={"boxstyle": "round", "fc": "#fff2cc"})
    axis.annotate("", (0.36, 0.69), (0.20, 0.69), arrowprops={"arrowstyle": "->"})
    axis.annotate("", (0.70, 0.69), (0.56, 0.69), arrowprops={"arrowstyle": "->"})
    axis.text(0.02, 0.20, "Operational effect = policy + exposure\nBoundary effect = exact − jitter control", fontsize=9)
    path1 = output_dir / "figure1_treatment_diagram.pdf"
    fig.savefig(path1, bbox_inches="tight")
    plt.close(fig)
    figures.append(path1)

    h1 = [row for row in analysis.get("primary", []) if row["test_id"].startswith("H1:")]
    fig, axis = plt.subplots(figsize=(7.2, max(2.6, 0.42 * len(h1))))
    positions = list(range(len(h1)))
    values = [row["mean_difference"] for row in h1]
    low = [value - row["ci95_low"] for value, row in zip(values, h1)]
    high = [row["ci95_high"] - value for value, row in zip(values, h1)]
    axis.errorbar(values, positions, xerr=[low, high], fmt="o", color="#2F6690", capsize=3)
    axis.axvspan(-2, 2, color="#dddddd", alpha=0.5)
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_yticks(positions, [row["test_id"].replace("H1:", "") for row in h1])
    axis.set_xlabel("Exact structured policy minus mean jitter control (F1 points)")
    path2 = output_dir / "figure2_boundary_forest.pdf"
    fig.savefig(path2, bbox_inches="tight")
    plt.close(fig)
    figures.append(path2)
    rows = list(analysis.get("exposure_rows", []))
    fig, axes = plt.subplots(2, 3, figsize=(9.0, 4.8), sharey="row")
    datasets = ("squad_v2", "hotpot_qa", "techqa")
    for column, dataset in enumerate(datasets):
        evidence_rows = [row for row in rows if row["dataset"] == dataset]
        answer_rows = [] if dataset == "techqa" and analysis.get("techqa", {}).get("remove_from_main", False) else evidence_rows
        axes[0, column].axhline(0, color="black", linewidth=0.8)
        answer_means = [row["answer_mean"] for row in answer_rows]
        axes[0, column].bar(
            range(len(answer_rows)), answer_means, color="#2F6690",
            yerr=[
                [mean - row["answer_ci_low"] for mean, row in zip(answer_means, answer_rows)],
                [row["answer_ci_high"] - mean for mean, row in zip(answer_means, answer_rows)],
            ], capsize=2,
        )
        title = dataset
        if dataset == "techqa" and not analysis.get("techqa", {}).get("validated", False):
            title += " (exploratory)"
        axes[0, column].set_title(title)
        axes[1, column].axhline(0, color="black", linewidth=0.8)
        evidence_means = [row["evidence_mean"] for row in evidence_rows]
        axes[1, column].bar(
            range(len(evidence_rows)), evidence_means, color="#D18F00",
            yerr=[
                [mean - row["evidence_ci_low"] for mean, row in zip(evidence_means, evidence_rows)],
                [row["evidence_ci_high"] - mean for mean, row in zip(evidence_means, evidence_rows)],
            ], capsize=2,
        )
        labels = [f"{row['policy'].replace('192','')}\n{row['condition'].replace('operational','op').replace('matched','match')}" for row in evidence_rows]
        axes[1, column].set_xticks(range(len(labels)), labels, rotation=90, fontsize=6)
    axes[0, 0].set_ylabel("Answer effect")
    axes[1, 0].set_ylabel("Consumed-evidence effect")
    path3 = output_dir / "figure3_exposure_mechanism.pdf"
    fig.savefig(path3, bbox_inches="tight")
    plt.close(fig)
    figures.append(path3)
    return figures


def regenerate_protocol_tables(analysis: Mapping[str, Any], output_dir: Path) -> list[Path]:
    """Generate all three frozen main tables from locked E0/E2/E3/E4 summaries."""
    outputs = regenerate_tables(analysis, output_dir)
    row_end = chr(92) * 2
    table3_lines = ["\\begin{tabular}{llll}", "Dataset & Condition & Metric & Value " + row_end, "\\hline"]
    gold_techqa = analysis.get("gold_techqa", {})
    for row in gold_techqa.get("gold", []):
        table3_lines.append(
            f"{_escape(row['dataset'])} & {_escape(row['condition'])} & {row['metric']} & {row['value']:.4f} " + row_end
        )
    if gold_techqa.get("techqa_semantic_utility") is not None:
        table3_lines.append(
            f"techqa & retrieved & semantic utility & {gold_techqa['techqa_semantic_utility']:.4f} " + row_end
        )
        table3_lines.append(
            f"techqa & retrieved & groundedness & {gold_techqa['techqa_groundedness']:.4f} " + row_end
        )
    table3_lines.extend(["\\end{tabular}", ""])
    (output_dir / "table3_gold_techqa.tex").write_text("\n".join(table3_lines), encoding="utf-8")
    return outputs


def regenerate_paper_artifacts(analysis: Mapping[str, Any], output_dir: Path) -> list[Path]:
    validate_output_assignments()
    return regenerate_protocol_tables(analysis, output_dir) + regenerate_figures(analysis, output_dir)
