from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MAIN_SYSTEM_ORDER = [
    "parametric_only",
    "fixed_128",
    "fixed_256",
    "fixed_512",
    "recursive_256",
    "sentence_256",
    "semantic_256",
]

CHONKIE_SYSTEM_ORDER = [
    "recursive_256",
    "chonkie_recursive_256",
    "semantic_256",
    "chonkie_semantic_256",
]

DATASET_LABELS = {
    "squad_v2": "SQuAD v2",
    "hotpot_qa": "HotpotQA",
}


def load_results(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of results in {path}")
    return data


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a config object in {path}")
    return data


def percent(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{value * 100:.1f}"


def decimal(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{value:.1f}"


def int_or_dash(value: int | None) -> str:
    if value is None:
        return "--"
    return str(value)


def group_by_dataset(results: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for row in results:
        dataset = row["dataset"]
        system = row["system"]
        grouped.setdefault(dataset, {})[system] = row
    return grouped


def rows_for_dataset(grouped: dict[str, dict[str, dict[str, Any]]], dataset: str, order: list[str]) -> list[dict[str, Any]]:
    dataset_rows = grouped.get(dataset, {})
    return [dataset_rows[system] for system in order if system in dataset_rows]


def sample_size_note(results: list[dict[str, Any]], config: dict[str, Any]) -> str:
    grouped = group_by_dataset(results)
    sizes: list[str] = []
    for dataset_config in config.get("datasets", []):
        dataset_name = dataset_config["name"]
        label = DATASET_LABELS.get(dataset_name, dataset_name)
        rows = grouped.get(dataset_name, {})
        num_examples = None
        for row in rows.values():
            if row.get("num_examples") is not None:
                num_examples = row["num_examples"]
                break
        if num_examples is not None:
            sizes.append(f"{label} ($n={num_examples}$)")
    size_text = ", ".join(sizes)
    if size_text:
        return rf"\noindent\textit{{Single-run percentages. Current subsets: {size_text}.}}"
    return r"\noindent\textit{Single-run percentages from the current midway subsets.}"


def latex_escape_system(name: str) -> str:
    return name.replace("_", r"\_")


def max_metric_values(rows: list[dict[str, Any]], metric_names: list[str]) -> dict[str, float | None]:
    maxima: dict[str, float | None] = {}
    for metric_name in metric_names:
        values = [float(row[metric_name]) for row in rows if row.get(metric_name) is not None]
        maxima[metric_name] = max(values) if values else None
    return maxima


def highlight_if_max(value: float | None, maximum: float | None, rendered: str) -> str:
    if value is None or maximum is None:
        return rendered
    if abs(float(value) - maximum) < 1e-12:
        return rf"\textbf{{{rendered}}}"
    return rendered


def caption_for_dataset(dataset: str, rows: list[dict[str, Any]]) -> str:
    if dataset == "hotpot_qa":
        return "Results on HotpotQA. Doc-level Recall@4 shows that recovering all supporting documents remains incomplete in the current run."

    dataset_label = DATASET_LABELS.get(dataset, dataset)
    f1_values = [float(row["f1"]) for row in rows if row.get("f1") is not None]
    if not f1_values:
        return f"Results on {dataset_label}."
    max_f1 = max(f1_values)
    winners = [row["system"] for row in rows if row.get("f1") is not None and abs(float(row["f1"]) - max_f1) < 1e-12]
    if len(winners) == 1:
        winner_text = winners[0].replace("_", " ")
        return f"Results on {dataset_label}. {winner_text.capitalize()} is the strongest end-to-end setting in the current run."
    winner_text = " and ".join(system.replace("_", " ") for system in winners)
    return f"Results on {dataset_label}. {winner_text.capitalize()} tie for the strongest end-to-end setting in the current run."


def latex_main_table(dataset: str, rows: list[dict[str, Any]], caption: str, label: str) -> str:
    maxima = max_metric_values(rows, ["exact_match", "f1", "recall_at_k", "precision_at_k"])
    body_lines: list[str] = []
    for row in rows:
        system = row["system"]
        em = highlight_if_max(row.get("exact_match"), maxima["exact_match"], percent(row.get("exact_match")))
        f1 = highlight_if_max(row.get("f1"), maxima["f1"], percent(row.get("f1")))
        recall = highlight_if_max(row.get("recall_at_k"), maxima["recall_at_k"], percent(row.get("recall_at_k")))
        precision = highlight_if_max(row.get("precision_at_k"), maxima["precision_at_k"], percent(row.get("precision_at_k")))
        body_lines.append(
            " & ".join(
                [
                    rf"\texttt{{{latex_escape_system(system)}}}",
                    em,
                    f1,
                    recall,
                    precision,
                    decimal(row.get("avg_chunk_tokens")),
                    int_or_dash(row.get("num_chunks")),
                ]
            )
            + r" \\"
        )
    return "\n".join(
        [
            r"\begin{table*}[t]",
            r"\centering",
            r"\small",
            r"\begin{tabular}{lrrrrrr}",
            r"\toprule",
            r"System & EM & F1 & Recall@4 & Precision@4 & Avg chunk tokens & \# chunks \\",
            r"\midrule",
            *body_lines,
            r"\bottomrule",
            r"\end{tabular}",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            r"\end{table*}",
        ]
    )


def latex_chonkie_table(grouped: dict[str, dict[str, dict[str, Any]]]) -> str:
    squad_rows = grouped.get("squad_v2", {})
    hotpot_rows = grouped.get("hotpot_qa", {})
    combined_rows = []
    for system in CHONKIE_SYSTEM_ORDER:
        if system not in squad_rows or system not in hotpot_rows:
            continue
        combined_rows.append(
            {
                "system": system,
                "squad_exact_match": squad_rows[system].get("exact_match"),
                "squad_f1": squad_rows[system].get("f1"),
                "hotpot_f1": hotpot_rows[system].get("f1"),
            }
        )
    maxima = max_metric_values(combined_rows, ["squad_exact_match", "squad_f1", "hotpot_f1"])
    body_lines: list[str] = []
    for system in CHONKIE_SYSTEM_ORDER:
        if system not in squad_rows or system not in hotpot_rows:
            continue
        squad = squad_rows[system]
        hotpot = hotpot_rows[system]
        body_lines.append(
            " & ".join(
                [
                    rf"\texttt{{{latex_escape_system(system)}}}",
                    highlight_if_max(squad.get("exact_match"), maxima["squad_exact_match"], percent(squad.get("exact_match"))),
                    highlight_if_max(squad.get("f1"), maxima["squad_f1"], percent(squad.get("f1"))),
                    highlight_if_max(hotpot.get("f1"), maxima["hotpot_f1"], percent(hotpot.get("f1"))),
                    decimal(squad.get("avg_chunk_tokens")),
                ]
            )
            + r" \\"
        )
    return "\n".join(
        [
            r"\begin{table*}[t]",
            r"\centering",
            r"\small",
            r"\begin{tabular}{lrrrr}",
            r"\toprule",
            r"System & SQuAD EM & SQuAD F1 & HotpotQA F1 & Avg chunk tokens \\",
            r"\midrule",
            *body_lines,
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Auxiliary Chonkie implementation comparison. Recursive chunking is stable across implementations, while semantic chunking is more sensitive to boundary heuristics.}",
            r"\label{tab:chonkie}",
            r"\end{table*}",
        ]
    )


def markdown_main_table(dataset_label: str, rows: list[dict[str, Any]]) -> str:
    lines = [
        f"## {dataset_label}",
        "",
        "| System | EM | F1 | Recall@4 | Precision@4 | Avg chunk tokens | # chunks |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['system']}`",
                    percent(row.get("exact_match")),
                    percent(row.get("f1")),
                    percent(row.get("recall_at_k")),
                    percent(row.get("precision_at_k")),
                    decimal(row.get("avg_chunk_tokens")),
                    int_or_dash(row.get("num_chunks")),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def markdown_chonkie_table(grouped: dict[str, dict[str, dict[str, Any]]]) -> str:
    lines = [
        "## Chonkie Comparison",
        "",
        "| System | SQuAD EM | SQuAD F1 | HotpotQA F1 | Avg chunk tokens |",
        "|---|---:|---:|---:|---:|",
    ]
    squad_rows = grouped.get("squad_v2", {})
    hotpot_rows = grouped.get("hotpot_qa", {})
    for system in CHONKIE_SYSTEM_ORDER:
        if system not in squad_rows or system not in hotpot_rows:
            continue
        squad = squad_rows[system]
        hotpot = hotpot_rows[system]
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{system}`",
                    percent(squad.get("exact_match")),
                    percent(squad.get("f1")),
                    percent(hotpot.get("f1")),
                    decimal(squad.get("avg_chunk_tokens")),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export auto-generated report tables from saved experiment outputs.")
    parser.add_argument(
        "--main-results",
        type=Path,
        default=Path("outputs/report_run/all_results.json"),
        help="Path to the primary experiment result JSON.",
    )
    parser.add_argument(
        "--main-config",
        type=Path,
        default=Path("outputs/report_run/experiment_config.json"),
        help="Path to the primary experiment config JSON.",
    )
    parser.add_argument(
        "--chonkie-results",
        type=Path,
        default=Path("outputs/chonkie_report_run/all_results.json"),
        help="Path to the auxiliary Chonkie result JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/generated"),
        help="Directory where generated Markdown and LaTeX snippets will be written.",
    )
    args = parser.parse_args()

    main_results = load_results(args.main_results)
    main_config = load_config(args.main_config)
    chonkie_results = load_results(args.chonkie_results) if args.chonkie_results.exists() else []

    main_grouped = group_by_dataset(main_results)
    chonkie_grouped = group_by_dataset(chonkie_results)

    squad_rows = rows_for_dataset(main_grouped, "squad_v2", MAIN_SYSTEM_ORDER)
    hotpot_rows = rows_for_dataset(main_grouped, "hotpot_qa", MAIN_SYSTEM_ORDER)

    latex_parts = [
        "% Auto-generated by scripts/export_report_tables.py",
        latex_main_table(
            "squad_v2",
            squad_rows,
            caption_for_dataset("squad_v2", squad_rows),
            "tab:squad",
        ),
        "",
        latex_main_table(
            "hotpot_qa",
            hotpot_rows,
            caption_for_dataset("hotpot_qa", hotpot_rows),
            "tab:hotpot",
        ),
        "",
        sample_size_note(main_results, main_config),
    ]

    markdown_parts = [
        "# Auto-generated Report Tables",
        "",
        sample_size_note(main_results, main_config)
        .replace(r"\noindent\textit{Note:} ", "Note: ")
        .replace(r"$", "")
        .replace(r"\_", "_"),
        "",
        markdown_main_table("SQuAD v2", squad_rows),
        "",
        markdown_main_table("HotpotQA", hotpot_rows),
    ]
    if chonkie_results:
        markdown_parts.extend(["", markdown_chonkie_table(chonkie_grouped)])

    write_text(args.output_dir / "midway_tables.tex", "\n".join(latex_parts))
    write_text(args.output_dir / "midway_tables.md", "\n".join(markdown_parts))
    if chonkie_results:
        write_text(args.output_dir / "chonkie_table.tex", latex_chonkie_table(chonkie_grouped))


if __name__ == "__main__":
    main()
