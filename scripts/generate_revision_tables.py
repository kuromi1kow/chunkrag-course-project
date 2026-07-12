#!/usr/bin/env python3
"""Generate reviewer-revision LaTeX tables from committed result artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SYSTEMS = (
    "parametric_only",
    "fixed_128",
    "fixed_256",
    "fixed_512",
    "recursive_256",
    "sentence_256",
    "semantic_256",
)
CHUNKERS = SYSTEMS[1:]
DISPLAY = {
    "parametric_only": r"No context",
    "fixed_128": r"\texttt{fixed\_128}",
    "fixed_256": r"\texttt{fixed\_256}",
    "fixed_512": r"\texttt{fixed\_512}",
    "recursive_256": r"\texttt{recursive\_256}",
    "sentence_256": r"\texttt{sentence\_256}",
    "semantic_256": r"\texttt{semantic\_256}",
}


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}"


def bold_if(value: float, maximum: float, rendered: str) -> str:
    return rf"\textbf{{{rendered}}}" if abs(value - maximum) < 1e-12 else rendered


def results_table(dataset: str, summaries: list[dict], statistics: dict) -> str:
    by_system = {row["system"]: row for row in summaries}
    statistical_systems = statistics["datasets"][dataset]["systems"]
    max_em = max(by_system[system]["exact_match"] for system in CHUNKERS)
    max_f1 = max(statistical_systems[system]["f1"]["mean"] for system in CHUNKERS)
    max_cov = max(by_system[system]["supporting_doc_coverage"] for system in CHUNKERS)
    max_all = max(by_system[system]["all_supporting_docs_found"] for system in CHUNKERS)
    rows = []
    for system in SYSTEMS:
        item = by_system[system]
        f1_stats = statistical_systems[system]["f1"]
        em = pct(item["exact_match"])
        f1 = pct(f1_stats["mean"])
        f1_ci = f"[{pct(f1_stats['ci_low'])}, {pct(f1_stats['ci_high'])}]"
        if system != "parametric_only":
            em = bold_if(item["exact_match"], max_em, em)
            f1 = bold_if(f1_stats["mean"], max_f1, f1)
            coverage = bold_if(item["supporting_doc_coverage"], max_cov, pct(item["supporting_doc_coverage"]))
            all_hit = bold_if(item["all_supporting_docs_found"], max_all, pct(item["all_supporting_docs_found"]))
            avg_tokens = f"{item['avg_chunk_tokens']:.1f}"
            num_chunks = str(item["num_chunks"])
        else:
            coverage = all_hit = avg_tokens = num_chunks = "--"
        rows.append(
            f"{DISPLAY[system]} & {em} & {f1} & {f1_ci} & {coverage} & {all_hit} & {avg_tokens} & {num_chunks} \\\\"
        )
    title = "SQuAD 2.0" if dataset == "squad_v2" else "HotpotQA"
    label = "squad" if dataset == "squad_v2" else "hotpot"
    return "\n".join(
        [
            r"\begin{table*}[t]",
            r"\centering",
            r"\small",
            r"\setlength{\tabcolsep}{4.1pt}",
            r"\begin{tabular}{lrrrrrrr}",
            r"\toprule",
            r"System & EM & F1 & F1 95\% CI & DocCov@4 & AllHit@4 & Avg. tokens & \# chunks \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            rf"\caption{{Observed results on the fixed {title} sample. EM, F1, DocCov@4, and AllHit@4 are percentage points. Intervals are question-level percentile-bootstrap intervals and do not measure run-to-run variation. Bold marks the largest observed chunker mean; it does not indicate statistical significance.}}",
            rf"\label{{tab:{label}-results}}",
            r"\end{table*}",
        ]
    )


def paired_table(statistics: dict) -> str:
    rows = []
    for comparator in ("fixed_128", "fixed_256", "fixed_512", "sentence_256", "semantic_256"):
        cells = []
        for dataset in ("squad_v2", "hotpot_qa"):
            item = statistics["datasets"][dataset]["paired_against_recursive_256"][comparator]["f1"]
            cells.extend(
                [
                    f"{100*item['mean_difference']:+.1f} [{100*item['ci_low']:+.1f}, {100*item['ci_high']:+.1f}]",
                    f"{item['randomization_p_holm']:.3f}",
                ]
            )
        rows.append(f"{DISPLAY[comparator]} & " + " & ".join(cells) + r" \\")
    return "\n".join(
        [
            r"\begin{table*}[t]",
            r"\centering",
            r"\small",
            r"\begin{tabular}{lrrrr}",
            r"\toprule",
            r"Comparator & \multicolumn{2}{c}{SQuAD 2.0} & \multicolumn{2}{c}{HotpotQA} \\",
            r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
            r" & $\Delta$F1 [95\% CI] & Holm $p$ & $\Delta$F1 [95\% CI] & Holm $p$ \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Paired comparisons defined as \texttt{recursive\_256} minus each comparator. We resample aligned questions 20,000 times and compute two-sided paired sign-flip tests with 100,000 draws; $p$-values are Holm-adjusted across the five post-hoc contrasts within each dataset.}",
            r"\label{tab:paired-differences}",
            r"\end{table*}",
        ]
    )


def failure_table(failure: dict) -> str:
    names = {
        "evidence_limited": "Evidence limited",
        "response_form_candidate": "Form/refusal",
        "answer_content_error": "Content mismatch",
    }
    rows = []
    for category in ("evidence_limited", "response_form_candidate", "answer_content_error"):
        cells = []
        for dataset in ("squad_v2", "hotpot_qa"):
            item = failure["datasets"][dataset]["recursive_256"]["coarse"][category]
            cells.append(
                f"{item['count']} ({item['percentage']:.1f}; "
                f"{100*item['wilson_ci_low']:.1f}--{100*item['wilson_ci_high']:.1f})"
            )
        rows.append(f"{names[category]} & " + " & ".join(cells) + r" \\")
    return "\n".join(
        [
            r"\begin{table}[t]",
            r"\centering",
            r"\scriptsize",
            r"\setlength{\tabcolsep}{2.2pt}",
            r"\begin{tabular}{lrr}",
            r"\toprule",
            r"Category & SQuAD & HotpotQA \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Corrected audit of \texttt{recursive\_256} errors (EM$=0$). Cells show count (percentage; Wilson 95\% CI). SQuAD evidence is checked after prompt truncation; HotpotQA requires both supporting documents. Form/refusal labels are hypotheses, not verified fixes.}",
            r"\label{tab:failure-audit}",
            r"\end{table}",
        ]
    )


def context_table(context: dict) -> str:
    rows = []
    for system in CHUNKERS:
        squad = context["datasets"]["squad_v2"][system]
        hotpot = context["datasets"]["hotpot_qa"][system]
        rows.append(
            f"{DISPLAY[system]} & {100*squad['corpus_chunk_over_embedding_limit_rate']:.1f} & "
            f"{100*squad['summary']['truncation_rate']:.1f} & {squad['summary']['mean_fully_consumed_chunks']:.2f} & "
            f"{100*hotpot['summary']['truncation_rate']:.1f} & {hotpot['summary']['mean_fully_consumed_chunks']:.2f} \\\\"
        )
    return "\n".join(
        [
            r"\begin{table*}[t]",
            r"\centering",
            r"\scriptsize",
            r"\setlength{\tabcolsep}{3.0pt}",
            r"\begin{tabular}{lrrrrr}",
            r"\toprule",
            r" & \multicolumn{3}{c}{SQuAD 2.0} & \multicolumn{2}{c}{HotpotQA} \\",
            r"\cmidrule(lr){2-4}\cmidrule(lr){5-6}",
            r"System & Embed cut (\%) & Prompts cut (\%) & Full chunks & Prompts cut (\%) & Full chunks \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Replayed token-budget audit. ``Embed cut'' is the percentage of corpus chunks whose special-token-inclusive MiniLM input exceeds its maximum sequence length of 256. ``Prompts cut'' reports questions whose full chat exceeds 1,024 Mistral tokens; ``full chunks'' is the mean number of top-four chunks completely retained after prefix truncation.}",
            r"\label{tab:context-audit}",
            r"\end{table*}",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_root", type=Path)
    args = parser.parse_args()
    root = args.repo_root
    output_root = root / "outputs" / "revision_audit"
    run_root = root / "outputs" / "midway_mistral_endpoint_v2"
    statistics = json.loads((output_root / "statistics.json").read_text(encoding="utf-8"))
    failure = json.loads((output_root / "failure_reanalysis.json").read_text(encoding="utf-8"))
    context = json.loads((output_root / "context_budget_audit.json").read_text(encoding="utf-8"))
    squad = json.loads((run_root / "squad_v2" / "all_summaries.json").read_text(encoding="utf-8"))
    hotpot = json.loads((run_root / "hotpot_qa" / "all_summaries.json").read_text(encoding="utf-8"))

    fragments = {
        "table_squad_results.tex": results_table("squad_v2", squad, statistics),
        "table_hotpot_results.tex": results_table("hotpot_qa", hotpot, statistics),
        "table_paired_differences.tex": paired_table(statistics),
        "table_failure_audit.tex": failure_table(failure),
        "table_context_audit.tex": context_table(context),
    }
    destination_dir = root / "reports" / "generated"
    destination_dir.mkdir(parents=True, exist_ok=True)
    for filename, content in fragments.items():
        (destination_dir / filename).write_text(
            "% Generated by scripts/generate_revision_tables.py; do not edit by hand.\n\n"
            + content
            + "\n",
            encoding="utf-8",
        )
    rendered = "\n\n".join(
        ["% Generated by scripts/generate_revision_tables.py; do not edit by hand.", *fragments.values()]
    )
    (destination_dir / "revision_tables.tex").write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
