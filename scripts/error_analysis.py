#!/usr/bin/env python3.11
"""Bucket Mistral v2 EM=0 predictions into failure categories and emit report artifacts.

Two bucketings are computed in parallel:

* The *fine* 7-category split exposes whether a failure is a false refusal,
  verbose extraction, terse extraction, paraphrase, partial answer, or wrong
  answer. This drives Phase 2 prioritization (different buckets need different
  fixes).
* The *coarse* 3-category roll-up groups failures into ``retrieval_failure``,
  ``format_or_refusal`` (anything fixable by prompt/post-processing), and
  ``model_error`` (genuinely wrong content). This is what the report appendix
  table reports, since the headline question is "how much of the EM gap is
  formatting vs. genuine error?".

Why F1>=0.6 alone (the previous bucketing) under-counts formatting:
when the gold answer is short ("lobes", "colloblasts"), even a perfect
extraction wrapped in a sentence drops F1 below 0.5 because the long
prediction kills precision. Token-set containment catches these cases
correctly.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# Allow running from repo root without installing the package.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from chunkrag.text_utils import normalize_answer  # noqa: E402

DATASETS = ["squad_v2", "hotpot_qa"]

FINE_ORDER = [
    "retrieval_failure",
    "false_refusal",
    "format_verbose",
    "format_terse",
    "paraphrase",
    "partial_answer",
    "wrong_answer",
]
FINE_TO_COARSE = {
    "retrieval_failure": "retrieval_failure",
    "false_refusal": "format_or_refusal",
    "format_verbose": "format_or_refusal",
    "format_terse": "format_or_refusal",
    "paraphrase": "format_or_refusal",
    "partial_answer": "model_error",
    "wrong_answer": "model_error",
}
COARSE_ORDER = ["retrieval_failure", "format_or_refusal", "model_error"]

BEST_CHUNKER: dict[str, str] = {"squad_v2": "recursive_256", "hotpot_qa": "recursive_256"}
PARAPHRASE_F1_THRESHOLD = 0.6  # used only when token-set containment doesn't trigger


def _token_set(text: str) -> set[str]:
    return set(normalize_answer(text).split())


def fine_bucket(record: dict) -> str:
    """Assign one fine-grained error category to a single EM=0 prediction record."""
    if record["supporting_doc_coverage"] == 0.0:
        return "retrieval_failure"

    pred_norm = normalize_answer(record["prediction"])
    if pred_norm in ("", "unanswerable"):
        return "false_refusal"

    pred_tokens = set(pred_norm.split())
    for gold in record.get("gold_answers", []):
        gold_tokens = _token_set(gold)
        if not gold_tokens:
            continue
        # Verbose: model produced everything in gold plus extras.
        if gold_tokens.issubset(pred_tokens) and gold_tokens != pred_tokens:
            return "format_verbose"
        # Terse: model produced a strict subset of gold tokens.
        if pred_tokens.issubset(gold_tokens) and gold_tokens != pred_tokens:
            return "format_terse"

    f1 = record["f1"]
    if f1 >= PARAPHRASE_F1_THRESHOLD:
        return "paraphrase"
    if f1 > 0.0:
        return "partial_answer"
    return "wrong_answer"


def coarse_bucket(record: dict) -> str:
    return FINE_TO_COARSE[fine_bucket(record)]


def threshold_sensitivity(em0_records: list[dict], thresholds=(0.5, 0.6, 0.7)) -> dict:
    """For diagnostic purposes only: how many retrieved-and-failed cases have F1>=t?"""
    retrieved = [r for r in em0_records if r["supporting_doc_coverage"] > 0.0]
    n = len(retrieved)
    out: dict[float, dict] = {}
    for t in thresholds:
        hi = sum(1 for r in retrieved if r["f1"] >= t)
        out[t] = {
            "above": hi,
            "below": n - hi,
            "total_retrieved": n,
            "above_pct": round(100 * hi / n, 1) if n else 0.0,
        }
    return out


def load_predictions(predictions_dir: Path) -> dict[str, dict[str, list[dict]]]:
    """Return {dataset: {chunker: [records]}}."""
    result: dict[str, dict[str, list[dict]]] = {}
    for dataset in DATASETS:
        result[dataset] = {}
        dataset_dir = predictions_dir / dataset
        if not dataset_dir.exists():
            continue
        for pred_file in sorted(dataset_dir.glob("*_predictions.json")):
            chunker = pred_file.stem.replace("_predictions", "")
            with pred_file.open() as f:
                result[dataset][chunker] = json.load(f)
    return result


def compute_summary(all_preds: dict[str, dict[str, list[dict]]]) -> list[dict]:
    rows = []
    for dataset, chunkers in all_preds.items():
        for chunker, records in chunkers.items():
            total = len(records)
            em1 = sum(1 for r in records if r["exact_match"] == 1.0)
            em0 = [r for r in records if r["exact_match"] == 0.0]
            fine: dict[str, int] = {b: 0 for b in FINE_ORDER}
            coarse: dict[str, int] = {b: 0 for b in COARSE_ORDER}
            for r in em0:
                fb = fine_bucket(r)
                fine[fb] += 1
                coarse[FINE_TO_COARSE[fb]] += 1
            row = {
                "dataset": dataset,
                "chunker": chunker,
                "total": total,
                "em_correct": em1,
                "em_zero": len(em0),
            }
            for b in FINE_ORDER:
                row[b] = fine[b]
                row[f"{b}_pct"] = round(100 * fine[b] / len(em0), 1) if em0 else 0.0
            for b in COARSE_ORDER:
                row[f"coarse_{b}"] = coarse[b]
                row[f"coarse_{b}_pct"] = round(100 * coarse[b] / len(em0), 1) if em0 else 0.0
            row["threshold_sensitivity"] = threshold_sensitivity(em0)
            rows.append(row)
    return rows


def build_appendix_table(summary_rows: list[dict]) -> str:
    """LaTeX table for the best chunker on each dataset, suitable for \\input{}.

    Uses the coarse 3-bucket roll-up (retrieval / format-or-refusal / model error).
    """
    coarse_labels = {
        "retrieval_failure": r"Retrieval failure",
        "format_or_refusal": r"Format or refusal (fixable)",
        "model_error": r"Model error (partial or wrong)",
    }
    sq = next((r for r in summary_rows if r["dataset"] == "squad_v2"
               and r["chunker"] == BEST_CHUNKER["squad_v2"]), None)
    hp = next((r for r in summary_rows if r["dataset"] == "hotpot_qa"
               and r["chunker"] == BEST_CHUNKER["hotpot_qa"]), None)

    header = (
        r"\begin{table}[h]" "\n"
        r"\centering" "\n"
        r"\caption{Failure-type distribution for EM\,=\,0 predictions on the best "
        r"chunker per dataset (\texttt{" + BEST_CHUNKER["squad_v2"] + r"}). "
        r"`Format or refusal' aggregates verbose extraction, terse extraction, "
        r"close paraphrases, and false `unanswerable' refusals: cases where the "
        r"model produced the right content but the surface form did not match "
        r"the gold answer exactly. `Model error' aggregates partial answers and "
        r"wrong answers.}" "\n"
        r"\label{tab:error-analysis}" "\n"
        r"\begin{tabular}{lrrrr}" "\n"
        r"\toprule" "\n"
        r"Failure type & \multicolumn{2}{c}{SQuAD} & \multicolumn{2}{c}{HotpotQA} \\" "\n"
        r" & $n$ & \% & $n$ & \% \\" "\n"
        r"\midrule" "\n"
    )
    body = ""
    for b in COARSE_ORDER:
        sq_n = sq[f"coarse_{b}"] if sq else 0
        sq_pct = sq[f"coarse_{b}_pct"] if sq else 0.0
        hp_n = hp[f"coarse_{b}"] if hp else 0
        hp_pct = hp[f"coarse_{b}_pct"] if hp else 0.0
        body += f"{coarse_labels[b]} & {sq_n} & {sq_pct:.1f} & {hp_n} & {hp_pct:.1f} \\\\\n"
    sq_total = sq["em_zero"] if sq else 0
    hp_total = hp["em_zero"] if hp else 0
    footer = (
        r"\midrule" "\n"
        f"Total EM\\,=\\,0 & {sq_total} & 100.0 & {hp_total} & 100.0 \\\\\n"
        r"\bottomrule" "\n"
        r"\end{tabular}" "\n"
        r"\end{table}" "\n"
    )
    return header + body + footer


def build_fine_table(summary_rows: list[dict]) -> str:
    """A second LaTeX table with the fine-grained 7-bucket breakdown, for the appendix."""
    fine_labels = {
        "retrieval_failure": r"Retrieval failure",
        "false_refusal": r"False ``unanswerable''",
        "format_verbose": r"Verbose extraction",
        "format_terse": r"Terse extraction",
        "paraphrase": r"Close paraphrase",
        "partial_answer": r"Partial answer",
        "wrong_answer": r"Wrong answer",
    }
    sq = next((r for r in summary_rows if r["dataset"] == "squad_v2"
               and r["chunker"] == BEST_CHUNKER["squad_v2"]), None)
    hp = next((r for r in summary_rows if r["dataset"] == "hotpot_qa"
               and r["chunker"] == BEST_CHUNKER["hotpot_qa"]), None)
    header = (
        r"\begin{table}[h]" "\n"
        r"\centering" "\n"
        r"\caption{Fine-grained breakdown of the same EM\,=\,0 cases.}" "\n"
        r"\label{tab:error-analysis-fine}" "\n"
        r"\begin{tabular}{lrrrr}" "\n"
        r"\toprule" "\n"
        r"Failure type & \multicolumn{2}{c}{SQuAD} & \multicolumn{2}{c}{HotpotQA} \\" "\n"
        r" & $n$ & \% & $n$ & \% \\" "\n"
        r"\midrule" "\n"
    )
    body = ""
    for b in FINE_ORDER:
        sq_n = sq[b] if sq else 0
        sq_pct = sq[f"{b}_pct"] if sq else 0.0
        hp_n = hp[b] if hp else 0
        hp_pct = hp[f"{b}_pct"] if hp else 0.0
        body += f"{fine_labels[b]} & {sq_n} & {sq_pct:.1f} & {hp_n} & {hp_pct:.1f} \\\\\n"
    footer = (
        r"\bottomrule" "\n"
        r"\end{tabular}" "\n"
        r"\end{table}" "\n"
    )
    return header + body + footer


def pick_examples(records: list[dict], target_bucket: str, n: int = 3) -> list[dict]:
    em0 = [r for r in records if r["exact_match"] == 0.0 and fine_bucket(r) == target_bucket]
    if target_bucket in ("paraphrase", "format_verbose"):
        em0.sort(key=lambda r: r["f1"], reverse=True)
    elif target_bucket == "partial_answer":
        em0.sort(key=lambda r: abs(r["f1"] - 0.35))
    elif target_bucket in ("wrong_answer", "false_refusal"):
        em0.sort(key=lambda r: r["supporting_doc_coverage"], reverse=True)
    return em0[:n]


def format_example(r: dict) -> str:
    titles = ", ".join(r.get("retrieved_titles", []))
    golds = " | ".join(r.get("gold_answers", []))
    recall = r.get("recall_at_k", 0.0)
    return (
        f"- **Q:** {r['question']}\n"
        f"  **Gold:** {golds}\n"
        f"  **Pred:** {r['prediction']}\n"
        f"  **Retrieved titles:** {titles}\n"
        f"  **F1:** {r['f1']:.3f}  **Recall@4:** {recall:.3f}\n"
    )


def build_examples_md(all_preds: dict[str, dict[str, list[dict]]]) -> str:
    lines = ["# Error analysis examples (Mistral v2 predictions, fine-grained buckets)\n"]
    for dataset in DATASETS:
        best = BEST_CHUNKER[dataset]
        records = all_preds[dataset].get(best, [])
        if not records:
            continue
        lines.append(f"## Dataset: {dataset}  Chunker: {best}\n")
        for b in FINE_ORDER:
            examples = pick_examples(records, b, n=3)
            label = b.replace("_", " ").title()
            lines.append(f"### {label} ({len(examples)} examples shown)\n")
            if not examples:
                lines.append("_No examples in this bucket._\n")
                continue
            for ex in examples:
                lines.append(format_example(ex))
            lines.append("")
    return "\n".join(lines)


def print_report(summary_rows: list[dict]) -> None:
    print("\n=== Coarse 3-bucket summary (all chunkers) ===")
    print(f"  {'dataset':12s}  {'chunker':18s}  {'EM=0':>4s}  "
          f"{'retr':>10s}  {'fmt/ref':>10s}  {'modelErr':>10s}")
    for row in summary_rows:
        if row["chunker"] == "parametric_only":
            continue
        print(
            f"  {row['dataset']:12s}  {row['chunker']:18s}  {row['em_zero']:>4d}  "
            f"{row['coarse_retrieval_failure']:>3d}({row['coarse_retrieval_failure_pct']:>4.1f}%)  "
            f"{row['coarse_format_or_refusal']:>3d}({row['coarse_format_or_refusal_pct']:>4.1f}%)  "
            f"{row['coarse_model_error']:>3d}({row['coarse_model_error_pct']:>4.1f}%)"
        )

    print("\n=== Fine 7-bucket summary (best chunker per dataset) ===")
    for dataset in DATASETS:
        best = BEST_CHUNKER[dataset]
        row = next((r for r in summary_rows if r["dataset"] == dataset
                    and r["chunker"] == best), None)
        if not row:
            continue
        print(f"\n  {dataset} / {best}  (EM=0 total: {row['em_zero']}):")
        for b in FINE_ORDER:
            print(f"    {b:18s}  n={row[b]:>3d}  ({row[f'{b}_pct']:>4.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Bucket EM=0 predictions into error categories.")
    parser.add_argument("--predictions-dir", default="outputs/midway_mistral_endpoint_v2",
                        type=Path)
    parser.add_argument("--output-dir", default="outputs/error_analysis_v2", type=Path)
    args = parser.parse_args()

    pred_dir = _REPO_ROOT / args.predictions_dir
    out_dir = _REPO_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading predictions from {pred_dir} ...")
    all_preds = load_predictions(pred_dir)
    total_files = sum(len(v) for v in all_preds.values())
    print(f"Loaded {total_files} prediction files across {len(DATASETS)} datasets.")

    summary_rows = compute_summary(all_preds)

    json_rows = [{k: v for k, v in r.items() if k != "threshold_sensitivity"}
                 for r in summary_rows]
    (out_dir / "summary.json").write_text(json.dumps(json_rows, indent=2))

    if json_rows:
        fieldnames = list(json_rows[0].keys())
        with (out_dir / "summary.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(json_rows)

    (out_dir / "appendix_table.tex").write_text(build_appendix_table(summary_rows))
    (out_dir / "appendix_table_fine.tex").write_text(build_fine_table(summary_rows))
    (out_dir / "examples.md").write_text(build_examples_md(all_preds))

    print_report(summary_rows)

    print(f"\nOutputs written to {out_dir}/")
    print("  summary.json, summary.csv, appendix_table.tex, "
          "appendix_table_fine.tex, examples.md")


if __name__ == "__main__":
    main()
