#!/usr/bin/env python3.11
"""Bucket Mistral v2 EM=0 predictions into failure categories and emit report artifacts."""

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
BUCKET_ORDER = ["retrieval_failure", "formatting_mismatch", "partial_overlap", "content_error"]
# Best chunker per dataset (by F1 in the v2 aggregate_results.json).
BEST_CHUNKER: dict[str, str] = {"squad_v2": "recursive_256", "hotpot_qa": "recursive_256"}
FORMATTING_F1_THRESHOLD = 0.6  # primary threshold used for bucketing and the report table


def bucket(record: dict) -> str:
    """Assign one error category to a single EM=0 prediction record."""
    if record["supporting_doc_coverage"] == 0.0:
        return "retrieval_failure"
    f1 = record["f1"]
    if f1 >= FORMATTING_F1_THRESHOLD:
        return "formatting_mismatch"
    if f1 > 0.0:
        return "partial_overlap"
    return "content_error"


def threshold_sensitivity(em0_records: list[dict], thresholds=(0.5, 0.6, 0.7)) -> dict:
    retrieved = [r for r in em0_records if r["supporting_doc_coverage"] > 0.0]
    n = len(retrieved)
    out: dict[float, dict] = {}
    for t in thresholds:
        fmt = sum(1 for r in retrieved if r["f1"] >= t)
        rest = n - fmt
        out[t] = {"formatting_mismatch": fmt, "other": rest, "total_retrieved": n,
                  "formatting_pct": round(100 * fmt / n, 1) if n else 0.0}
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
            buckets: dict[str, int] = {b: 0 for b in BUCKET_ORDER}
            for r in em0:
                buckets[bucket(r)] += 1
            sensitivity = threshold_sensitivity(em0)
            row = {
                "dataset": dataset, "chunker": chunker, "total": total,
                "em_correct": em1, "em_zero": len(em0),
            }
            for b in BUCKET_ORDER:
                row[b] = buckets[b]
                row[f"{b}_pct"] = round(100 * buckets[b] / len(em0), 1) if em0 else 0.0
            row["threshold_sensitivity"] = sensitivity
            rows.append(row)
    return rows


def build_appendix_table(summary_rows: list[dict]) -> str:
    """LaTeX table for the best chunker on each dataset, suitable for \input{}."""
    bucket_labels = {
        "retrieval_failure": r"Retrieval failure",
        "formatting_mismatch": r"Formatting mismatch",
        "partial_overlap": r"Partial overlap",
        "content_error": r"Content error",
    }
    # Pull rows for best chunkers.
    sq = next((r for r in summary_rows if r["dataset"] == "squad_v2"
               and r["chunker"] == BEST_CHUNKER["squad_v2"]), None)
    hp = next((r for r in summary_rows if r["dataset"] == "hotpot_qa"
               and r["chunker"] == BEST_CHUNKER["hotpot_qa"]), None)

    header = (
        r"\begin{table}[h]" "\n"
        r"\centering" "\n"
        r"\caption{Failure-type distribution for EM\,=\,0 predictions, best chunker per dataset "
        r"(\texttt{" + BEST_CHUNKER["squad_v2"] + r"}).}" "\n"
        r"\label{tab:error-analysis}" "\n"
        r"\begin{tabular}{lrrrr}" "\n"
        r"\toprule" "\n"
        r"Error type & \multicolumn{2}{c}{SQuAD} & \multicolumn{2}{c}{HotpotQA} \\" "\n"
        r" & $n$ & \% & $n$ & \% \\" "\n"
        r"\midrule" "\n"
    )
    body = ""
    for b in BUCKET_ORDER:
        sq_n = sq[b] if sq else 0
        sq_pct = sq[f"{b}_pct"] if sq else 0.0
        hp_n = hp[b] if hp else 0
        hp_pct = hp[f"{b}_pct"] if hp else 0.0
        body += f"{bucket_labels[b]} & {sq_n} & {sq_pct:.1f} & {hp_n} & {hp_pct:.1f} \\\\\n"
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


def pick_examples(records: list[dict], target_bucket: str, n: int = 3) -> list[dict]:
    em0 = [r for r in records if r["exact_match"] == 0.0 and bucket(r) == target_bucket]
    if target_bucket == "formatting_mismatch":
        em0.sort(key=lambda r: r["f1"], reverse=True)
    elif target_bucket == "partial_overlap":
        em0.sort(key=lambda r: abs(r["f1"] - 0.35))
    elif target_bucket == "content_error":
        em0.sort(key=lambda r: r["supporting_doc_coverage"], reverse=True)
    return em0[:n]


def format_example(r: dict) -> str:
    titles = ", ".join(r.get("retrieved_titles", []))
    golds = " | ".join(r.get("gold_answers", []))
    return (
        f"- **Q:** {r['question']}\n"
        f"  **Gold:** {golds}\n"
        f"  **Pred:** {r['prediction']}\n"
        f"  **Retrieved titles:** {titles}\n"
        f"  **F1:** {r['f1']:.3f}  **Recall@4:** {r.get('recall_at_k', '?'):.3f}\n"
    )


def build_examples_md(all_preds: dict[str, dict[str, list[dict]]]) -> str:
    lines = ["# Error analysis examples (v2 predictions)\n"]
    for dataset in DATASETS:
        best = BEST_CHUNKER[dataset]
        records = all_preds[dataset].get(best, [])
        if not records:
            continue
        lines.append(f"## Dataset: {dataset}  Chunker: {best}\n")
        for b in BUCKET_ORDER:
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
    print("\n=== Failure bucket summary (all chunkers) ===")
    for row in summary_rows:
        if row["chunker"] == "parametric_only":
            continue
        print(
            f"  {row['dataset']:12s}  {row['chunker']:20s}  "
            f"total={row['total']:3d}  EM=0={row['em_zero']:3d}  "
            + "  ".join(f"{b[:4]}={row[b]:2d}({row[f'{b}_pct']:4.1f}%)"
                        for b in BUCKET_ORDER)
        )
    print("\n=== Threshold sensitivity (retrieved-only, best chunkers) ===")
    for dataset in DATASETS:
        best = BEST_CHUNKER[dataset]
        row = next((r for r in summary_rows if r["dataset"] == dataset
                    and r["chunker"] == best), None)
        if not row:
            continue
        print(f"\n  {dataset} / {best}:")
        for t, counts in row["threshold_sensitivity"].items():
            print(f"    threshold={t}: formatting_mismatch={counts['formatting_mismatch']}"
                  f" ({counts['formatting_pct']:.1f}%)  other={counts['other']}"
                  f"  total_retrieved={counts['total_retrieved']}")


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

    # Drop threshold_sensitivity from JSON (it's nested, keep CSV/terminal clean).
    json_rows = [{k: v for k, v in r.items() if k != "threshold_sensitivity"}
                 for r in summary_rows]
    (out_dir / "summary.json").write_text(json.dumps(json_rows, indent=2))

    csv_path = out_dir / "summary.csv"
    if json_rows:
        fieldnames = list(json_rows[0].keys())
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(json_rows)

    appendix_tex = build_appendix_table(summary_rows)
    (out_dir / "appendix_table.tex").write_text(appendix_tex)

    examples_md = build_examples_md(all_preds)
    (out_dir / "examples.md").write_text(examples_md)

    print_report(summary_rows)

    print(f"\nOutputs written to {out_dir}/")
    print("  summary.json, summary.csv, appendix_table.tex, examples.md")


if __name__ == "__main__":
    main()
