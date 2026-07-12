#!/usr/bin/env python3
"""Reanalyse EM=0 predictions without treating partial retrieval as success.

The original repository grouped every HotpotQA case with non-zero supporting
document coverage under generation errors. This script separates zero, partial,
and complete retrieved-document coverage. It also avoids calling any category
"fixable": the archived artifacts contain no prospective prompt or
post-processing intervention, and they do not record which retrieved text
survived the generator's input-token truncation.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import string
from collections import Counter
from pathlib import Path


DATASETS = ("squad_v2", "hotpot_qa")
SYSTEMS = (
    "fixed_128",
    "fixed_256",
    "fixed_512",
    "recursive_256",
    "sentence_256",
    "semantic_256",
)
FINE_ORDER = (
    "answer_string_not_visible",
    "no_support_retrieved",
    "partial_support_retrieved",
    "refusal_after_full_retrieval",
    "gold_contained_with_extras",
    "prediction_contained_in_gold",
    "high_lexical_overlap",
    "partial_lexical_overlap",
    "zero_lexical_overlap",
)
COARSE_ORDER = (
    "evidence_limited",
    "response_form_candidate",
    "answer_content_error",
)
FINE_TO_COARSE = {
    "answer_string_not_visible": "evidence_limited",
    "no_support_retrieved": "evidence_limited",
    "partial_support_retrieved": "evidence_limited",
    "refusal_after_full_retrieval": "response_form_candidate",
    "gold_contained_with_extras": "response_form_candidate",
    "prediction_contained_in_gold": "response_form_candidate",
    "high_lexical_overlap": "answer_content_error",
    "partial_lexical_overlap": "answer_content_error",
    "zero_lexical_overlap": "answer_content_error",
}


def normalize_answer(text: str) -> str:
    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def remove_punctuation(value: str) -> str:
        return "".join(character for character in value if character not in string.punctuation)

    return " ".join(remove_articles(remove_punctuation(text.lower())).split())


def token_set(text: str) -> set[str]:
    return set(normalize_answer(text).split())


def fine_bucket(record: dict, dataset: str, context_row: dict | None = None) -> str:
    if dataset == "squad_v2" and context_row is not None:
        if not bool(context_row.get("gold_answer_string_visible", False)):
            return "answer_string_not_visible"
    coverage = float(record.get("supporting_doc_coverage", 0.0))
    complete = float(record.get("all_supporting_docs_found", 0.0)) == 1.0
    if coverage == 0.0:
        return "no_support_retrieved"
    if not complete:
        return "partial_support_retrieved"

    pred_norm = normalize_answer(record.get("prediction", ""))
    if pred_norm in ("", "unanswerable"):
        return "refusal_after_full_retrieval"

    pred_tokens = set(pred_norm.split())
    for gold in record.get("gold_answers", []):
        gold_tokens = token_set(gold)
        if gold_tokens and gold_tokens < pred_tokens:
            return "gold_contained_with_extras"
        if pred_tokens and pred_tokens < gold_tokens:
            return "prediction_contained_in_gold"

    f1 = float(record.get("f1", 0.0))
    if f1 >= 0.6:
        return "high_lexical_overlap"
    if f1 > 0.0:
        return "partial_lexical_overlap"
    return "zero_lexical_overlap"


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    ) / denominator
    return centre - half, centre + half


def analyse(path: Path, dataset: str, context_rows: dict[str, dict]) -> dict:
    with path.open(encoding="utf-8") as handle:
        records = json.load(handle)
    failures = [record for record in records if float(record["exact_match"]) == 0.0]
    fine = Counter(
        fine_bucket(record, dataset, context_rows.get(record["example_id"]))
        for record in failures
    )
    coarse = Counter(
        FINE_TO_COARSE[fine_bucket(record, dataset, context_rows.get(record["example_id"]))]
        for record in failures
    )
    coarse_summary = {}
    for category in COARSE_ORDER:
        count = coarse[category]
        low, high = wilson_interval(count, len(failures))
        coarse_summary[category] = {
            "count": count,
            "percentage": 100.0 * count / len(failures) if failures else 0.0,
            "wilson_ci_low": low,
            "wilson_ci_high": high,
        }
    return {
        "total_predictions": len(records),
        "em_zero": len(failures),
        "fine": {category: fine[category] for category in FINE_ORDER},
        "coarse": coarse_summary,
    }


def write_markdown(result: dict, path: Path) -> None:
    lines = [
        "# Corrected failure reanalysis",
        "",
        "For SQuAD, the audit checks whether a normalized gold answer string survives in the "
        "reconstructed post-truncation context. For HotpotQA, evidence is considered incomplete "
        "unless both supporting documents occur among fully consumed chunks. Response-form "
        "categories remain diagnostic hypotheses rather than verified fixes.",
        "",
    ]
    for dataset in DATASETS:
        lines.extend([f"## {dataset}", ""])
        for system in SYSTEMS:
            row = result["datasets"][dataset][system]
            lines.append(f"### {system} (EM=0: {row['em_zero']})")
            lines.append("")
            lines.append("| Coarse category | n | % | Wilson 95% CI |")
            lines.append("|---|---:|---:|---:|")
            for category in COARSE_ORDER:
                item = row["coarse"][category]
                lines.append(
                    f"| {category} | {item['count']} | {item['percentage']:.1f} | "
                    f"{100*item['wilson_ci_low']:.1f}-{100*item['wilson_ci_high']:.1f} |"
                )
            lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prediction_root", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("output_markdown", type=Path)
    parser.add_argument("context_audit", type=Path)
    args = parser.parse_args()

    context_payload = json.loads(args.context_audit.read_text(encoding="utf-8"))

    result = {
        "taxonomy_note": (
            "No category is asserted to be fixable. SQuAD evidence visibility is checked in the "
            "reconstructed post-truncation context; HotpotQA requires both supporting documents."
        ),
        "datasets": {},
    }
    for dataset in DATASETS:
        result["datasets"][dataset] = {}
        for system in SYSTEMS:
            context_rows = {
                row["example_id"]: row
                for row in context_payload["datasets"][dataset][system]["per_question"]
            }
            result["datasets"][dataset][system] = analyse(
                args.prediction_root / dataset / f"{system}_predictions.json",
                dataset,
                context_rows,
            )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_markdown(result, args.output_markdown)


if __name__ == "__main__":
    main()
