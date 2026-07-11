#!/usr/bin/env python3.11
"""Exploratory data analysis on top of the bucketing in error_analysis.py.

Produces eda_report.md with the following sections, designed for direct
inclusion (or paraphrase) in the final report's error-analysis discussion:

1. Headline numbers and the three-bucket coarse table.
2. Bucket distribution across ALL chunkers (does the best chunker also have
   the best 'fixable' rate?).
3. Bucket distribution by question type (what/who/when/where/how_many/yes_no/
   which/other).
4. Length statistics (gold tokens vs prediction tokens) per bucket.
5. What verbose predictions add: top phrases prepended to the gold span and
   top phrases appended after the gold span.
6. False-refusal retrieval quality (so we can claim "X% of false refusals
   had perfect retrieval").
7. Representative examples per fine bucket with one-line annotation.

Run from repo root: python3.11 scripts/error_analysis_eda.py
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from chunkrag.text_utils import normalize_answer  # noqa: E402
from error_analysis import (  # noqa: E402
    BEST_CHUNKER,
    DATASETS,
    FINE_ORDER,
    FINE_TO_COARSE,
    fine_bucket,
    load_predictions,
)


# ---------------------------------------------------------------- helpers ----

QUESTION_TYPES = ["how_many", "when", "where", "who", "what", "which", "yes_no", "other"]


def classify_question_type(question: str) -> str:
    q = question.strip().lower()
    if re.search(r"\bhow many\b|\bhow much\b", q) or q.startswith("how many"):
        return "how_many"
    if q.startswith("when") or re.search(r"\bwhat year\b|\bin what year\b", q):
        return "when"
    if q.startswith("where"):
        return "where"
    if q.startswith("who"):
        return "who"
    if q.startswith("what"):
        return "what"
    if q.startswith("which"):
        return "which"
    if re.match(r"^(was|were|is|are|did|does|do|has|have|had|can|could)\b", q):
        return "yes_no"
    return "other"


def token_recall_precision(pred: str, gold: str) -> tuple[float, float]:
    pred_tokens = Counter(normalize_answer(pred).split())
    gold_tokens = Counter(normalize_answer(gold).split())
    if sum(pred_tokens.values()) == 0 or sum(gold_tokens.values()) == 0:
        return 0.0, 0.0
    overlap = pred_tokens & gold_tokens
    shared = sum(overlap.values())
    if shared == 0:
        return 0.0, 0.0
    recall = shared / sum(gold_tokens.values())
    precision = shared / sum(pred_tokens.values())
    return recall, precision


def best_gold(record: dict) -> str:
    """Pick the gold answer that gives the highest token recall against pred."""
    pred = record["prediction"]
    best, best_r = "", -1.0
    for g in record.get("gold_answers", []):
        r, _ = token_recall_precision(pred, g)
        if r > best_r:
            best_r = r
            best = g
    return best


def bucket_aware_gold(record: dict, bucket: str) -> str:
    """Pick the gold variant that matches the bucket's classification criterion.

    For format_verbose (gold tokens are subset of pred tokens), return the gold
    that actually triggered the classification (its tokens are in pred). For
    format_terse (pred tokens are subset of gold tokens), return the gold that
    pred is a subset of (i.e. the long superset gold). Otherwise fall back to
    best_gold (highest recall).
    """
    pred_tokens = set(normalize_answer(record["prediction"]).split())
    candidates = record.get("gold_answers", [])
    if bucket == "format_verbose":
        for g in candidates:
            gt = set(normalize_answer(g).split())
            if gt and gt.issubset(pred_tokens) and gt != pred_tokens:
                return g
    elif bucket == "format_terse":
        # The triggering gold is the one whose token set CONTAINS pred's.
        # Prefer the LARGEST such gold (most informative for length contrast).
        best, best_size = None, -1
        for g in candidates:
            gt = set(normalize_answer(g).split())
            if gt and pred_tokens.issubset(gt) and gt != pred_tokens:
                if len(gt) > best_size:
                    best, best_size = g, len(gt)
        if best is not None:
            return best
    return best_gold(record)


def find_extras(pred: str, gold: str) -> tuple[list[str], list[str]]:
    """For format_verbose: tokens before and after the gold span in pred."""
    pred_norm = normalize_answer(pred)
    gold_norm = normalize_answer(gold)
    if not gold_norm or gold_norm not in pred_norm:
        return [], []
    idx = pred_norm.index(gold_norm)
    prefix = pred_norm[:idx].strip().split()
    suffix = pred_norm[idx + len(gold_norm):].strip().split()
    return prefix, suffix


# --------------------------------------------------------- section builders --


def section_headline(all_preds, summary_rows) -> str:
    out = ["# Error analysis — exploratory data analysis\n",
           "All numbers below are for Mistral 7B Instruct v0.3 on the v2 prediction "
           "set (60 SQuAD, 30 HotpotQA, dense retrieval, k=4).\n",
           "## 1. Headline coarse buckets (best chunker per dataset)\n",
           "| | SQuAD (recursive_256) | HotpotQA (recursive_256) |",
           "|---|---|---|"]
    for dataset in DATASETS:
        row = next(r for r in summary_rows
                   if r["dataset"] == dataset and r["chunker"] == BEST_CHUNKER[dataset])
    sq = next(r for r in summary_rows if r["dataset"] == "squad_v2" and r["chunker"] == BEST_CHUNKER["squad_v2"])
    hp = next(r for r in summary_rows if r["dataset"] == "hotpot_qa" and r["chunker"] == BEST_CHUNKER["hotpot_qa"])
    out.append(f"| EM correct | {sq['em_correct']} / {sq['total']} ({100*sq['em_correct']/sq['total']:.1f}%) | "
               f"{hp['em_correct']} / {hp['total']} ({100*hp['em_correct']/hp['total']:.1f}%) |")
    out.append(f"| Retrieval failure | {sq['coarse_retrieval_failure']} ({sq['coarse_retrieval_failure_pct']:.1f}% of EM=0) | "
               f"{hp['coarse_retrieval_failure']} ({hp['coarse_retrieval_failure_pct']:.1f}%) |")
    out.append(f"| Format / refusal (fixable) | {sq['coarse_format_or_refusal']} ({sq['coarse_format_or_refusal_pct']:.1f}%) | "
               f"{hp['coarse_format_or_refusal']} ({hp['coarse_format_or_refusal_pct']:.1f}%) |")
    out.append(f"| Model error | {sq['coarse_model_error']} ({sq['coarse_model_error_pct']:.1f}%) | "
               f"{hp['coarse_model_error']} ({hp['coarse_model_error_pct']:.1f}%) |")
    return "\n".join(out) + "\n"


def section_by_chunker(summary_rows) -> str:
    out = ["\n## 2. Bucket distribution across chunkers\n",
           "Does the best-F1 chunker also have the lowest fixable-error rate? "
           "(Higher fixable% = more EM=0 cases are due to format/refusal rather than "
           "genuine wrong answers.)\n"]
    for dataset in DATASETS:
        out.append(f"\n### {dataset}\n")
        out.append("| Chunker | EM correct | EM=0 total | Retrieval fail % | Fixable % | Model error % |")
        out.append("|---|---|---|---|---|---|")
        rows = [r for r in summary_rows if r["dataset"] == dataset and r["chunker"] != "parametric_only"]
        rows.sort(key=lambda r: -r["em_correct"])
        for r in rows:
            out.append(f"| {r['chunker']} | {r['em_correct']} | {r['em_zero']} | "
                       f"{r['coarse_retrieval_failure_pct']:.1f} | "
                       f"{r['coarse_format_or_refusal_pct']:.1f} | "
                       f"{r['coarse_model_error_pct']:.1f} |")
    return "\n".join(out) + "\n"


FOCUS_CHUNKERS = ["recursive_256", "semantic_256"]


def section_by_qtype(all_preds) -> str:
    out = ["\n## 3. Failure types by question type\n",
           "Which question types are hardest, and what kind of failure dominates each? "
           "Computed for the two top-performing chunkers (`recursive_256` is best by EM, "
           "`semantic_256` has a slightly higher fixable share).\n"]
    for dataset in DATASETS:
        for chunker in FOCUS_CHUNKERS:
            records = all_preds[dataset].get(chunker, [])
            if not records:
                continue
            qtype_total: dict[str, int] = Counter()
            qtype_em1: dict[str, int] = Counter()
            qtype_buckets: dict[tuple[str, str], int] = Counter()
            for r in records:
                qt = classify_question_type(r["question"])
                qtype_total[qt] += 1
                if r["exact_match"] == 1.0:
                    qtype_em1[qt] += 1
                else:
                    qtype_buckets[(qt, fine_bucket(r))] += 1

            out.append(f"\n### {dataset} / {chunker}\n")
            out.append("| Q type | n total | EM correct | False refusal | Verbose | Terse | Paraphrase | Partial | Wrong | Retr. fail |")
            out.append("|---|---|---|---|---|---|---|---|---|---|")
            for qt in QUESTION_TYPES:
                if qtype_total[qt] == 0:
                    continue
                n = qtype_total[qt]
                em1 = qtype_em1[qt]
                cells = [
                    qtype_buckets[(qt, "false_refusal")],
                    qtype_buckets[(qt, "format_verbose")],
                    qtype_buckets[(qt, "format_terse")],
                    qtype_buckets[(qt, "paraphrase")],
                    qtype_buckets[(qt, "partial_answer")],
                    qtype_buckets[(qt, "wrong_answer")],
                    qtype_buckets[(qt, "retrieval_failure")],
                ]
                out.append(f"| {qt} | {n} | {em1} ({100*em1/n:.0f}%) | "
                           + " | ".join(str(c) for c in cells) + " |")
    return "\n".join(out) + "\n"


def section_lengths(all_preds) -> str:
    out = ["\n## 4. Prediction vs gold length per bucket\n",
           "Token-level lengths after `normalize_answer`. "
           "Verbose buckets should have len(pred) >> len(gold); terse buckets the reverse. "
           "The compared gold is bucket-aware: the gold variant whose tokens are a "
           "subset of pred (for verbose), or whose tokens are a superset of pred "
           "(for terse). For other buckets we use the gold with highest token recall.\n"]
    for dataset in DATASETS:
        for chunker in FOCUS_CHUNKERS:
            records = all_preds[dataset].get(chunker, [])
            if not records:
                continue
            em0 = [r for r in records if r["exact_match"] == 0.0]
            out.append(f"\n### {dataset} / {chunker}\n")
            out.append("| Bucket | n | mean(len pred) | mean(len gold) | mean(pred − gold) |")
            out.append("|---|---|---|---|---|")
            bucket_to_records: dict[str, list[dict]] = defaultdict(list)
            for r in em0:
                bucket_to_records[fine_bucket(r)].append(r)
            for b in FINE_ORDER:
                recs = bucket_to_records.get(b, [])
                if not recs:
                    continue
                pred_lens = [len(normalize_answer(r["prediction"]).split()) for r in recs]
                gold_lens = [len(normalize_answer(bucket_aware_gold(r, b)).split()) for r in recs]
                diffs = [p - g for p, g in zip(pred_lens, gold_lens)]
                out.append(f"| {b} | {len(recs)} | "
                           f"{sum(pred_lens)/len(pred_lens):.1f} | "
                           f"{sum(gold_lens)/len(gold_lens):.1f} | "
                           f"{sum(diffs)/len(diffs):+.1f} |")
    return "\n".join(out) + "\n"


def section_verbose_patterns(all_preds) -> str:
    out = ["\n## 5. What verbose predictions add\n",
           "For every `format_verbose` case (gold tokens fully contained in pred), "
           "we extract the tokens **before** and **after** the gold span. The top "
           "patterns suggest where the model is wrapping the right answer in extra "
           "words. Counts are token-frequencies, not phrase frequencies.\n"]
    for dataset in DATASETS:
        for chunker in FOCUS_CHUNKERS:
            records = all_preds[dataset].get(chunker, [])
            if not records:
                continue
            verbose = [r for r in records
                       if r["exact_match"] == 0.0 and fine_bucket(r) == "format_verbose"]
            prefix_tokens: Counter[str] = Counter()
            suffix_tokens: Counter[str] = Counter()
            prefix_phrases: Counter[str] = Counter()
            suffix_phrases: Counter[str] = Counter()
            for r in verbose:
                gold = bucket_aware_gold(r, "format_verbose")
                pre, suf = find_extras(r["prediction"], gold)
                for tok in pre:
                    prefix_tokens[tok] += 1
                for tok in suf:
                    suffix_tokens[tok] += 1
                if pre:
                    prefix_phrases[" ".join(pre)] += 1
                if suf:
                    suffix_phrases[" ".join(suf)] += 1

            out.append(f"\n### {dataset} / {chunker} (n={len(verbose)} verbose cases)\n")
            if not verbose:
                out.append("_No format_verbose cases._\n")
                continue
            out.append("**Top prefix tokens** (model echoes question subject before gold):")
            for tok, c in prefix_tokens.most_common(8):
                out.append(f"- `{tok}` × {c}")
            out.append("\n**Top suffix tokens** (model adds clauses after gold):")
            for tok, c in suffix_tokens.most_common(8):
                out.append(f"- `{tok}` × {c}")
            out.append("\n**Sample full suffix phrases** (top 5):")
            for phrase, c in suffix_phrases.most_common(5):
                out.append(f"- `\"{phrase}\"` × {c}")
    return "\n".join(out) + "\n"


def section_false_refusal(all_preds) -> str:
    out = ["\n## 6. False-refusal retrieval quality\n",
           "How often was retrieval actually adequate when Mistral refused? "
           "If most false_refusals have `recall_at_k = 1.0`, the failure is purely "
           "a prompt-tone problem.\n"]
    for dataset in DATASETS:
        for chunker in FOCUS_CHUNKERS:
            records = all_preds[dataset].get(chunker, [])
            if not records:
                continue
            refusals = [r for r in records
                        if r["exact_match"] == 0.0 and fine_bucket(r) == "false_refusal"]
            out.append(f"\n### {dataset} / {chunker}\n")
            if not refusals:
                out.append("_No false_refusal cases._\n")
                continue
            recall_dist = Counter(round(r["recall_at_k"], 2) for r in refusals)
            sd_cov_dist = Counter(round(r["supporting_doc_coverage"], 2) for r in refusals)
            perfect = sum(1 for r in refusals if r["recall_at_k"] == 1.0)
            out.append(f"- n={len(refusals)} refusals")
            out.append(f"- **{perfect}/{len(refusals)} ({100*perfect/len(refusals):.0f}%)** had `recall_at_k = 1.0` (every gold doc was in the prompt).")
            out.append(f"- recall_at_k distribution: " + ", ".join(
                f"{v}: {c}" for v, c in sorted(recall_dist.items())))
            out.append(f"- supporting_doc_coverage distribution: " + ", ".join(
                f"{v}: {c}" for v, c in sorted(sd_cov_dist.items())))
            qtype_counts = Counter(classify_question_type(r["question"]) for r in refusals)
            out.append("- Question types of refusals: " + ", ".join(
                f"{qt}: {c}" for qt, c in qtype_counts.most_common()))
    return "\n".join(out) + "\n"


def section_examples(all_preds) -> str:
    out = ["\n## 7. Representative examples per bucket (annotated)\n",
           "Each example is followed by a one-line note explaining why it failed "
           "and what fix would help.\n"]

    annotations = {
        "retrieval_failure": "Wrong document retrieved; chunking/embedding can't recover the correct doc.",
        "false_refusal": "Retrieval recovered the gold doc(s) but Mistral refused. Soften the unanswerable instruction.",
        "format_verbose": "Gold span is in the prediction, surrounded by extra words. Tighten prompt to forbid subject echoing/clauses.",
        "format_terse": "Prediction is a strict subset of the gold tokens — model gave less than the gold required.",
        "paraphrase": "High token overlap but neither subset; close paraphrase that EM can't credit.",
        "partial_answer": "Some token overlap with gold but not subset; model partially understood.",
        "wrong_answer": "Zero token overlap; model genuinely produced an unrelated answer.",
    }

    for dataset in DATASETS:
        records = all_preds[dataset][BEST_CHUNKER[dataset]]
        em0 = [r for r in records if r["exact_match"] == 0.0]
        out.append(f"\n### {dataset} ({BEST_CHUNKER[dataset]})\n")
        for b in FINE_ORDER:
            cases = [r for r in em0 if fine_bucket(r) == b]
            if not cases:
                continue
            # Pick one representative example
            if b in ("format_verbose", "paraphrase"):
                cases.sort(key=lambda r: r["f1"], reverse=True)
            elif b == "false_refusal":
                cases.sort(key=lambda r: -r["recall_at_k"])
            else:
                cases.sort(key=lambda r: -r["supporting_doc_coverage"])
            r = cases[0]
            out.append(f"**{b}** ({len(cases)} cases). _{annotations[b]}_")
            out.append(f"- Q: {r['question']}")
            out.append(f"- Gold: {' | '.join(r['gold_answers'])}")
            out.append(f"- Pred: {r['prediction']}")
            out.append(f"- F1={r['f1']:.2f}, recall@4={r['recall_at_k']:.2f}\n")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------- driver -----


def main() -> None:
    pred_dir = _REPO_ROOT / "outputs" / "midway_mistral_endpoint_v2"
    out_dir = _REPO_ROOT / "outputs" / "error_analysis_v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading predictions from {pred_dir} ...")
    all_preds = load_predictions(pred_dir)

    summary_path = out_dir / "summary.json"
    summary_rows = json.loads(summary_path.read_text())

    sections = [
        section_headline(all_preds, summary_rows),
        section_by_chunker(summary_rows),
        section_by_qtype(all_preds),
        section_lengths(all_preds),
        section_verbose_patterns(all_preds),
        section_false_refusal(all_preds),
        section_examples(all_preds),
    ]
    report = "\n".join(sections)
    eda_path = out_dir / "eda_report.md"
    eda_path.write_text(report)
    print(f"Wrote EDA report to {eda_path}")
    print(f"({len(report.splitlines())} lines, {len(report)} chars)")


if __name__ == "__main__":
    main()
