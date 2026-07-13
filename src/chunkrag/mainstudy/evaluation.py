"""Answer/evidence evaluation and TechQA judge protocol (Specification Sections 18--21)."""

from __future__ import annotations

import json
import re
import string
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from .canonical import canonical_json_hash, identifier_hash
from .constants import PROTOCOL_ID
from .schemas import validate_record


def normalize_answer(text: str) -> str:
    lowered = unicodedata.normalize("NFC", text).lower()
    no_punctuation = "".join(character for character in lowered if character not in string.punctuation)
    no_articles = re.sub(r"\b(a|an|the)\b", " ", no_punctuation)
    return " ".join(no_articles.split())


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(reference))


def token_f1(prediction: str, reference: str) -> float:
    predicted = normalize_answer(prediction).split()
    gold = normalize_answer(reference).split()
    if not predicted or not gold:
        return float(predicted == gold)
    overlap = sum((Counter(predicted) & Counter(gold)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(gold)
    return 2 * precision * recall / (precision + recall)


def best_answer_metrics(prediction: str, references: Sequence[str]) -> dict[str, float]:
    if not references:
        return {"exact_match": 0.0, "f1": 0.0}
    return {
        "exact_match": max(exact_match(prediction, reference) for reference in references),
        "f1": max(token_f1(prediction, reference) for reference in references),
    }


def interval_fully_covered(target_start: int, target_end: int, intervals: Sequence[tuple[int, int]]) -> bool:
    cursor = target_start
    for start, end in sorted(intervals):
        if end <= cursor or start > cursor:
            continue
        cursor = max(cursor, end)
        if cursor >= target_end:
            return True
    return target_start == target_end


def document_ranking(chunk_ids: Sequence[str], chunk_by_id: Mapping[str, Mapping[str, Any]]) -> list[str]:
    seen: set[str] = set()
    documents: list[str] = []
    for chunk_id in chunk_ids:
        document_id = str(chunk_by_id[chunk_id]["document_id"])
        if document_id not in seen:
            seen.add(document_id)
            documents.append(document_id)
    return documents


def document_metrics(document_ids: Sequence[str], gold_ids: Sequence[str], depth: int) -> dict[str, float]:
    ranked_ids = list(document_ids[:depth])
    gold = set(gold_ids)
    hits = gold.intersection(ranked_ids)
    coverage = len(hits) / len(gold) if gold else 0.0
    reciprocal_rank = next((1.0 / rank for rank, item in enumerate(ranked_ids, start=1) if item in gold), 0.0)
    dcg = sum((1.0 / __import__("math").log2(rank + 1)) for rank, item in enumerate(ranked_ids, start=1) if item in gold)
    ideal = sum(1.0 / __import__("math").log2(rank + 1) for rank in range(1, min(len(gold), depth) + 1))
    return {"doc_coverage": coverage, "all_hit": float(coverage == 1.0), "mrr": reciprocal_rank, "ndcg": dcg / ideal if ideal else 0.0}


TECHQA_JUDGE_SYSTEM = """You are evaluating a technical question-answering system. Judge only the candidate
answer using the question, reference answer, and consumed context. Do not reward wording
similarity by itself. Return valid JSON only."""

TECHQA_JUDGE_USER = """Question:
{question}

Reference answer:
{reference}

Consumed context:
{context}

Candidate answer:
{candidate}

Assign integer scores:

- correctness: 0 incorrect, 1 partly correct, 2 fully correct;
- completeness: 0 misses the resolution, 1 contains part of the needed resolution,
  2 contains the information needed to resolve the question;
- groundedness: 0 contains a major unsupported claim, 1 has a minor unsupported or
  unverifiable detail, 2 is fully supported by the consumed context.

Return exactly:
{{"correctness": 0, "completeness": 0, "groundedness": 0, "reason": "brief reason"}}"""


def techqa_judge_messages(question: str, reference: str, context: str, candidate: str) -> list[dict[str, str]]:
    user = TECHQA_JUDGE_USER.format(question=question, reference=reference, context=context, candidate=candidate)
    return [{"role": "system", "content": TECHQA_JUDGE_SYSTEM}, {"role": "user", "content": user}]


def techqa_judge_template_hash(model: Mapping[str, Any]) -> str:
    return canonical_json_hash({
        "prompt_version": "techqa-judge-v1", "system": TECHQA_JUDGE_SYSTEM,
        "user_template": TECHQA_JUDGE_USER, "model": dict(model),
    })


def parse_judge_json(text: str) -> dict[str, Any]:
    value = json.loads(text)
    if not isinstance(value, dict) or set(value) != {"correctness", "completeness", "groundedness", "reason"}:
        raise ValueError("Judge response has incorrect keys")
    for key in ("correctness", "completeness", "groundedness"):
        if type(value[key]) is not int or value[key] not in (0, 1, 2):
            raise ValueError(f"Invalid judge score: {key}")
    if not isinstance(value["reason"], str):
        raise ValueError("Judge reason must be a string")
    value["semantic_utility"] = (value["correctness"] + value["completeness"]) / 4
    return value


def build_evaluation_record(
    generation: Mapping[str, Any], question: Mapping[str, Any], metrics: Mapping[str, Any],
    evaluator_config_hash: str, *, judge: Mapping[str, Any] | None = None,
    human_annotation_ids: Sequence[str] = (),
) -> dict[str, Any]:
    evaluation_id = identifier_hash(generation["generation_id"], evaluator_config_hash)
    record = {
        "schema_version": PROTOCOL_ID, "evaluation_id": evaluation_id,
        "generation_id": generation["generation_id"], "references": question["references"],
        "gold_document_ids": question["gold_document_ids"], "metrics": dict(metrics),
        "judge": dict(judge or {}), "human_annotation_ids": list(human_annotation_ids),
        "evaluator_config_hash": evaluator_config_hash, "upstream_hash": generation["record_hash"],
    }
    validate_record("evaluation", record)
    return record
