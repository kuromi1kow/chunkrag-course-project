"""Blinded TechQA annotation package (Specification Section 22 and E4)."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

from .canonical import canonical_json_hash, identifier_hash
from .statistics import ordinal_krippendorff_alpha, quadratic_weighted_kappa


HUMAN_CONDITIONS = ("fixed192", "recursive192", "sentence192", "semantic192", "semantic192-jitter-1103", "gold")


def human_question_order(question_ids: Sequence[str]) -> list[str]:
    return sorted(question_ids, key=lambda item: hashlib.sha256(f"chunkrag-human-v1\0{item}".encode()).hexdigest())


def build_blinded_package(
    questions: Sequence[Mapping[str, Any]], generations_by_key: Mapping[tuple[str, str], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    selected_ids = human_question_order([str(row["question_id"]) for row in questions])[:60]
    question_by_id = {str(row["question_id"]): row for row in questions}
    records: list[dict[str, Any]] = []
    for question_index, question_id in enumerate(selected_ids):
        question = question_by_id[question_id]
        candidates: list[dict[str, Any]] = []
        for condition in HUMAN_CONDITIONS:
            generation = generations_by_key[(question_id, condition)]
            artifact_hash = canonical_json_hash(generation)
            order_hash = hashlib.sha256(f"chunkrag-human-order-v1\0{question_id}\0{artifact_hash}".encode()).hexdigest()
            annotation_id = identifier_hash("human", question_id, artifact_hash)
            candidates.append({
                "annotation_record_id": annotation_id, "order_hash": order_hash,
                "question": question["question"], "reference": question["references"][0],
                "candidate": generation["normalized_output"],
                "groundedness_subset": question_index < 10,
                "consumed_context": generation["consumed_context"] if question_index < 10 else None,
            })
        records.extend(sorted(candidates, key=lambda row: row["order_hash"]))
    if len(records) != 360:
        raise ValueError(f"Human package expected 360 records, got {len(records)}")
    return records


def build_training_package(
    questions: Sequence[Mapping[str, Any]], generations_by_key: Mapping[tuple[str, str], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    ordered = human_question_order([str(row["question_id"]) for row in questions])[60:]
    question_by_id = {str(row["question_id"]): row for row in questions}
    candidates: list[tuple[str, dict[str, Any]]] = []
    for question_id in ordered:
        question = question_by_id[question_id]
        for condition in HUMAN_CONDITIONS:
            generation = generations_by_key[(question_id, condition)]
            artifact_hash = canonical_json_hash(generation)
            record = {
                "annotation_record_id": identifier_hash("training", question_id, artifact_hash),
                "question": question["question"], "reference": question["references"][0],
                "candidate": generation["normalized_output"], "consumed_context": generation["consumed_context"],
            }
            candidates.append((artifact_hash, record))
        if len(candidates) >= 24:
            break
    return [record for _, record in sorted(candidates, key=lambda item: item[0])[:20]]


def blindness_scan(records: Sequence[Mapping[str, Any]]) -> None:
    forbidden_keys = {"model", "model_repository", "policy", "condition_id", "seed", "score", "f1"}
    for record in records:
        overlap = forbidden_keys.intersection(record)
        if overlap:
            raise ValueError(f"Blinded package leaks fields: {sorted(overlap)}")


def validate_label_rows(rows: Sequence[Mapping[str, Any]], expected_ids: set[str], *, adjudicated: bool) -> None:
    ids = [str(row.get("annotation_record_id")) for row in rows]
    if len(ids) != len(set(ids)) or set(ids) != expected_ids:
        raise ValueError("Human label IDs do not match the frozen 360-record package")
    for row in rows:
        for dimension in ("correctness", "completeness"):
            if row.get(dimension) not in (0, 1, 2, None):
                raise ValueError(f"Invalid human {dimension} label")
        if row.get("groundedness") not in (0, 1, 2, None):
            raise ValueError("Invalid human groundedness label")
        if any(row.get(key) is None for key in ("correctness", "completeness")) and not row.get("cannot_assess_reason"):
            raise ValueError("Missing answer labels require cannot_assess_reason")


def agreement_report(left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    right_by_id = {row["annotation_record_id"]: row for row in right}
    report: dict[str, Any] = {}
    for dimension in ("correctness", "completeness", "groundedness"):
        pairs = [(row[dimension], right_by_id[row["annotation_record_id"]][dimension]) for row in left if row.get(dimension) is not None and right_by_id[row["annotation_record_id"]].get(dimension) is not None]
        a = [int(pair[0]) for pair in pairs]
        b = [int(pair[1]) for pair in pairs]
        report[dimension] = {
            "n": len(pairs),
            "quadratic_weighted_kappa": quadratic_weighted_kappa(a, b),
            "ordinal_krippendorff_alpha": ordinal_krippendorff_alpha(a, b),
            "exact_agreement": sum(x == y for x, y in pairs) / len(pairs),
        }
    return report
