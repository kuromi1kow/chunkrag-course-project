"""Canonical 25-question audit comparison (Specification Section 25 and E7)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


class ReproducibilityError(AssertionError):
    pass


def compare_retrieval(original: Mapping[str, Any], recomputed: Mapping[str, Any]) -> None:
    if original["top16_chunk_ids"] != recomputed["top16_chunk_ids"]:
        raise ReproducibilityError("Top-16 retrieved chunk IDs differ")
    original_scores = {row["chunk_id"]: float(row["reranker_score"]) for row in original["reranked_candidates"]}
    recomputed_scores = {row["chunk_id"]: float(row["reranker_score"]) for row in recomputed["reranked_candidates"]}
    if original_scores.keys() != recomputed_scores.keys():
        raise ReproducibilityError("Reranker candidate IDs differ")
    if any(abs(original_scores[key] - recomputed_scores[key]) > 1e-5 for key in original_scores):
        raise ReproducibilityError("Reranker scores exceed 1e-5 tolerance")


def compare_generation(original: Mapping[str, Any], recomputed: Mapping[str, Any]) -> None:
    if original["prompt_token_ids"] != recomputed["prompt_token_ids"]:
        raise ReproducibilityError("Prompt token IDs differ")
    if original["normalized_output"] != recomputed["normalized_output"]:
        raise ReproducibilityError("Deterministic normalized greedy output differs")


def compare_metrics(original: Mapping[str, Any], recomputed: Mapping[str, Any]) -> None:
    if original["metrics"].keys() != recomputed["metrics"].keys():
        raise ReproducibilityError("Evaluation metric keys differ")
    for key in original["metrics"]:
        if abs(float(original["metrics"][key]) - float(recomputed["metrics"][key])) > 1e-12:
            raise ReproducibilityError(f"Metric {key} exceeds 1e-12 tolerance")


def aggregate_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    if not records:
        return {}
    keys = sorted(records[0]["metrics"])
    return {key: sum(float(row["metrics"][key]) for row in records) / len(records) for key in keys}
