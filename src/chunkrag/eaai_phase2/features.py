from __future__ import annotations

import math
from statistics import mean, pstdev
from typing import Any, Sequence

from chunkrag.eaai_phase2.constants import CHUNKERS, NUMERIC_FEATURES


def _token_list(text: str) -> list[str]:
    import re

    return re.findall(r"\w+", text.lower())


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _normalized_entropy(scores: Sequence[float]) -> float:
    positive = [max(0.0, float(value)) for value in scores]
    total = sum(positive)
    if total <= 0.0 or len(positive) < 2:
        return 0.0
    probabilities = [value / total for value in positive if value > 0.0]
    if len(probabilities) < 2:
        return 0.0
    entropy = -sum(value * math.log(value) for value in probabilities)
    return entropy / math.log(len(positive))


def extract_pre_rerank_features(
    *,
    question: str,
    chunker: str,
    dense_results: Sequence[tuple[Any, float]],
    bm25_results: Sequence[tuple[Any, float]],
    fused_results: Sequence[tuple[Any, float]],
) -> dict[str, float | str]:
    if chunker not in CHUNKERS:
        raise ValueError(f"Unexpected chunker: {chunker}")
    if len(fused_results) < 5:
        raise ValueError("At least five fused candidates are required by the frozen feature schema")

    dense_ids = [str(chunk.chunk_id) for chunk, _ in dense_results[:20]]
    bm25_ids = [str(chunk.chunk_id) for chunk, _ in bm25_results[:20]]
    fused_top20 = list(fused_results[:20])
    fused_top4 = fused_top20[:4]
    dense_ranks = {chunk_id: rank for rank, chunk_id in enumerate(dense_ids, start=1)}
    bm25_ranks = {chunk_id: rank for rank, chunk_id in enumerate(bm25_ids, start=1)}
    missing_rank = 21
    query_token_list = _token_list(question)
    query_tokens = set(query_token_list)

    overlap_values: list[float] = []
    token_counts: list[float] = []
    for chunk, _ in fused_top4:
        overlap_values.append(
            len(query_tokens & set(_token_list(str(chunk.text)))) / max(1, len(query_tokens))
        )
        token_counts.append(float(chunk.token_count))

    fused_scores = [float(score) for _, score in fused_top20]
    top4_ids = [str(chunk.chunk_id) for chunk, _ in fused_top4]
    features: dict[str, float | str] = {
        "chunker": chunker,
        "query_token_count": float(len(query_token_list)),
        "dense_bm25_jaccard_at_20": _jaccard(set(dense_ids), set(bm25_ids)),
        "dense_bm25_jaccard_at_4": _jaccard(set(dense_ids[:4]), set(bm25_ids[:4])),
        "fused_top1_score": fused_scores[0],
        "fused_top1_top2_margin": fused_scores[0] - fused_scores[1],
        "fused_top4_top5_margin": fused_scores[3] - fused_scores[4],
        "fused_score_entropy": _normalized_entropy(fused_scores),
        "fused_top4_mean_dense_rank": mean(
            dense_ranks.get(chunk_id, missing_rank) for chunk_id in top4_ids
        ),
        "fused_top4_mean_bm25_rank": mean(
            bm25_ranks.get(chunk_id, missing_rank) for chunk_id in top4_ids
        ),
        "fused_top4_mean_query_overlap": mean(overlap_values),
        "fused_top4_max_query_overlap": max(overlap_values),
        "fused_top4_mean_chunk_tokens": mean(token_counts),
        "fused_top4_sd_chunk_tokens": pstdev(token_counts),
    }
    if tuple(key for key in features if key != "chunker") != NUMERIC_FEATURES:
        raise AssertionError("Feature implementation order differs from the frozen protocol")
    return features


def validate_feature_row(features: dict[str, Any]) -> None:
    expected = {"chunker", *NUMERIC_FEATURES}
    if set(features) != expected:
        raise ValueError(
            f"Feature fields differ from protocol: missing={expected - set(features)}, "
            f"extra={set(features) - expected}"
        )
    if features["chunker"] not in CHUNKERS:
        raise ValueError(f"Unsupported chunker feature: {features['chunker']}")
    for name in NUMERIC_FEATURES:
        value = float(features[name])
        if not math.isfinite(value):
            raise ValueError(f"Non-finite feature {name}: {value}")
