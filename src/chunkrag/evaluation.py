from __future__ import annotations

import time
from statistics import mean

import numpy as np

from chunkrag.schemas import Chunk, QAExample
from chunkrag.text_utils import best_exact_match, best_f1, contains_normalized_answer


def answer_metrics(prediction: str, gold_answers: list[str]) -> dict[str, float]:
    return {
        "exact_match": best_exact_match(prediction, gold_answers),
        "f1": best_f1(prediction, gold_answers),
    }


def retrieval_metrics(retrieved_chunks: list[Chunk], example: QAExample) -> dict[str, float]:
    relevant_doc_ids = set(example.relevant_doc_ids)
    relevant_flags: list[int] = []
    for chunk in retrieved_chunks:
        is_relevant = chunk.doc_id in relevant_doc_ids
        if not is_relevant and example.answers:
            is_relevant = contains_normalized_answer(chunk.text, example.answers)
        relevant_flags.append(int(is_relevant))

    top_k = len(retrieved_chunks)
    precision = sum(relevant_flags) / top_k if top_k else 0.0
    retrieved_doc_ids = {chunk.doc_id for chunk in retrieved_chunks}
    if relevant_doc_ids:
        supporting_doc_hits = len(retrieved_doc_ids & relevant_doc_ids)
        recall = supporting_doc_hits / len(relevant_doc_ids)
        supporting_doc_coverage = recall
        all_supporting_docs_found = float(supporting_doc_hits == len(relevant_doc_ids))
    else:
        recall = float(any(relevant_flags))
        supporting_doc_coverage = 0.0
        all_supporting_docs_found = 0.0
    return {
        "precision_at_k": precision,
        "recall_at_k": recall,
        "supporting_doc_coverage": supporting_doc_coverage,
        "all_supporting_docs_found": all_supporting_docs_found,
    }


def bootstrap_confidence_interval(
    values: list[float],
    num_bootstrap_samples: int = 1_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], values[0]

    sample = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(num_bootstrap_samples, dtype=float)
    for index in range(num_bootstrap_samples):
        draw = rng.choice(sample, size=len(sample), replace=True)
        bootstrap_means[index] = draw.mean()

    alpha = (1.0 - confidence) / 2.0
    return (
        float(np.quantile(bootstrap_means, alpha)),
        float(np.quantile(bootstrap_means, 1.0 - alpha)),
    )


def summarize_metric(
    prefix: str,
    values: list[float],
    bootstrap_samples: int = 1_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, float]:
    metric_mean = mean(values) if values else 0.0
    ci_low, ci_high = bootstrap_confidence_interval(
        values,
        num_bootstrap_samples=bootstrap_samples,
        confidence=confidence,
        seed=seed,
    )
    return {
        prefix: metric_mean,
        f"{prefix}_ci_low": ci_low,
        f"{prefix}_ci_high": ci_high,
    }


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.elapsed = time.perf_counter() - self.start
