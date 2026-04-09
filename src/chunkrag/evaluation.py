from __future__ import annotations

import time

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
    recall = float(any(relevant_flags))
    return {"precision_at_k": precision, "recall_at_k": recall}


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.elapsed = time.perf_counter() - self.start
