from __future__ import annotations

import unittest

from chunkrag.evaluation import retrieval_metrics
from chunkrag.schemas import Chunk, QAExample


class EvaluationTests(unittest.TestCase):
    def test_retrieval_metrics_use_doc_level_recall_for_multi_doc_examples(self) -> None:
        example = QAExample(
            example_id="ex-1",
            dataset="hotpot_qa",
            question="Who?",
            answers=["yes"],
            relevant_doc_ids=["doc-a", "doc-b"],
        )
        retrieved = [
            Chunk("c1", "doc-a", "A", "hotpot_qa", "support text", 2),
            Chunk("c2", "doc-x", "X", "hotpot_qa", "distractor", 1),
            Chunk("c3", "doc-y", "Y", "hotpot_qa", "other distractor", 2),
            Chunk("c4", "doc-z", "Z", "hotpot_qa", "more distractor", 2),
        ]

        metrics = retrieval_metrics(retrieved, example)

        self.assertEqual(metrics["precision_at_k"], 0.25)
        self.assertEqual(metrics["recall_at_k"], 0.5)
        self.assertEqual(metrics["supporting_doc_coverage"], 0.5)
        self.assertEqual(metrics["all_supporting_docs_found"], 0.0)
        self.assertEqual(metrics["answer_string_visible_at_k"], 0.0)

    def test_retrieval_metrics_report_answer_string_visibility_separately(self) -> None:
        example = QAExample(
            example_id="ex-2",
            dataset="unit",
            question="What?",
            answers=["target answer"],
            relevant_doc_ids=["gold-doc"],
        )
        retrieved = [
            Chunk("c1", "gold-doc", "Gold", "unit", "unrelated part of the gold document", 6),
            Chunk("c2", "other-doc", "Other", "unit", "the target answer appears here", 5),
        ]

        metrics = retrieval_metrics(retrieved, example)

        self.assertEqual(metrics["all_supporting_docs_found"], 1.0)
        self.assertEqual(metrics["answer_string_visible_at_k"], 1.0)

    def test_answer_visibility_requires_normalized_token_boundaries(self) -> None:
        example = QAExample(
            example_id="ex-3",
            dataset="unit",
            question="Which field?",
            answers=["IT"],
            relevant_doc_ids=[],
        )
        retrieved = [
            Chunk(
                "c1",
                "doc-a",
                "A",
                "unit",
                "The internet connection is initialized.",
                5,
            )
        ]

        metrics = retrieval_metrics(retrieved, example)

        self.assertEqual(metrics["precision_at_k"], 0.0)
        self.assertEqual(metrics["answer_string_visible_at_k"], 0.0)

    def test_answer_visibility_matches_normalized_token_sequence(self) -> None:
        example = QAExample(
            example_id="ex-4",
            dataset="unit",
            question="What is the answer?",
            answers=["the target answer!"],
            relevant_doc_ids=[],
        )
        retrieved = [
            Chunk(
                "c1",
                "doc-a",
                "A",
                "unit",
                "Here, target answer appears with different punctuation.",
                7,
            )
        ]

        metrics = retrieval_metrics(retrieved, example)

        self.assertEqual(metrics["precision_at_k"], 1.0)
        self.assertEqual(metrics["answer_string_visible_at_k"], 1.0)


if __name__ == "__main__":
    unittest.main()
