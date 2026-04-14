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


if __name__ == "__main__":
    unittest.main()
