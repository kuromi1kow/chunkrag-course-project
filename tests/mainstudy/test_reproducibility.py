"""Specification Section 25 and E7: exact audit comparisons."""

import unittest

from chunkrag.mainstudy.reproducibility import (
    ReproducibilityError, compare_generation, compare_metrics, compare_retrieval,
)


class ReproducibilityTests(unittest.TestCase):
    def test_retrieval_tolerance(self) -> None:
        original = {"top16_chunk_ids": ["a"], "reranked_candidates": [{"chunk_id": "a", "reranker_score": 1.0}]}
        compare_retrieval(original, {"top16_chunk_ids": ["a"], "reranked_candidates": [{"chunk_id": "a", "reranker_score": 1.0 + 1e-6}]})
        with self.assertRaises(ReproducibilityError):
            compare_retrieval(original, {"top16_chunk_ids": ["b"], "reranked_candidates": [{"chunk_id": "b", "reranker_score": 1.0}]})

    def test_generation_and_metric_exactness(self) -> None:
        compare_generation({"prompt_token_ids": [1], "normalized_output": "x"}, {"prompt_token_ids": [1], "normalized_output": "x"})
        compare_metrics({"metrics": {"f1": 1.0}}, {"metrics": {"f1": 1.0 + 1e-13}})
        with self.assertRaises(ReproducibilityError):
            compare_generation({"prompt_token_ids": [1], "normalized_output": "x"}, {"prompt_token_ids": [2], "normalized_output": "x"})


if __name__ == "__main__":
    unittest.main()
