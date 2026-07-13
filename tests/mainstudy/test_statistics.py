"""Specification Section 20: synthetic cluster-aware inference tests."""

import math
import unittest

from chunkrag.mainstudy.determinism import derived_seed
from chunkrag.mainstudy.statistics import (
    cliffs_delta, cluster_bootstrap, cluster_sign_flip, holm_adjust, quadratic_weighted_kappa,
    rank_biserial, spearman,
)


class StatisticsTests(unittest.TestCase):
    def test_seed_is_order_independent(self) -> None:
        self.assertEqual(derived_seed("H1:squad:x", "cluster-bootstrap"), derived_seed("H1:squad:x", "cluster-bootstrap"))
        self.assertNotEqual(derived_seed("H1:squad:x", "a"), derived_seed("H1:squad:x", "b"))

    def test_holm_monotone(self) -> None:
        adjusted = holm_adjust({"a": 0.01, "b": 0.02, "c": 0.5})
        self.assertEqual(adjusted, {"a": 0.03, "b": 0.04, "c": 0.5})

    def test_effect_sizes(self) -> None:
        self.assertEqual(rank_biserial([1, 2, -1, 0]), 1 / 3)
        self.assertEqual(cliffs_delta([3, 4], [1, 2]), 1.0)

    def test_bootstrap_reproducible(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0]
        clusters = ["a", "a", "b", "c"]
        left = cluster_bootstrap(values, clusters, "test", draws=200)
        right = cluster_bootstrap(values, clusters, "test", draws=200)
        self.assertEqual(left, right)

    def test_exact_sign_flip(self) -> None:
        p_value = cluster_sign_flip([1.0, 1.0], ["a", "b"], "test")
        self.assertEqual(p_value, 0.5)

    def test_judge_agreement(self) -> None:
        self.assertAlmostEqual(spearman([0, 1, 2], [0, 1, 2]), 1.0)
        self.assertAlmostEqual(quadratic_weighted_kappa([0, 1, 2], [0, 1, 2]), 1.0)
        self.assertTrue(math.isnan(spearman([1, 1, 1], [0, 1, 2])))


if __name__ == "__main__":
    unittest.main()
