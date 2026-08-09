from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from chunkrag.eaai_phase2.analysis import _primary_analysis
from chunkrag.eaai_phase2.config import load_phase2_config
from chunkrag.eaai_phase2.constants import CHUNKERS, NUMERIC_FEATURES
from chunkrag.eaai_phase2.features import (
    extract_pre_rerank_features,
    validate_feature_row,
)
from chunkrag.eaai_phase2.gate import ConstantGate, fit_gate, gate_probabilities
from chunkrag.eaai_phase2.integrity import (
    repository_root,
    verify_baseline,
    verify_protocol_commit,
)
from chunkrag.eaai_phase2.io import (
    add_row_hash,
    read_json,
    validate_row_hash,
    write_immutable_json,
)
from chunkrag.eaai_phase2.partition import make_frozen_partition
from chunkrag.eaai_phase2.statistics import (
    paired_bootstrap_ci,
    paired_estimate,
    paired_sign_flip_p,
)


@dataclass
class FakeChunk:
    chunk_id: str
    text: str
    token_count: int
    doc_id: str = "doc"
    title: str = "title"


class PartitionTests(unittest.TestCase):
    def test_partition_is_stable_disjoint_and_complete(self) -> None:
        ids = [f"question-{index:03d}" for index in range(608)]
        first = make_frozen_partition(ids)
        second = make_frozen_partition(reversed(ids))
        self.assertEqual(first, second)
        self.assertEqual(len(first.development), 200)
        self.assertEqual(len(first.heldout_test), 200)
        self.assertEqual(len(first.reserve), 208)
        self.assertFalse(set(first.development) & set(first.heldout_test))
        self.assertFalse(set(first.development) & set(first.reserve))
        self.assertFalse(set(first.heldout_test) & set(first.reserve))
        self.assertEqual(
            len(set(first.development) | set(first.heldout_test) | set(first.reserve)),
            608,
        )

    def test_partition_rejects_wrong_denominator(self) -> None:
        with self.assertRaises(ValueError):
            make_frozen_partition(["a", "b"])


class ImmutableIoTests(unittest.TestCase):
    def test_immutable_json_accepts_identical_and_rejects_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "artifact.json"
            first_hash = write_immutable_json(path, {"b": 2, "a": 1})
            second_hash = write_immutable_json(path, {"a": 1, "b": 2})
            self.assertEqual(first_hash, second_hash)
            with self.assertRaises(FileExistsError):
                write_immutable_json(path, {"a": 2})

    def test_row_hash_detects_mutation(self) -> None:
        row = add_row_hash({"question_id": "q1", "f1": 0.5})
        validate_row_hash(row)
        row["f1"] = 0.6
        with self.assertRaises(ValueError):
            validate_row_hash(row)


class FeatureTests(unittest.TestCase):
    def test_frozen_feature_schema_uses_only_pre_rerank_inputs(self) -> None:
        chunks = [
            FakeChunk(f"c{index}", f"network adapter error code {index}", 100 + index)
            for index in range(30)
        ]
        dense = [(chunk, 1.0 - index / 100.0) for index, chunk in enumerate(chunks[:20])]
        bm25_order = chunks[10:30]
        bm25 = [(chunk, 20.0 - index) for index, chunk in enumerate(bm25_order)]
        fused = [(chunk, 0.02 - index / 10_000.0) for index, chunk in enumerate(chunks[:20])]
        features = extract_pre_rerank_features(
            question="How do I fix network adapter error code 7?",
            chunker="fixed_128",
            dense_results=dense,
            bm25_results=bm25,
            fused_results=fused,
        )
        validate_feature_row(features)
        self.assertEqual(set(features), {"chunker", *NUMERIC_FEATURES})
        self.assertNotIn("reranker_score", features)
        self.assertNotIn("gold_answer", features)
        self.assertAlmostEqual(float(features["dense_bm25_jaccard_at_20"]), 10 / 30)
        self.assertGreaterEqual(float(features["fused_score_entropy"]), 0.0)
        self.assertLessEqual(float(features["fused_score_entropy"]), 1.0)


def synthetic_feature_row(index: int, chunker: str) -> dict[str, float | str]:
    row: dict[str, float | str] = {"chunker": chunker}
    for feature_index, name in enumerate(NUMERIC_FEATURES):
        row[name] = float((index + 1) * (feature_index + 1)) / 100.0
    return row


class GateTests(unittest.TestCase):
    def test_logistic_gate_has_fixed_probability_threshold_inputs(self) -> None:
        rows = [synthetic_feature_row(index, CHUNKERS[index % 4]) for index in range(20)]
        labels = [index % 2 for index in range(20)]
        model, metadata = fit_gate(rows, labels)
        probabilities = gate_probabilities(model, rows)
        self.assertEqual(probabilities.shape, (20,))
        self.assertTrue(np.all((probabilities >= 0.0) & (probabilities <= 1.0)))
        self.assertEqual(metadata["threshold"], 0.5)
        self.assertEqual(metadata["model_type"], "l2_logistic_regression")
        self.assertEqual(metadata["training_rows"], 20)

    def test_single_class_fallback_is_prespecified(self) -> None:
        rows = [synthetic_feature_row(index, CHUNKERS[index % 4]) for index in range(8)]
        model, metadata = fit_gate(rows, [0] * 8)
        probabilities = gate_probabilities(model, rows)
        self.assertTrue(np.all(probabilities == 0.0))
        self.assertEqual(metadata["model_type"], "constant_fallback")


class StatisticsTests(unittest.TestCase):
    def test_paired_statistics_are_deterministic(self) -> None:
        differences = [0.2, 0.1, -0.1, 0.0, 0.3]
        first_ci = paired_bootstrap_ci(differences, draws=500, seed=7)
        second_ci = paired_bootstrap_ci(differences, draws=500, seed=7)
        self.assertEqual(first_ci, second_ci)
        first_p = paired_sign_flip_p(differences, draws=1_000, seed=8)
        second_p = paired_sign_flip_p(differences, draws=1_000, seed=8)
        self.assertEqual(first_p, second_p)
        estimate = paired_estimate(
            differences,
            bootstrap_draws=500,
            bootstrap_seed=7,
            randomization_draws=1_000,
            randomization_seed=8,
        )
        self.assertEqual(estimate.n, 5)
        self.assertEqual(estimate.positive_n, 3)
        self.assertEqual(estimate.negative_n, 1)
        self.assertEqual(estimate.tied_n, 1)

    def test_zero_effect_has_unit_p_value(self) -> None:
        self.assertEqual(paired_sign_flip_p([0.0, 0.0], draws=100, seed=1), 1.0)


class AnalysisMatrixTests(unittest.TestCase):
    @staticmethod
    def synthetic_matrices() -> tuple[list[dict], list[dict]]:
        generation_rows: list[dict] = []
        retrieval_rows: list[dict] = []
        for question_index in range(200):
            question_id = f"heldout-{question_index:03d}"
            for chunker_index, chunker in enumerate(CHUNKERS):
                features = synthetic_feature_row(question_index + chunker_index, chunker)
                retrieval_rows.append(
                    add_row_hash(
                        {
                            "question_id": question_id,
                            "chunker": chunker,
                            "features": features,
                        }
                    )
                )
                for condition in ("hybrid", "reranked"):
                    reranked = condition == "reranked"
                    generation_rows.append(
                        add_row_hash(
                            {
                                "question_id": question_id,
                                "chunker": chunker,
                                "condition": condition,
                                "f1": 0.3 if reranked else 0.2,
                                "exact_match": 1.0 if reranked and question_index % 5 == 0 else 0.0,
                                "context_truncated": bool(chunker_index % 2),
                                "retrieval_metrics": {
                                    "all_supporting_docs_found": float(not reranked),
                                    "answer_string_visible_at_k": float(reranked),
                                },
                                "timing_seconds": {
                                    "end_to_end_component_sum": 2.0 if reranked else 1.0
                                },
                            }
                        )
                    )
        return generation_rows, retrieval_rows

    def test_primary_question_aggregation_is_one_global_test(self) -> None:
        generation_rows, _ = self.synthetic_matrices()
        primary = _primary_analysis(
            generation_rows,
            input_sha256="a" * 64,
            gate_sha256="b" * 64,
            config_sha256="c" * 64,
        )
        self.assertEqual(primary["estimate"]["n"], 200)
        self.assertAlmostEqual(primary["estimate"]["mean_difference"], 0.1)
        self.assertEqual(primary["analysis_family"], "new_single_test_techqa_reranked_generation")
        self.assertFalse(primary["previous_holm_family_modified"])

    def test_secondary_adaptive_analysis_never_reranks_with_constant_zero_gate(self) -> None:
        generation_rows, retrieval_rows = self.synthetic_matrices()
        report, decisions = self._analyze_in_memory(generation_rows, retrieval_rows)
        self.assertEqual(len(decisions), 800)
        self.assertEqual(report["adaptive"]["reranker_invocation_rate"], 0.0)
        self.assertAlmostEqual(
            report["adaptive"]["question_level_system_means"]["adaptive_f1"],
            0.2,
        )

    @staticmethod
    def _analyze_in_memory(
        generation_rows: list[dict], retrieval_rows: list[dict]
    ) -> tuple[dict, list[dict]]:
        from chunkrag.eaai_phase2.analysis import (
            _adaptive_analysis,
            _chunker_effects,
            _condition_means,
            _generation_index,
            _propagation_analysis,
            _truncation_analysis,
        )

        index = _generation_index(generation_rows)
        adaptive, decisions = _adaptive_analysis(
            generation_rows,
            retrieval_rows,
            gate_model=ConstantGate(0.0),
        )
        report = {
            "adaptive": adaptive,
            "chunker_f1": _chunker_effects(index, "f1"),
            "propagation": _propagation_analysis(index),
            "truncation": _truncation_analysis(index),
        }
        return report, decisions


class FrozenConfigurationTests(unittest.TestCase):
    def test_config_and_frozen_baseline_validate(self) -> None:
        root = repository_root()
        config, config_hash = load_phase2_config(
            root / "configs" / "eaai_phase2" / "techqa_adaptive_v1.json"
        )
        self.assertEqual(config["run_id"], "techqa_adaptive_v1")
        self.assertEqual(len(config_hash), 64)
        baseline = verify_baseline(root)
        self.assertEqual(baseline.verified_files, 1322)
        self.assertEqual(len(verify_protocol_commit(root)), 64)


if __name__ == "__main__":
    unittest.main()
