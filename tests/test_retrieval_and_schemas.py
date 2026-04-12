from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from chunkrag.retrieval import DenseRetriever, mean_reciprocal_rank_fusion
from chunkrag.schemas import (
    AggregateMetricSummary,
    AggregateSummaryRow,
    Chunk,
    MetricSummary,
    SummaryRow,
)


class FakeEncoder:
    def __init__(self, *, fail_on_encode: bool = False) -> None:
        self.calls = 0
        self.fail_on_encode = fail_on_encode

    def encode(
        self,
        texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ):
        if self.fail_on_encode:
            raise AssertionError("Encoder should not be called when loading from cache.")
        self.calls += 1
        vectors = []
        for index, text in enumerate(texts):
            base = float(len(text) + index + 1)
            vectors.append([base, base / 2.0, 1.0])
        matrix = np.asarray(vectors, dtype="float32")
        if normalize_embeddings:
            matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix


class RetrievalAndSchemaTests(unittest.TestCase):
    def test_dense_retriever_reuses_disk_cache(self) -> None:
        chunks = [
            Chunk("chunk-1", "doc-1", "Doc 1", "unit", "alpha beta", 2),
            Chunk("chunk-2", "doc-2", "Doc 2", "unit", "gamma delta", 2),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            first_encoder = FakeEncoder()
            first = DenseRetriever(
                encoder=first_encoder,
                encoder_identifier="fake-encoder",
                cache_dir=cache_dir,
                cache_namespace="unit-cache",
            )
            first.build(chunks)
            self.assertEqual(first_encoder.calls, 1)

            second_encoder = FakeEncoder(fail_on_encode=True)
            second = DenseRetriever(
                encoder=second_encoder,
                encoder_identifier="fake-encoder",
                cache_dir=cache_dir,
                cache_namespace="unit-cache",
            )
            second.build(chunks)
            self.assertEqual(second_encoder.calls, 0)
            self.assertIsNotNone(second.index)

    def test_rrf_helper_orders_documents_by_fused_score(self) -> None:
        chunk_a = Chunk("a", "doc-a", "A", "unit", "alpha", 1)
        chunk_b = Chunk("b", "doc-b", "B", "unit", "beta", 1)
        chunk_c = Chunk("c", "doc-c", "C", "unit", "gamma", 1)

        fused = mean_reciprocal_rank_fusion(
            [
                [(chunk_a, 1.0), (chunk_b, 0.8)],
                [(chunk_b, 1.0), (chunk_c, 0.7)],
            ],
            [1.0, 1.0],
            60.0,
        )

        self.assertEqual([chunk.chunk_id for chunk, _ in fused[:3]], ["b", "a", "c"])

    def test_summary_rows_flatten_to_report_compatible_dicts(self) -> None:
        summary = SummaryRow(
            dataset="squad_v2",
            system="semantic_256",
            seed=7,
            retriever="dense",
            chunker="semantic_256",
            num_examples=8,
            metrics={"f1": MetricSummary(value=0.85, ci_low=0.75, ci_high=0.92)},
            num_chunks=12,
            avg_chunk_tokens=120.0,
        )
        aggregate = AggregateSummaryRow(
            dataset="squad_v2",
            system="semantic_256",
            retriever="dense",
            chunker="semantic_256",
            num_seeds=2,
            seed_values=[7, 9],
            aggregates={"f1": AggregateMetricSummary(mean=0.84, std=0.01, min=0.83, max=0.85)},
        )

        flat_summary = summary.to_flat_dict()
        flat_aggregate = aggregate.to_flat_dict()

        self.assertEqual(flat_summary["f1"], 0.85)
        self.assertEqual(flat_summary["f1_ci_low"], 0.75)
        self.assertEqual(flat_summary["f1_ci_high"], 0.92)
        self.assertEqual(flat_aggregate["f1_mean"], 0.84)
        self.assertEqual(flat_aggregate["f1_std"], 0.01)
        self.assertEqual(flat_aggregate["f1_min"], 0.83)
        self.assertEqual(flat_aggregate["f1_max"], 0.85)


if __name__ == "__main__":
    unittest.main()
