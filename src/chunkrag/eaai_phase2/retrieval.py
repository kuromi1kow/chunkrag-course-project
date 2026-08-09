from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Sequence

from chunkrag.eaai_phase2.features import extract_pre_rerank_features
from chunkrag.retrieval import mean_reciprocal_rank_fusion
from chunkrag.schemas import Chunk


RankedChunks = list[tuple[Chunk, float]]


@dataclass(slots=True)
class PairedRetrievalResult:
    dense: RankedChunks
    bm25: RankedChunks
    fused_candidates: RankedChunks
    hybrid_top_k: RankedChunks
    reranked_candidates: RankedChunks
    reranked_top_k: RankedChunks
    features: dict[str, float | str]
    dense_latency_s: float
    bm25_latency_s: float
    fusion_latency_s: float
    hybrid_retrieval_latency_s: float
    reranker_latency_s: float


class PairedRetrievalEngine:
    """Expose paired rankings while preserving the retained retrieval algorithms."""

    def __init__(
        self,
        *,
        dense_retriever: Any,
        bm25_retriever: Any,
        cross_encoder: Any,
        chunker: str,
        candidate_pool_size: int = 20,
        final_top_k: int = 4,
        dense_weight: float = 0.6,
        bm25_weight: float = 0.4,
        rrf_k: float = 60.0,
        reranker_batch_size: int = 32,
    ) -> None:
        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.cross_encoder = cross_encoder
        self.chunker = chunker
        self.candidate_pool_size = candidate_pool_size
        self.final_top_k = final_top_k
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight
        self.rrf_k = rrf_k
        self.reranker_batch_size = reranker_batch_size

    def retrieve_pair(self, question: str) -> PairedRetrievalResult:
        started = time.perf_counter()
        dense = self.dense_retriever.retrieve(question, self.candidate_pool_size)
        dense_elapsed = time.perf_counter() - started

        started = time.perf_counter()
        bm25 = self.bm25_retriever.retrieve(question, self.candidate_pool_size)
        bm25_elapsed = time.perf_counter() - started

        started = time.perf_counter()
        fused = mean_reciprocal_rank_fusion(
            [dense, bm25],
            [self.dense_weight, self.bm25_weight],
            self.rrf_k,
        )[: self.candidate_pool_size]
        fusion_elapsed = time.perf_counter() - started
        if len(fused) < max(5, self.final_top_k):
            raise RuntimeError(
                f"Only {len(fused)} fused candidates were available for {self.chunker}"
            )
        features = extract_pre_rerank_features(
            question=question,
            chunker=self.chunker,
            dense_results=dense,
            bm25_results=bm25,
            fused_results=fused,
        )

        started = time.perf_counter()
        scores = self.cross_encoder.predict(
            [(question, chunk.text) for chunk, _ in fused],
            batch_size=self.reranker_batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        reranker_elapsed = time.perf_counter() - started
        if len(scores) != len(fused):
            raise RuntimeError("Cross-encoder score count differs from the fused candidate count")
        reranked = sorted(
            zip(fused, scores, strict=True),
            key=lambda item: float(item[1]),
            reverse=True,
        )
        reranked_candidates = [
            (chunk, float(score))
            for (chunk, _), score in reranked
        ]
        reranked_top_k = reranked_candidates[: self.final_top_k]
        hybrid_elapsed = dense_elapsed + bm25_elapsed + fusion_elapsed
        return PairedRetrievalResult(
            dense=list(dense),
            bm25=list(bm25),
            fused_candidates=list(fused),
            hybrid_top_k=list(fused[: self.final_top_k]),
            reranked_candidates=reranked_candidates,
            reranked_top_k=reranked_top_k,
            features=features,
            dense_latency_s=dense_elapsed,
            bm25_latency_s=bm25_elapsed,
            fusion_latency_s=fusion_elapsed,
            hybrid_retrieval_latency_s=hybrid_elapsed,
            reranker_latency_s=reranker_elapsed,
        )


def serialize_ranking(rows: Sequence[tuple[Chunk, float]]) -> list[dict[str, Any]]:
    return [
        {
            "rank": rank,
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.doc_id,
            "title": chunk.title,
            "token_count": int(chunk.token_count),
            "score": float(score),
        }
        for rank, (chunk, score) in enumerate(rows, start=1)
    ]
