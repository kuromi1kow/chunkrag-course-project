from __future__ import annotations

import re
from collections.abc import Iterable

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from chunkrag.schemas import Chunk


def lexical_tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class DenseRetriever:
    def __init__(
        self,
        model_name: str | None = None,
        device: str = "cpu",
        batch_size: int = 32,
        encoder: SentenceTransformer | None = None,
    ) -> None:
        if encoder is None:
            if model_name is None:
                raise ValueError("DenseRetriever requires either an encoder instance or a model name.")
            encoder = SentenceTransformer(model_name, device=device)
        self.encoder = encoder
        self.batch_size = batch_size
        self.chunks: list[Chunk] = []
        self.index: faiss.IndexFlatIP | None = None

    def build(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        if not chunks:
            self.index = None
            return
        embeddings = self.encoder.encode(
            [chunk.text for chunk in chunks],
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        embeddings = np.asarray(embeddings, dtype="float32")
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

    def retrieve(self, query: str, top_k: int) -> list[tuple[Chunk, float]]:
        if self.index is None:
            raise RuntimeError("The dense retriever index has not been built yet.")
        query_embedding = self.encoder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")
        scores, indices = self.index.search(query_embedding, top_k)
        return [(self.chunks[idx], float(score)) for idx, score in zip(indices[0], scores[0]) if idx != -1]


class BM25Retriever:
    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self.index: BM25Okapi | None = None

    def build(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        corpus_tokens = [lexical_tokenize(chunk.text) for chunk in chunks]
        self.index = BM25Okapi(corpus_tokens)

    def retrieve(self, query: str, top_k: int) -> list[tuple[Chunk, float]]:
        if self.index is None:
            raise RuntimeError("The BM25 index has not been built yet.")
        scores = np.asarray(self.index.get_scores(lexical_tokenize(query)), dtype=float)
        if len(scores) == 0:
            return []
        top_k = min(top_k, len(scores))
        candidate_indices = np.argpartition(scores, -top_k)[-top_k:]
        ranked_indices = candidate_indices[np.argsort(scores[candidate_indices])[::-1]]
        return [(self.chunks[idx], float(scores[idx])) for idx in ranked_indices]


class HybridRetriever:
    def __init__(
        self,
        dense_retriever: DenseRetriever,
        sparse_retriever: BM25Retriever,
        candidate_pool_size: int = 20,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
        rrf_k: float = 60.0,
    ) -> None:
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.candidate_pool_size = candidate_pool_size
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.rrf_k = rrf_k

    def build(self, chunks: list[Chunk]) -> None:
        self.dense_retriever.build(chunks)
        self.sparse_retriever.build(chunks)

    def retrieve(self, query: str, top_k: int) -> list[tuple[Chunk, float]]:
        candidate_k = max(top_k, self.candidate_pool_size)
        dense_results = self.dense_retriever.retrieve(query, candidate_k)
        sparse_results = self.sparse_retriever.retrieve(query, candidate_k)

        fused_scores: dict[str, float] = {}
        chunk_lookup: dict[str, Chunk] = {}
        for weight, results in ((self.dense_weight, dense_results), (self.sparse_weight, sparse_results)):
            for rank, (chunk, _) in enumerate(results, start=1):
                chunk_lookup[chunk.chunk_id] = chunk
                fused_scores.setdefault(chunk.chunk_id, 0.0)
                fused_scores[chunk.chunk_id] += weight / (self.rrf_k + rank)

        ranked_chunk_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)[:top_k]
        return [(chunk_lookup[chunk_id], fused_scores[chunk_id]) for chunk_id in ranked_chunk_ids]


class RerankRetriever:
    def __init__(
        self,
        base_retriever,
        model_name: str,
        device: str,
        candidate_pool_size: int = 20,
        batch_size: int = 16,
    ) -> None:
        self.base_retriever = base_retriever
        self.cross_encoder = CrossEncoder(model_name, device=device)
        self.candidate_pool_size = candidate_pool_size
        self.batch_size = batch_size

    def build(self, chunks: list[Chunk]) -> None:
        self.base_retriever.build(chunks)

    def retrieve(self, query: str, top_k: int) -> list[tuple[Chunk, float]]:
        candidate_k = max(top_k, self.candidate_pool_size)
        candidates = self.base_retriever.retrieve(query, candidate_k)
        if not candidates:
            return []
        pairs = [(query, chunk.text) for chunk, _ in candidates]
        scores = self.cross_encoder.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        ranked = sorted(
            zip(candidates, scores, strict=True),
            key=lambda item: float(item[1]),
            reverse=True,
        )[:top_k]
        return [(chunk, float(score)) for (chunk, _), score in ranked]


def mean_reciprocal_rank_fusion(
    result_sets: Iterable[list[tuple[Chunk, float]]],
    weights: Iterable[float],
    rrf_k: float,
) -> list[tuple[Chunk, float]]:
    fused_scores: dict[str, float] = {}
    chunk_lookup: dict[str, Chunk] = {}
    for weight, results in zip(weights, result_sets, strict=True):
        for rank, (chunk, _) in enumerate(results, start=1):
            chunk_lookup[chunk.chunk_id] = chunk
            fused_scores.setdefault(chunk.chunk_id, 0.0)
            fused_scores[chunk.chunk_id] += weight / (rrf_k + rank)
    ranked_chunk_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)
    return [(chunk_lookup[chunk_id], fused_scores[chunk_id]) for chunk_id in ranked_chunk_ids]
