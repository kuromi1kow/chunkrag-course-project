from __future__ import annotations

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from chunkrag.schemas import Chunk


class DenseRetriever:
    def __init__(self, model_name: str, device: str, batch_size: int = 32) -> None:
        self.encoder = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size
        self.chunks: list[Chunk] = []
        self.index: faiss.IndexFlatIP | None = None

    def build(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
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
            raise RuntimeError("The retriever index has not been built yet.")
        query_embedding = self.encoder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")
        scores, indices = self.index.search(query_embedding, top_k)
        return [(self.chunks[idx], float(score)) for idx, score in zip(indices[0], scores[0]) if idx != -1]
