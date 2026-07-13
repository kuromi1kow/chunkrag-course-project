"""Score-complete retrieval traces (Specification Sections 13, 18.4, 23.3, E1/E5)."""

from __future__ import annotations

import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Callable

from .canonical import identifier_hash
from .constants import PROTOCOL_ID
from .schemas import validate_record


def lexical_tokenize(text: str) -> list[str]:
    return re.findall(r"(?u)\b\w+\b", text.lower())


def ranked(scores: Mapping[str, float], limit: int) -> list[dict[str, Any]]:
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]
    return [{"chunk_id": chunk_id, "score": float(score), "rank": index} for index, (chunk_id, score) in enumerate(ordered, start=1)]


def weighted_rrf(
    dense: Sequence[Mapping[str, Any]], sparse: Sequence[Mapping[str, Any]],
    *, dense_weight: float = 0.6, sparse_weight: float = 0.4, constant: int = 60,
) -> list[dict[str, Any]]:
    components: dict[str, dict[str, Any]] = {}
    for name, rows, weight in (("dense", dense, dense_weight), ("sparse", sparse, sparse_weight)):
        for row in rows:
            chunk_id = str(row["chunk_id"])
            component = components.setdefault(chunk_id, {"chunk_id": chunk_id})
            component[f"{name}_rank"] = int(row["rank"])
            component[f"{name}_score"] = float(row["score"])
            component["fused_score"] = component.get("fused_score", 0.0) + weight / (constant + int(row["rank"]))
    ordered = sorted(components.values(), key=lambda row: (-row["fused_score"], row["chunk_id"]))
    for index, row in enumerate(ordered, start=1):
        row["rank"] = index
        row.setdefault("dense_rank", None)
        row.setdefault("dense_score", None)
        row.setdefault("sparse_rank", None)
        row.setdefault("sparse_score", None)
    return ordered


def rerank_candidates(
    fused: Sequence[Mapping[str, Any]], scores: Mapping[str, float], limit: int = 50,
) -> list[dict[str, Any]]:
    rows = [{**dict(row), "reranker_score": float(scores[str(row["chunk_id"])])} for row in fused[:limit]]
    rows.sort(key=lambda row: (-row["reranker_score"], -row["fused_score"], row["chunk_id"]))
    for rank_value, row in enumerate(rows, start=1):
        row["rank"] = rank_value
    return rows


def build_retrieval_record(
    *, question_id: str, condition_id: str, question_manifest_hash: str,
    corpus_manifest_hash: str, dense: list[dict[str, Any]], sparse: list[dict[str, Any]],
    fused: list[dict[str, Any]], reranked: list[dict[str, Any]], config_hash: str,
    upstream_hash: str, latency: Mapping[str, float], memory: Mapping[str, int],
) -> dict[str, Any]:
    retrieval_id = identifier_hash(PROTOCOL_ID, question_id, condition_id, config_hash)
    record = {
        "schema_version": PROTOCOL_ID, "retrieval_id": retrieval_id,
        "question_id": question_id, "condition_id": condition_id,
        "question_manifest_hash": question_manifest_hash, "corpus_manifest_hash": corpus_manifest_hash,
        "dense_candidates": dense, "sparse_candidates": sparse,
        "fused_candidates": fused[:50], "reranked_candidates": reranked[:50],
        "top16_chunk_ids": [row["chunk_id"] for row in reranked[:16]],
        "latency": dict(latency), "memory": dict(memory), "config_hash": config_hash,
        "upstream_hash": upstream_hash,
    }
    validate_record("retrieval", record)
    return record


@dataclass(slots=True)
class PrimaryRetriever:
    """Lazy real-model adapter. Construction is allowed in Phase 3; ``build``/``query`` are Phase 4."""

    chunk_records: list[Mapping[str, Any]]
    dense_repository: str
    dense_revision: str
    reranker_repository: str
    reranker_revision: str
    query_prefix: str
    device: str
    _model: Any = None
    _reranker: Any = None
    _reranker_tokenizer: Any = None
    _dense_index: Any = None
    _bm25: Any = None
    _chunks_by_id: dict[str, Mapping[str, Any]] | None = None

    def build(self) -> None:
        import faiss
        import numpy as np
        from rank_bm25 import BM25Okapi
        from sentence_transformers import SentenceTransformer
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._chunks_by_id = {str(row["chunk_id"]): row for row in self.chunk_records}
        self._model = SentenceTransformer(self.dense_repository, revision=self.dense_revision, device=self.device, local_files_only=True)
        self._model.max_seq_length = 512
        texts = [str(row["text"]) for row in self.chunk_records]
        dense_tokenizer = self._model.tokenizer
        for row, text in zip(self.chunk_records, texts):
            count = len(dense_tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])
            if count > 512:
                raise RuntimeError(f"Dense encoder would truncate chunk {row['chunk_id']}: {count}")
        vectors = self._model.encode(texts, batch_size=64, normalize_embeddings=True, convert_to_numpy=True).astype("float32")
        if vectors.shape[0] != len(texts):
            raise RuntimeError("Dense encoder returned wrong record count")
        self._dense_index = faiss.IndexFlatIP(vectors.shape[1])
        self._dense_index.add(vectors)
        self._bm25 = BM25Okapi([lexical_tokenize(text) for text in texts])
        self._reranker_tokenizer = AutoTokenizer.from_pretrained(self.reranker_repository, revision=self.reranker_revision, local_files_only=True)
        self._reranker = AutoModelForSequenceClassification.from_pretrained(self.reranker_repository, revision=self.reranker_revision, local_files_only=True).to(self.device)
        self._reranker.eval()

    def query(self, question: str, *, stack: str = "hybrid-rerank") -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, float]]:
        import numpy as np

        if self._dense_index is None or self._chunks_by_id is None:
            raise RuntimeError("Retriever must be built before query")
        dense_query_tokens = len(self._model.tokenizer(self.query_prefix + question, add_special_tokens=True, truncation=False)["input_ids"])
        if dense_query_tokens > 512:
            raise RuntimeError(f"Dense encoder would truncate question: {dense_query_tokens}")
        started = time.perf_counter()
        vector = self._model.encode([self.query_prefix + question], batch_size=64, normalize_embeddings=True, convert_to_numpy=True).astype("float32")
        dense_scores, dense_indices = self._dense_index.search(vector, 50)
        dense = [{"chunk_id": self.chunk_records[int(index)]["chunk_id"], "score": float(score), "rank": rank_value} for rank_value, (score, index) in enumerate(zip(dense_scores[0], dense_indices[0]), start=1) if int(index) >= 0]
        dense_elapsed = time.perf_counter() - started
        started = time.perf_counter()
        sparse_values = self._bm25.get_scores(lexical_tokenize(question))
        sparse = ranked({str(row["chunk_id"]): float(sparse_values[index]) for index, row in enumerate(self.chunk_records)}, 50)
        sparse_elapsed = time.perf_counter() - started
        fused = weighted_rrf(dense, sparse)[:50]
        if stack in ("dense", "hybrid"):
            selected = dense if stack == "dense" else fused
            reranked = [{**row, "reranker_score": row["score"] if "score" in row else row["fused_score"]} for row in selected]
            return dense, sparse, fused, reranked, {"dense_seconds": dense_elapsed, "sparse_seconds": sparse_elapsed, "reranker_seconds": 0.0}
        if stack != "hybrid-rerank":
            raise ValueError(f"Unknown retrieval stack: {stack}")
        started = time.perf_counter()
        import torch
        reranker_values: list[float] = []
        token_audits: dict[str, dict[str, int]] = {}
        for start_index in range(0, len(fused), 32):
            batch_rows = fused[start_index:start_index + 32]
            passages = [str(self._chunks_by_id[row["chunk_id"]]["text"]) for row in batch_rows]
            question_tokens = len(self._reranker_tokenizer(question, add_special_tokens=False, truncation=False)["input_ids"])
            if question_tokens + self._reranker_tokenizer.num_special_tokens_to_add(pair=True) >= 512:
                raise RuntimeError("Reranker question cannot fit without truncation")
            encoded = self._reranker_tokenizer(
                [question] * len(passages), passages, padding=True, truncation="only_second",
                max_length=512, return_tensors="pt",
            )
            with torch.inference_mode():
                logits = self._reranker(**{key: value.to(self.device) for key, value in encoded.items()}).logits.squeeze(-1)
            reranker_values.extend(float(value) for value in logits.detach().cpu().tolist())
            for row, passage, retained in zip(batch_rows, passages, encoded["attention_mask"].sum(dim=1).tolist()):
                token_audits[row["chunk_id"]] = {
                    "question_tokens": question_tokens,
                    "passage_tokens_original": len(self._reranker_tokenizer(passage, add_special_tokens=False, truncation=False)["input_ids"]),
                    "pair_tokens_retained": int(retained),
                }
        reranked = rerank_candidates(fused, {row["chunk_id"]: float(score) for row, score in zip(fused, reranker_values)})
        for row in reranked:
            row.update(token_audits[row["chunk_id"]])
        rerank_elapsed = time.perf_counter() - started
        return dense, sparse, fused, reranked, {"dense_seconds": dense_elapsed, "sparse_seconds": sparse_elapsed, "reranker_seconds": rerank_elapsed}
