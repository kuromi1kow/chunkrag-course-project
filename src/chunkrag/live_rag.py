from __future__ import annotations

import io
import json
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from chunkrag.chunking import count_tokens, sentence_split
from chunkrag.generation import ExtractiveFallbackGenerator
from chunkrag.pipeline import build_chunks
from chunkrag.retrieval import BM25Retriever, DenseRetriever, HybridRetriever, RerankRetriever, lexical_tokenize
from chunkrag.schemas import Chunk, Document


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


@dataclass(slots=True)
class UploadPayload:
    name: str
    data: bytes


@dataclass(slots=True)
class RetrievedEvidence:
    chunk: Chunk
    score: float
    raw_score: float
    matched_queries: list[str] = field(default_factory=list)
    compressed_text: str = ""
    selected_sentences: list[str] = field(default_factory=list)
    support_score: float = 0.0
    compression_ratio: float = 1.0


@dataclass(slots=True)
class DemoIndex:
    documents: list[Document]
    chunks: list[Chunk]
    retriever: Any
    tokenizer: PreTrainedTokenizerBase
    encoder: SentenceTransformer
    embedding_model: str
    device: str
    chunker_spec: dict[str, Any]
    retriever_spec: dict[str, Any]


def _sanitize_identifier(value: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return clean or "document"


def _decode_text(data: bytes) -> str:
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="ignore")


def _documents_from_pdf(payload: UploadPayload) -> list[Document]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Install the demo extras to upload PDF files.") from exc

    reader = PdfReader(io.BytesIO(payload.data))
    text = "\n\n".join((page.extract_text() or "").strip() for page in reader.pages).strip()
    title = Path(payload.name).stem
    if not text:
        return []
    return [
        Document(
            doc_id=f"upload::{_sanitize_identifier(title)}",
            title=title,
            text=text,
            dataset="uploaded",
        )
    ]


def _documents_from_text(payload: UploadPayload) -> list[Document]:
    title = Path(payload.name).stem
    text = _decode_text(payload.data).strip()
    if not text:
        return []
    return [
        Document(
            doc_id=f"upload::{_sanitize_identifier(title)}",
            title=title,
            text=text,
            dataset="uploaded",
        )
    ]


def _documents_from_csv(payload: UploadPayload) -> list[Document]:
    suffix = Path(payload.name).suffix.lower()
    separator = "\t" if suffix == ".tsv" else ","
    frame = pd.read_csv(io.BytesIO(payload.data), sep=separator)
    text_column = "text" if "text" in frame.columns else None
    title_column = "title" if "title" in frame.columns else None
    documents: list[Document] = []
    base_name = _sanitize_identifier(Path(payload.name).stem)
    for index, row in frame.fillna("").iterrows():
        if text_column is not None:
            text = str(row[text_column]).strip()
        else:
            text = " ".join(f"{column}: {row[column]}" for column in frame.columns if str(row[column]).strip())
        if not text:
            continue
        title = str(row[title_column]).strip() if title_column is not None else f"{Path(payload.name).stem} row {index + 1}"
        documents.append(
            Document(
                doc_id=f"upload::{base_name}::{index}",
                title=title,
                text=text,
                dataset="uploaded",
            )
        )
    return documents


def _documents_from_json(payload: UploadPayload) -> list[Document]:
    parsed = json.loads(_decode_text(payload.data))
    base_name = _sanitize_identifier(Path(payload.name).stem)
    documents: list[Document] = []

    if isinstance(parsed, list):
        for index, item in enumerate(parsed):
            if isinstance(item, dict):
                title = str(item.get("title") or f"{Path(payload.name).stem} item {index + 1}")
                text = str(item.get("text") or item.get("content") or "").strip()
                if not text:
                    text = json.dumps(item, ensure_ascii=False, indent=2)
            else:
                title = f"{Path(payload.name).stem} item {index + 1}"
                text = str(item).strip()
            if not text:
                continue
            documents.append(
                Document(
                    doc_id=f"upload::{base_name}::{index}",
                    title=title,
                    text=text,
                    dataset="uploaded",
                )
            )
        return documents

    if isinstance(parsed, dict):
        if "text" in parsed or "content" in parsed:
            title = str(parsed.get("title") or Path(payload.name).stem)
            text = str(parsed.get("text") or parsed.get("content") or "").strip()
            if text:
                return [
                    Document(
                        doc_id=f"upload::{base_name}",
                        title=title,
                        text=text,
                        dataset="uploaded",
                    )
                ]
        return [
            Document(
                doc_id=f"upload::{base_name}",
                title=Path(payload.name).stem,
                text=json.dumps(parsed, ensure_ascii=False, indent=2),
                dataset="uploaded",
            )
        ]

    return _documents_from_text(payload)


def load_documents_from_uploads(payloads: list[UploadPayload]) -> list[Document]:
    documents: list[Document] = []
    for payload in payloads:
        suffix = Path(payload.name).suffix.lower()
        if suffix == ".pdf":
            documents.extend(_documents_from_pdf(payload))
        elif suffix in {".txt", ".md", ".markdown", ".rst"}:
            documents.extend(_documents_from_text(payload))
        elif suffix in {".csv", ".tsv"}:
            documents.extend(_documents_from_csv(payload))
        elif suffix == ".json":
            documents.extend(_documents_from_json(payload))
        else:
            documents.extend(_documents_from_text(payload))
    return documents


def _make_chunker_spec(
    *,
    chunker_type: str,
    chunk_size: int,
    chunk_overlap: int = 0,
    similarity_threshold: float = 0.72,
) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "name": f"{chunker_type}_{chunk_size}",
        "type": chunker_type,
        "chunk_size": chunk_size,
    }
    if chunker_type in {"fixed", "recursive"}:
        spec["chunk_overlap"] = chunk_overlap
    if chunker_type == "semantic":
        spec["similarity_threshold"] = similarity_threshold
    return spec


def _materialize_retriever(
    chunks: list[Chunk],
    encoder: SentenceTransformer,
    *,
    device: str,
    retriever_name: str,
    retrieval_top_k: int,
) -> tuple[Any, dict[str, Any]]:
    if retriever_name == "dense":
        dense = DenseRetriever(encoder=encoder, device=device, batch_size=32)
        dense.build(chunks)
        return dense, {"name": "dense", "type": "dense"}

    if retriever_name == "bm25":
        sparse = BM25Retriever()
        sparse.build(chunks)
        return sparse, {"name": "bm25", "type": "bm25"}

    dense = DenseRetriever(encoder=encoder, device=device, batch_size=32)
    dense.build(chunks)
    sparse = BM25Retriever()
    sparse.build(chunks)
    hybrid = HybridRetriever(
        dense_retriever=dense,
        sparse_retriever=sparse,
        candidate_pool_size=max(retrieval_top_k * 4, 12),
        dense_weight=0.55,
        sparse_weight=0.45,
        rrf_k=60.0,
    )

    if retriever_name == "hybrid":
        return hybrid, {"name": "hybrid", "type": "hybrid"}

    if retriever_name == "hybrid_rerank":
        reranker = RerankRetriever(
            base_retriever=hybrid,
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
            device=device,
            candidate_pool_size=max(retrieval_top_k * 4, 12),
            batch_size=16,
        )
        return reranker, {"name": "hybrid_rerank", "type": "rerank", "base_retriever": {"type": "hybrid"}}

    raise ValueError(f"Unsupported retriever: {retriever_name}")


def build_demo_index(
    documents: list[Document],
    *,
    embedding_model: str,
    chunker_type: str,
    chunk_size: int,
    chunk_overlap: int = 32,
    similarity_threshold: float = 0.72,
    retriever_name: str = "hybrid",
    retrieval_top_k: int = 4,
    device: str = "cpu",
) -> DemoIndex:
    if not documents:
        raise ValueError("At least one document is required to build a demo index.")
    tokenizer = AutoTokenizer.from_pretrained(embedding_model)
    tokenizer.model_max_length = 1_000_000
    encoder = SentenceTransformer(embedding_model, device=device)
    chunker_spec = _make_chunker_spec(
        chunker_type=chunker_type,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        similarity_threshold=similarity_threshold,
    )
    chunks = build_chunks(documents, chunker_spec, tokenizer, encoder)
    if not chunks:
        raise ValueError("The configured chunker did not produce any chunks for the provided documents.")
    retriever, retriever_spec = _materialize_retriever(
        chunks,
        encoder,
        device=device,
        retriever_name=retriever_name,
        retrieval_top_k=retrieval_top_k,
    )
    return DemoIndex(
        documents=documents,
        chunks=chunks,
        retriever=retriever,
        tokenizer=tokenizer,
        encoder=encoder,
        embedding_model=embedding_model,
        device=device,
        chunker_spec=chunker_spec,
        retriever_spec=retriever_spec,
    )


def _unique_in_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        clean = value.strip()
        if not clean or clean in seen:
            continue
        unique.append(clean)
        seen.add(clean)
    return unique


def generate_query_variants(question: str, max_variants: int = 4) -> list[str]:
    variants = [question.strip()]
    keywords = [token for token in lexical_tokenize(question) if token not in STOPWORDS]
    if keywords:
        variants.append(" ".join(keywords[: min(len(keywords), 12)]))
    capitalized_phrases = re.findall(r"(?:[A-Z][\w/-]*\s*){1,4}", question)
    if capitalized_phrases:
        variants.append(" ".join(_unique_in_order(capitalized_phrases)[:2]))
    if len(keywords) >= 6:
        midpoint = len(keywords) // 2
        variants.append(" ".join(keywords[:midpoint]))
        variants.append(" ".join(keywords[midpoint:]))
    return _unique_in_order(variants)[:max_variants]


def plan_subqueries(question: str, max_subqueries: int = 3) -> list[str]:
    query_variants = generate_query_variants(question, max_variants=max_subqueries + 1)
    if len(query_variants) > 1:
        return query_variants[1 : max_subqueries + 1]
    keywords = [token for token in lexical_tokenize(question) if token not in STOPWORDS]
    if len(keywords) <= 4:
        return []
    window = max(2, len(keywords) // max_subqueries)
    return _unique_in_order(
        " ".join(keywords[index : index + window]) for index in range(0, len(keywords), window)
    )[:max_subqueries]


def retrieve_for_queries(
    retriever: Any,
    queries: list[str],
    *,
    top_k: int,
    per_query_k: int,
    rrf_k: float = 60.0,
) -> list[RetrievedEvidence]:
    fused_scores: dict[str, float] = defaultdict(float)
    raw_scores: dict[str, float] = defaultdict(float)
    matched_queries: dict[str, set[str]] = defaultdict(set)
    chunk_lookup: dict[str, Chunk] = {}

    for query in _unique_in_order(queries):
        results = retriever.retrieve(query, per_query_k)
        for rank, (chunk, score) in enumerate(results, start=1):
            chunk_lookup[chunk.chunk_id] = chunk
            fused_scores[chunk.chunk_id] += 1.0 / (rrf_k + rank)
            raw_scores[chunk.chunk_id] = max(raw_scores[chunk.chunk_id], float(score))
            matched_queries[chunk.chunk_id].add(query)

    ranked_chunk_ids = sorted(
        fused_scores,
        key=lambda chunk_id: (fused_scores[chunk_id], raw_scores[chunk_id]),
        reverse=True,
    )[:top_k]
    return [
        RetrievedEvidence(
            chunk=chunk_lookup[chunk_id],
            score=fused_scores[chunk_id],
            raw_score=raw_scores[chunk_id],
            matched_queries=sorted(matched_queries[chunk_id]),
        )
        for chunk_id in ranked_chunk_ids
    ]


def _sentence_scores(
    query: str,
    sentences: list[str],
    encoder: SentenceTransformer,
) -> np.ndarray:
    if not sentences:
        return np.zeros(0, dtype=float)
    embeddings = encoder.encode(
        [query, *sentences],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    query_embedding = embeddings[0]
    sentence_embeddings = embeddings[1:]
    return sentence_embeddings @ query_embedding


def _compress_evidence_text(
    query: str,
    text: str,
    *,
    encoder: SentenceTransformer,
    tokenizer: PreTrainedTokenizerBase,
    token_budget: int,
    max_sentences: int,
) -> tuple[str, list[str], float, float]:
    sentences = sentence_split(text)
    if len(sentences) <= 1:
        clean_text = text.strip()
        return clean_text, [clean_text] if clean_text else [], 1.0, 1.0

    scores = _sentence_scores(query, sentences, encoder)
    ranked_indices = sorted(range(len(sentences)), key=lambda index: float(scores[index]), reverse=True)
    selected_indices: list[int] = []
    token_total = 0
    for index in ranked_indices:
        sentence = sentences[index]
        sentence_tokens = count_tokens(tokenizer, sentence)
        if selected_indices and token_total + sentence_tokens > token_budget:
            continue
        selected_indices.append(index)
        token_total += sentence_tokens
        if len(selected_indices) >= max_sentences:
            break

    if not selected_indices:
        selected_indices = [0]

    selected_indices.sort()
    selected_sentences = [sentences[index] for index in selected_indices]
    compressed_text = " ".join(selected_sentences).strip()
    original_tokens = max(count_tokens(tokenizer, text), 1)
    compression_ratio = count_tokens(tokenizer, compressed_text) / original_tokens
    selected_scores = [float(scores[index]) for index in selected_indices if len(scores) > index]
    support_score = float(np.mean(selected_scores)) if selected_scores else 0.0
    return compressed_text, selected_sentences, support_score if selected_scores else 1.0, compression_ratio


def compress_evidence(
    query: str,
    evidences: list[RetrievedEvidence],
    *,
    encoder: SentenceTransformer,
    tokenizer: PreTrainedTokenizerBase,
    total_token_budget: int,
    max_sentences_per_chunk: int = 3,
) -> list[RetrievedEvidence]:
    if not evidences:
        return []

    remaining_budget = max(total_token_budget, 64)
    per_chunk_budget = max(48, remaining_budget // max(1, len(evidences)))
    compressed: list[RetrievedEvidence] = []
    for evidence in evidences:
        if remaining_budget <= 0:
            break
        budget = min(per_chunk_budget, remaining_budget)
        compressed_text, selected_sentences, support_score, compression_ratio = _compress_evidence_text(
            query,
            evidence.chunk.text,
            encoder=encoder,
            tokenizer=tokenizer,
            token_budget=budget,
            max_sentences=max_sentences_per_chunk,
        )
        compressed_tokens = count_tokens(tokenizer, compressed_text)
        if not compressed_text or compressed_tokens == 0:
            continue
        evidence.compressed_text = compressed_text
        evidence.selected_sentences = selected_sentences
        evidence.support_score = support_score
        evidence.compression_ratio = compression_ratio
        compressed.append(evidence)
        remaining_budget -= compressed_tokens
    return compressed


def select_diverse_evidence(evidences: list[RetrievedEvidence], max_items: int) -> list[RetrievedEvidence]:
    selected: list[RetrievedEvidence] = []
    seen_doc_ids: set[str] = set()
    for evidence in evidences:
        if evidence.chunk.doc_id in seen_doc_ids:
            continue
        selected.append(evidence)
        seen_doc_ids.add(evidence.chunk.doc_id)
        if len(selected) >= max_items:
            return selected
    for evidence in evidences:
        if evidence in selected:
            continue
        selected.append(evidence)
        if len(selected) >= max_items:
            break
    return selected


def format_context(evidences: list[RetrievedEvidence], *, compressed: bool) -> str:
    blocks: list[str] = []
    for index, evidence in enumerate(evidences, start=1):
        text = evidence.compressed_text if compressed and evidence.compressed_text else evidence.chunk.text
        matched = ", ".join(evidence.matched_queries[:2])
        block = f"[{index}] Title: {evidence.chunk.title}\nMatched queries: {matched}\n{text}"
        blocks.append(block)
    return "\n\n".join(blocks)


def estimate_answer_support(answer: str, evidences: list[RetrievedEvidence]) -> float:
    answer_terms = {token for token in lexical_tokenize(answer) if token not in STOPWORDS}
    if not answer_terms:
        return 1.0
    evidence_terms = set()
    for evidence in evidences:
        evidence_terms.update(lexical_tokenize(evidence.compressed_text or evidence.chunk.text))
    return len(answer_terms & evidence_terms) / len(answer_terms)


def support_label(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.4:
        return "medium"
    return "low"


def _trace(stage: str, detail: str, started_at: float) -> dict[str, Any]:
    return {
        "stage": stage,
        "detail": detail,
        "seconds": round(time.perf_counter() - started_at, 4),
    }


def build_citations(evidences: list[RetrievedEvidence]) -> list[dict[str, Any]]:
    citations: list[dict[str, Any]] = []
    for index, evidence in enumerate(evidences, start=1):
        snippet = (evidence.selected_sentences[0] if evidence.selected_sentences else evidence.chunk.text).strip()
        citations.append(
            {
                "index": index,
                "title": evidence.chunk.title,
                "doc_id": evidence.chunk.doc_id,
                "snippet": snippet[:280],
                "matched_queries": evidence.matched_queries,
                "score": round(evidence.score, 4),
                "support_score": round(evidence.support_score, 4),
                "compression_ratio": round(evidence.compression_ratio, 4),
            }
        )
    return citations


def run_live_rag(
    index: DemoIndex,
    *,
    question: str,
    generator: Any | None = None,
    mode: str = "advanced",
    top_k: int = 4,
    per_query_k: int = 8,
    compression_token_budget: int = 384,
    max_subqueries: int = 3,
) -> dict[str, Any]:
    generator = generator or ExtractiveFallbackGenerator()
    trace: list[dict[str, Any]] = []

    if mode not in {"traditional", "advanced", "multi_agent"}:
        raise ValueError(f"Unsupported mode: {mode}")

    started_at = time.perf_counter()
    query_variants = [question.strip()]
    subqueries: list[str] = []

    if mode in {"advanced", "multi_agent"}:
        rewrite_started = time.perf_counter()
        query_variants = generate_query_variants(question)
        trace.append(_trace("query_rewrite", f"Generated {len(query_variants)} retrieval queries.", rewrite_started))

    if mode == "multi_agent":
        planning_started = time.perf_counter()
        subqueries = plan_subqueries(question, max_subqueries=max_subqueries)
        trace.append(_trace("planner_agent", f"Proposed {len(subqueries)} subqueries for decomposition.", planning_started))

    retrieval_queries = _unique_in_order([question, *query_variants, *subqueries])
    retrieval_started = time.perf_counter()
    candidate_depth = max(top_k, per_query_k, len(retrieval_queries) * 2)
    evidences = retrieve_for_queries(
        index.retriever,
        retrieval_queries,
        top_k=max(top_k * 2, candidate_depth),
        per_query_k=candidate_depth,
    )
    trace.append(
        _trace(
            "retrieval",
            f"Retrieved {len(evidences)} evidence chunks from {len({e.chunk.doc_id for e in evidences})} documents.",
            retrieval_started,
        )
    )

    selected = evidences[:top_k]
    uses_compression = False
    if mode in {"advanced", "multi_agent"}:
        compression_started = time.perf_counter()
        compressed = compress_evidence(
            question,
            evidences,
            encoder=index.encoder,
            tokenizer=index.tokenizer,
            total_token_budget=compression_token_budget,
            max_sentences_per_chunk=3,
        )
        trace.append(
            _trace(
                "compression",
                f"Compressed {len(compressed)} chunks to a {compression_token_budget}-token budget.",
                compression_started,
            )
        )
        selected = compressed[:top_k]
        uses_compression = True

    if mode == "multi_agent":
        filter_started = time.perf_counter()
        selected = select_diverse_evidence(selected, top_k)
        trace.append(
            _trace(
                "filter_agent",
                f"Selected {len(selected)} diverse evidence chunks across {len({e.chunk.doc_id for e in selected})} documents.",
                filter_started,
            )
        )

    generation_started = time.perf_counter()
    context = format_context(selected, compressed=uses_compression)
    answer = generator.answer(question, context=context)
    trace.append(_trace("writer", "Generated the final grounded answer.", generation_started))

    evaluation_started = time.perf_counter()
    answer_support = estimate_answer_support(answer, selected)
    trace.append(
        _trace(
            "evaluator",
            f"Estimated grounding support as {support_label(answer_support)} ({answer_support:.2f}).",
            evaluation_started,
        )
    )

    return {
        "mode": mode,
        "question": question,
        "answer": answer,
        "query_variants": query_variants,
        "subqueries": subqueries,
        "retrieval_queries": retrieval_queries,
        "trace": trace,
        "answer_support": answer_support,
        "support_label": support_label(answer_support),
        "citations": build_citations(selected),
        "context": context,
        "selected_chunks": [
            {
                "title": evidence.chunk.title,
                "doc_id": evidence.chunk.doc_id,
                "text": evidence.chunk.text,
                "compressed_text": evidence.compressed_text,
                "matched_queries": evidence.matched_queries,
                "score": evidence.score,
                "raw_score": evidence.raw_score,
                "support_score": evidence.support_score,
                "compression_ratio": evidence.compression_ratio,
            }
            for evidence in selected
        ],
        "num_documents": len(index.documents),
        "num_chunks": len(index.chunks),
        "elapsed_seconds": round(time.perf_counter() - started_at, 4),
    }
