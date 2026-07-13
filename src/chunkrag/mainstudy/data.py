"""E0 frozen data/corpus materialization (Specification Sections 8--10 and E0)."""

from __future__ import annotations

import hashlib
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .canonical import canonical_json_hash, identifier_hash, nfc, sha256_bytes
from .constants import EXPECTED_QUESTION_COUNTS, PROTOCOL_ID
from .protocol import ProtocolError


def normalize_corpus_text(value: str) -> str:
    return nfc(value.replace("\r\n", "\n"))


def normalize_question_text(value: str) -> str:
    return normalize_corpus_text(value).strip()


def selection_hash(dataset: str, example_id: str) -> str:
    return sha256_bytes(f"chunkrag-main-v1\0{dataset}\0{example_id}".encode("utf-8"))


def squad_document_id(title: str, context: str) -> str:
    return "squad::" + sha256_bytes(f"{title}\0{context}".encode("utf-8"))


def hotpot_document_id(title: str, text: str) -> str:
    return "hotpot::" + sha256_bytes(f"{title}\0{text}".encode("utf-8"))


def techqa_document_id(filename: str) -> str:
    return "techqa::" + nfc(filename)


def _base_question(
    *, dataset: str, revision: str, row: Mapping[str, Any], question: str, references: list[str],
    gold_document_ids: list[str], gold_spans: list[dict[str, Any]],
    supporting_facts: list[dict[str, Any]], eligibility: dict[str, Any],
) -> dict[str, Any]:
    example_id = str(row["id"])
    return {
        "schema_version": PROTOCOL_ID,
        "question_id": example_id,
        "dataset": dataset,
        "revision": revision,
        "selection_hash": selection_hash(dataset, example_id),
        "selection_rank": -1,
        "question": normalize_question_text(question),
        "references": [normalize_question_text(item) for item in references],
        "gold_document_ids": list(dict.fromkeys(gold_document_ids)),
        "cluster_id": "",
        "eligibility": eligibility,
        "gold_spans": gold_spans,
        "supporting_facts": supporting_facts,
        "source_provenance": {"source_row_id": example_id},
    }


def materialize_squad_rows(rows: Iterable[Mapping[str, Any]], revision: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    corpus: dict[str, dict[str, Any]] = {}
    candidates: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        title = nfc(str(row["title"]))
        original_context = str(row["context"])
        context = normalize_corpus_text(original_context)
        document_id = squad_document_id(title, context)
        corpus.setdefault(document_id, {
            "schema_version": PROTOCOL_ID, "document_id": document_id, "dataset": "squad_v2",
            "title": title, "text": context, "split": "validation", "revision": revision,
            "text_sha256": sha256_bytes(context.encode("utf-8")),
            "source_provenance": [{"row_index": row_index, "example_id": str(row["id"])}],
        })
        answers = row.get("answers", {})
        texts = list(answers.get("text", []))
        starts = list(answers.get("answer_start", []))
        if not texts or not any(str(text).strip() for text in texts):
            continue
        spans: list[dict[str, Any]] = []
        for text, start in zip(texts, starts):
            raw_text = str(text)
            raw_start = int(start)
            normalized_start = len(normalize_corpus_text(original_context[:raw_start]))
            normalized_answer = normalize_corpus_text(raw_text)
            normalized_end = normalized_start + len(normalized_answer)
            if context[normalized_start:normalized_end] != normalized_answer:
                raise ProtocolError(f"SQuAD normalized answer span mismatch: {row['id']}")
            spans.append({"document_id": document_id, "char_start": normalized_start, "char_end": normalized_end, "text": normalized_answer})
        question = _base_question(
            dataset="squad_v2", revision=revision, row=row, question=str(row["question"]),
            references=[str(item) for item in texts], gold_document_ids=[document_id],
            gold_spans=spans, supporting_facts=[], eligibility={"answerable": True, "allocation_key": title},
        )
        candidates.append(question)
    selected = _select(candidates, EXPECTED_QUESTION_COUNTS["squad_v2"], 20)
    _assign_simple_clusters(selected, lambda row: str(row["eligibility"]["allocation_key"]), "squad_v2")
    return list(corpus.values()), selected


def _hotpot_documents(row: Mapping[str, Any], revision: str, row_index: int) -> tuple[list[dict[str, Any]], dict[str, str], list[dict[str, Any]]]:
    titles = list(row["context"]["title"])
    sentence_lists = list(row["context"]["sentences"])
    documents: list[dict[str, Any]] = []
    by_title: dict[str, str] = {}
    sentence_provenance: list[dict[str, Any]] = []
    for document_index, (raw_title, raw_sentences) in enumerate(zip(titles, sentence_lists)):
        title = nfc(str(raw_title))
        sentences = [normalize_corpus_text(str(item)).strip() for item in raw_sentences]
        text = " ".join(sentences)
        doc_id = hotpot_document_id(title, text)
        by_title[title] = doc_id
        cursor = 0
        spans: list[dict[str, int]] = []
        for sentence_index, sentence in enumerate(sentences):
            start = cursor
            end = start + len(sentence)
            spans.append({"sentence_index": sentence_index, "char_start": start, "char_end": end})
            cursor = end + 1
        sentence_provenance.extend({"document_id": doc_id, "document_index": document_index, **span} for span in spans)
        documents.append({
            "schema_version": PROTOCOL_ID, "document_id": doc_id, "dataset": "hotpot_qa",
            "title": title, "text": text, "split": "validation", "revision": revision,
            "text_sha256": sha256_bytes(text.encode("utf-8")),
            "source_provenance": [{"row_index": row_index, "document_index": document_index, "sentence_spans": spans}],
        })
    return documents, by_title, sentence_provenance


def materialize_hotpot_rows(rows: Iterable[Mapping[str, Any]], revision: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    corpus: dict[str, dict[str, Any]] = {}
    candidates: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        documents, by_title, sentence_provenance = _hotpot_documents(row, revision, row_index)
        for document in documents:
            existing = corpus.get(document["document_id"])
            if existing is None:
                corpus[document["document_id"]] = document
            else:
                existing["source_provenance"].extend(document["source_provenance"])
        fact_titles = [nfc(str(item)) for item in row["supporting_facts"]["title"]]
        fact_indices = [int(item) for item in row["supporting_facts"]["sent_id"]]
        if not str(row.get("answer", "")).strip() or not fact_titles or any(title not in by_title for title in fact_titles):
            continue
        facts: list[dict[str, Any]] = []
        for title, sentence_index in zip(fact_titles, fact_indices):
            doc_id = by_title[title]
            provenance = next(item for item in sentence_provenance if item["document_id"] == doc_id and item["sentence_index"] == sentence_index)
            facts.append({"title": title, "document_id": doc_id, **{key: provenance[key] for key in ("document_index", "sentence_index", "char_start", "char_end")}})
        allocation_key = min(fact_titles)
        question = _base_question(
            dataset="hotpot_qa", revision=revision, row=row, question=str(row["question"]),
            references=[str(row["answer"])], gold_document_ids=[by_title[title] for title in fact_titles],
            gold_spans=[], supporting_facts=facts,
            eligibility={"answerable": True, "allocation_key": allocation_key},
        )
        candidates.append(question)
    selected = _select(candidates, EXPECTED_QUESTION_COUNTS["hotpot_qa"], 2)
    _assign_component_clusters(selected, "hotpot_qa")
    return list(corpus.values()), selected


def materialize_techqa_rows(rows: Iterable[Mapping[str, Any]], revision: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    corpus: dict[str, dict[str, Any]] = {}
    candidates: list[dict[str, Any]] = []
    eligible_rows: list[tuple[int, Mapping[str, Any]]] = []
    for row_index, row in enumerate(rows):
        if bool(row["is_impossible"]) or not str(row["answer"]).strip() or not row["contexts"]:
            continue
        eligible_rows.append((row_index, row))
        for context_index, context in enumerate(row["contexts"]):
            filename = nfc(str(context["filename"]))
            text = normalize_corpus_text(str(context["text"]))
            doc_id = techqa_document_id(filename)
            record = {
                "schema_version": PROTOCOL_ID, "document_id": doc_id, "dataset": "techqa",
                "title": filename, "text": text, "split": "train", "revision": revision,
                "text_sha256": sha256_bytes(text.encode("utf-8")),
                "source_provenance": [{"row_index": row_index, "context_index": context_index, "filename": filename}],
            }
            existing = corpus.get(doc_id)
            if existing is not None and existing["text_sha256"] != record["text_sha256"]:
                raise ProtocolError(f"Conflicting TechQA filename: {filename}")
            if existing is None:
                corpus[doc_id] = record
            else:
                existing["source_provenance"].extend(record["source_provenance"])
    for _, row in eligible_rows:
        filenames = [nfc(str(item["filename"])) for item in row["contexts"]]
        gold_ids = [techqa_document_id(item) for item in filenames]
        question = _base_question(
            dataset="techqa", revision=revision, row=row, question=str(row["question"]),
            references=[str(row["answer"])], gold_document_ids=gold_ids, gold_spans=[],
            supporting_facts=[], eligibility={"answerable": True, "allocation_key": min(filenames)},
        )
        candidates.append(question)
    selected = _select(candidates, EXPECTED_QUESTION_COUNTS["techqa"], 2)
    _assign_component_clusters(selected, "techqa")
    return list(corpus.values()), selected


def _select(candidates: list[dict[str, Any]], target: int, cap: int) -> list[dict[str, Any]]:
    ordered = sorted(candidates, key=lambda row: (row["selection_hash"], row["question_id"]))
    counts: dict[str, int] = defaultdict(int)
    selected: list[dict[str, Any]] = []
    for candidate in ordered:
        key = str(candidate["eligibility"]["allocation_key"])
        if counts[key] >= cap:
            continue
        candidate["selection_rank"] = len(selected)
        selected.append(candidate)
        counts[key] += 1
        if len(selected) == target:
            break
    if len(selected) != target:
        raise ProtocolError(f"Frozen sample target {target} cannot be satisfied under cap {cap}")
    return selected


def _assign_simple_clusters(rows: list[dict[str, Any]], key, dataset: str) -> None:
    for row in rows:
        value = key(row)
        row["cluster_id"] = f"{dataset}::cluster::{sha256_bytes(value.encode('utf-8'))}"


def _assign_component_clusters(rows: list[dict[str, Any]], dataset: str) -> None:
    parent = {row["question_id"]: row["question_id"] for row in rows}

    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: str, right: str) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[max(a, b)] = min(a, b)

    by_document: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        for doc_id in row["gold_document_ids"]:
            by_document[doc_id].append(row["question_id"])
    for ids in by_document.values():
        for item in ids[1:]:
            union(ids[0], item)
    members: dict[str, list[str]] = defaultdict(list)
    for question_id in parent:
        members[find(question_id)].append(question_id)
    cluster_by_question: dict[str, str] = {}
    for values in members.values():
        cluster_id = f"{dataset}::cluster::{identifier_hash(*sorted(values))}"
        for question_id in values:
            cluster_by_question[question_id] = cluster_id
    for row in rows:
        row["cluster_id"] = cluster_by_question[row["question_id"]]


def cluster_records(dataset: str, questions: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    members: dict[str, list[str]] = defaultdict(list)
    for row in questions:
        members[str(row["cluster_id"])].append(str(row["question_id"]))
    return [{
        "schema_version": PROTOCOL_ID, "cluster_id": cluster_id, "dataset": dataset,
        "question_ids": sorted(ids), "size": len(ids),
    } for cluster_id, ids in sorted(members.items())]


def validate_cluster_constraints(clusters: list[Mapping[str, Any]], total_questions: int) -> None:
    if len(clusters) < 30:
        raise ProtocolError(f"Expected at least 30 clusters, found {len(clusters)}")
    if max(int(row["size"]) for row in clusters) / total_questions > 0.10:
        raise ProtocolError("A frozen cluster exceeds 10% of selected questions")


def load_pinned_dataset(spec: Mapping[str, Any]):
    """Lazy Phase-4 loader. Calling this function materializes external data."""
    from datasets import load_dataset

    kwargs: dict[str, Any] = {"split": spec["split"], "revision": spec["revision"]}
    if spec.get("config") is not None:
        return load_dataset(spec["repository"], spec["config"], **kwargs)
    return load_dataset(spec["repository"], **kwargs)
