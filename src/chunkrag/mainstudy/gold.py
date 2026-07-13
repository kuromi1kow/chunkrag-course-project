"""Gold-evidence source manifests and packing order (Specification Section 17 and E3)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .canonical import canonical_json_hash, identifier_hash
from .constants import PROTOCOL_ID


def gold_manifest(question: Mapping[str, Any], corpus_by_id: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    dataset = str(question["dataset"])
    units: list[dict[str, Any]] = []
    if dataset == "squad_v2":
        for span in sorted(question["gold_spans"], key=lambda item: (item["char_start"], -(item["char_end"] - item["char_start"]), item["text"])):
            units.append({"kind": "squad_paragraph", "document_id": span["document_id"], "answer_span": span})
    elif dataset == "hotpot_qa":
        for fact in sorted(question["supporting_facts"], key=lambda item: (item["document_index"], item["sentence_index"])):
            units.append({"kind": "supporting_sentence", **dict(fact)})
    elif dataset == "techqa":
        for document_id in sorted(question["gold_document_ids"]):
            units.append({"kind": "techqa_document", "document_id": document_id})
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    source_hash = canonical_json_hash([corpus_by_id[item["document_id"]] for item in units])
    return {
        "schema_version": PROTOCOL_ID,
        "gold_id": identifier_hash(question["question_id"], dataset, source_hash),
        "question_id": question["question_id"],
        "dataset": dataset,
        "ordered_units": units,
        "source_hash": source_hash,
    }
