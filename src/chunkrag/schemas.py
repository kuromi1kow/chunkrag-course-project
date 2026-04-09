from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Document:
    doc_id: str
    title: str
    text: str
    dataset: str


@dataclass(slots=True)
class QAExample:
    example_id: str
    dataset: str
    question: str
    answers: list[str]
    relevant_doc_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    dataset: str
    text: str
    token_count: int
    metadata: dict[str, Any] = field(default_factory=dict)
