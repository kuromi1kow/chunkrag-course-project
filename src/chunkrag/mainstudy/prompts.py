"""Exact prompt versions and rendering (Specification Sections 15--16)."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from .canonical import canonical_json_hash

EXTRACTIVE_SYSTEM = """You are an extractive question answering assistant. Use only the provided context.
Copy the shortest answer span supported by the context. Do not explain your reasoning.
If the answer is not fully supported, reply with exactly unanswerable."""

EXTRACTIVE_USER = """Answer the following question using only the context.

Question: {question}

Context passages:
{context}

Return only the answer text with no explanation."""

TECHNICAL_SYSTEM = """You are a grounded technical question answering assistant. Use only the provided
context. Give a concise but complete answer containing the information needed to resolve
the question. Do not add citations or unsupported details. If the answer is not
supported, reply with exactly unanswerable."""

TECHNICAL_USER = """Answer the following technical question using only the context.

Question: {question}

Context passages:
{context}

Return only the final answer."""


def prompt_version(dataset: str) -> str:
    return "technical-v1" if dataset == "techqa" else "extractive-v1"


def prompt_template_hash(dataset: str) -> str:
    if dataset == "techqa":
        payload = {"version": "technical-v1", "system": TECHNICAL_SYSTEM, "user_template": TECHNICAL_USER}
    else:
        payload = {"version": "extractive-v1", "system": EXTRACTIVE_SYSTEM, "user_template": EXTRACTIVE_USER}
    return canonical_json_hash(payload)


def messages(dataset: str, question: str, context: str) -> list[dict[str, str]]:
    if dataset == "techqa":
        return [{"role": "system", "content": TECHNICAL_SYSTEM}, {"role": "user", "content": TECHNICAL_USER.format(question=question, context=context)}]
    return [{"role": "system", "content": EXTRACTIVE_SYSTEM}, {"role": "user", "content": EXTRACTIVE_USER.format(question=question, context=context)}]


def render_passages(chunks: Sequence[Mapping[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    parts: list[str] = []
    spans: list[dict[str, Any]] = []
    cursor = 0
    for rank_value, chunk in enumerate(chunks, start=1):
        title = str(chunk.get("title") or chunk["document_id"])
        rendered = f"[{rank_value}] Title: {title}\nPassage: {chunk['text']}"
        if parts:
            cursor += 2
        start = cursor
        parts.append(rendered)
        cursor += len(rendered)
        marker = "Passage: "
        text_start = start + rendered.index(marker) + len(marker)
        spans.append({"chunk_id": chunk.get("chunk_id"), "document_id": chunk["document_id"], "rank": rank_value, "rendered_start": start, "rendered_end": cursor, "text_rendered_start": text_start, "text_rendered_end": cursor, "source_char_start": chunk.get("char_start"), "source_char_end": chunk.get("char_end")})
    return "\n\n".join(parts), spans


LABEL_RE = re.compile(r"^\s*(Answer|Final answer)\s*:\s*", re.IGNORECASE)


def normalize_generated_answer(text: str) -> str:
    return LABEL_RE.sub("", text.strip(), count=1)
