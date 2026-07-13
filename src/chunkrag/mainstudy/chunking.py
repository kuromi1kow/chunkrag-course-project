"""Span-preserving primary chunkers (Specification Section 11 and E1)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence

from .canonical import identifier_hash, sha256_bytes
from .constants import PROTOCOL_ID
from .schemas import validate_record


class TokenizerLike(Protocol):
    def __call__(self, text: str, **kwargs: Any) -> Mapping[str, Any]: ...


@dataclass(frozen=True, slots=True)
class TokenizedSource:
    text: str
    input_ids: tuple[int, ...]
    offsets: tuple[tuple[int, int], ...]
    boundaries: tuple[int, ...]

    @classmethod
    def build(cls, text: str, tokenizer: TokenizerLike) -> "TokenizedSource":
        encoded = tokenizer(
            text, add_special_tokens=False, truncation=False, return_offsets_mapping=True,
        )
        ids = tuple(int(item) for item in encoded["input_ids"])
        offsets = tuple((int(start), int(end)) for start, end in encoded["offset_mapping"])
        if len(ids) != len(offsets):
            raise ValueError("Tokenizer IDs and offsets have different lengths")
        if any(start < 0 or end < start or end > len(text) for start, end in offsets):
            raise ValueError("Tokenizer returned invalid source offsets")
        boundaries = [0]
        boundaries.extend(offsets[index][0] for index in range(1, len(offsets)))
        boundaries.append(len(text))
        if any(left > right for left, right in zip(boundaries, boundaries[1:])):
            raise ValueError("Tokenizer boundaries are not monotonic")
        return cls(text=text, input_ids=ids, offsets=offsets, boundaries=tuple(boundaries))

    @property
    def tokens(self) -> int:
        return len(self.input_ids)

    def char_span(self, start: int, end: int) -> tuple[int, int]:
        if not (0 <= start <= end <= self.tokens):
            raise ValueError("Token span outside source")
        return self.boundaries[start], self.boundaries[end]


def _merge_short_final(cuts: list[int], total: int, maximum: int = 254, minimum: int = 64) -> list[int]:
    if total <= 0:
        raise ValueError("Canonical chunking requires a nonempty tokenized source")
    if len(cuts) < 2 or cuts[0] != 0 or cuts[-1] != total:
        raise ValueError("Chunk cuts must include exact source boundaries")
    if len(cuts) < 3:
        # A source shorter than one target window is already one valid final-short
        # chunk.  There is no preceding chunk to merge it into.
        return cuts
    final_length = total - cuts[-2]
    if final_length < minimum and total - cuts[-3] <= maximum:
        del cuts[-2]
    return cuts


def fixed_cuts(total: int, target: int = 192) -> list[int]:
    cuts = [0] + list(range(target, total, target)) + [total]
    return _merge_short_final(cuts, total)


def _separator_class(source: TokenizedSource, start: int, cut: int) -> int | None:
    char_start, char_cut = source.char_span(start, cut)
    prefix = source.text[char_start:char_cut]
    if re.search(r"\n{2,}$", prefix):
        return 0
    if prefix.endswith("\n"):
        return 1
    if re.search(r'''[.!?]["')\]]*\s+$''', prefix):
        return 2
    if prefix and prefix[-1].isspace():
        return 3
    return None


def recursive_cuts(source: TokenizedSource) -> list[int]:
    cuts = [0]
    while source.tokens - cuts[-1] > 192:
        start = cuts[-1]
        candidates: dict[int, list[int]] = {0: [], 1: [], 2: [], 3: []}
        for cut in range(start + 128, min(start + 192, source.tokens) + 1):
            kind = _separator_class(source, start, cut)
            if kind is not None:
                candidates[kind].append(cut)
        chosen = next((max(candidates[kind]) for kind in range(4) if candidates[kind]), start + 192)
        cuts.append(chosen)
    cuts.append(source.tokens)
    return _merge_short_final(cuts, source.tokens)


def spacy_sentence_spans(text: str) -> list[tuple[int, int]]:
    import spacy

    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer", config={"punct_chars": [".", "!", "?"]})
    return [(sentence.start_char, sentence.end_char) for sentence in nlp(text).sents]


def _char_to_token_end(source: TokenizedSource, char_end: int) -> int:
    for index, boundary in enumerate(source.boundaries):
        if boundary >= char_end:
            return index
    return source.tokens


def sentence_units(source: TokenizedSource, spans: Sequence[tuple[int, int]] | None = None) -> list[tuple[int, int]]:
    spans = list(spans if spans is not None else spacy_sentence_spans(source.text))
    units: list[tuple[int, int]] = []
    previous = 0
    for _, char_end in spans:
        end = _char_to_token_end(source, char_end)
        if end <= previous:
            continue
        while end - previous > 254:
            units.append((previous, previous + 192))
            previous += 192
        units.append((previous, end))
        previous = end
    if previous < source.tokens:
        units.append((previous, source.tokens))
    return units


def sentence_cuts(source: TokenizedSource, spans: Sequence[tuple[int, int]] | None = None) -> list[int]:
    units = sentence_units(source, spans)
    if not units:
        return [0, source.tokens]
    cuts = [0]
    index = 0
    while index < len(units):
        start = cuts[-1]
        end = units[index][1]
        index += 1
        while index < len(units):
            candidate = units[index][1]
            current = end - start
            if candidate - start <= 192 or (current < 96 and candidate - start <= 254):
                end = candidate
                index += 1
            else:
                break
        cuts.append(end)
    return _merge_short_final(cuts, source.tokens)


def semantic_cuts(
    source: TokenizedSource,
    encode: Callable[[list[str]], Sequence[Sequence[float]]],
    spans: Sequence[tuple[int, int]] | None = None,
) -> list[int]:
    import numpy as np

    units = sentence_units(source, spans)
    if not units:
        return [0, source.tokens]
    texts = [source.text[slice(*source.char_span(start, end))] for start, end in units]
    vectors = np.asarray(encode(texts), dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Semantic encoder returned a zero vector")
    vectors = vectors / norms
    similarities = [float(vectors[i] @ vectors[i + 1]) for i in range(len(units) - 1)]
    cuts = [0]
    unit_index = 0
    while source.tokens - cuts[-1] > 254:
        start = cuts[-1]
        candidates: list[tuple[float, int, int]] = []
        for index in range(unit_index, len(units) - 1):
            boundary = units[index][1]
            length = boundary - start
            if 128 <= length <= 254:
                candidates.append((similarities[index], abs(length - 192), boundary))
            if length > 254:
                break
        if candidates:
            _, _, chosen = min(candidates, key=lambda item: (item[0], item[1], item[2]))
        else:
            chosen = min(start + 192, source.tokens)
        cuts.append(chosen)
        while unit_index < len(units) and units[unit_index][1] <= chosen:
            unit_index += 1
    cuts.append(source.tokens)
    return _merge_short_final(cuts, source.tokens)


def chunk_records(
    document: Mapping[str, Any], source: TokenizedSource, policy: str, cuts: list[int],
    tokenizer_repository: str, tokenizer_revision: str, *, condition_id: str | None = None,
    control_seed: int | None = None, boundary_generation_hash: str = "0" * 64,
) -> list[dict[str, Any]]:
    condition_id = condition_id or policy
    records: list[dict[str, Any]] = []
    for ordinal, (token_start, token_end) in enumerate(zip(cuts, cuts[1:])):
        char_start, char_end = source.char_span(token_start, token_end)
        text = source.text[char_start:char_end]
        text_hash = sha256_bytes(text.encode("utf-8"))
        chunk_id = identifier_hash(
            document["dataset"], condition_id, document["document_id"], token_start, token_end, text_hash,
        )
        token_count = token_end - token_start
        if token_count > 254 or (token_count < 64 and not (ordinal == len(cuts) - 2)):
            raise ValueError(f"Chunk length violates frozen bounds: {token_count}")
        record = {
            "schema_version": PROTOCOL_ID, "chunk_id": chunk_id, "condition_id": condition_id,
            "policy": policy, "policy_version": "1", "dataset": document["dataset"],
            "document_id": document["document_id"], "char_start": char_start, "char_end": char_end,
            "token_start": token_start, "token_end": token_end, "token_count": token_count,
            "text": text, "text_sha256": text_hash,
            "preceding_separator": source.text[max(0, char_start - 2):char_start],
            "following_separator": source.text[char_end:min(len(source.text), char_end + 2)],
            "ordinal": ordinal, "final_short": ordinal == len(cuts) - 2 and token_count < 64,
            "tokenizer_repository": tokenizer_repository, "tokenizer_revision": tokenizer_revision,
            "control_seed": control_seed, "parent_chunk_count": len(cuts) - 1,
            "boundary_generation_hash": boundary_generation_hash,
        }
        validate_record("chunk", record)
        records.append(record)
    validate_round_trip(document, records)
    return records


def validate_round_trip(document: Mapping[str, Any], chunks: Sequence[Mapping[str, Any]]) -> None:
    ordered = sorted(chunks, key=lambda row: int(row["ordinal"]))
    if "".join(str(row["text"]) for row in ordered) != document["text"]:
        raise ValueError(f"Chunk round trip failed: {document['document_id']}")
    if ordered and (ordered[0]["char_start"] != 0 or ordered[-1]["char_end"] != len(document["text"])):
        raise ValueError("Chunk coverage does not reach both source boundaries")
    for left, right in zip(ordered, ordered[1:]):
        if left["char_end"] != right["char_start"] or left["token_end"] != right["token_start"]:
            raise ValueError("Chunk coverage has a gap or overlap")
