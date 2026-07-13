"""Operational and exposure-matched context packing (Specification Section 16)."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .prompts import messages, render_passages


TokenCounter = Callable[[list[dict[str, str]]], list[int]]


@dataclass(frozen=True, slots=True)
class PackedContext:
    rendered_context: str
    consumed_context: str
    spans: tuple[dict[str, Any], ...]
    prompt_token_ids: tuple[int, ...]
    full_prompt_tokens: int
    context_tokens: int
    target: int
    truncation_location: int | None
    per_chunk_consumed_tokens: tuple[int, ...]


def _context_token_count(tokenizer: Any, context: str) -> int:
    return len(tokenizer(context, add_special_tokens=False, truncation=False)["input_ids"])


def _prompt_ids(tokenizer: Any, dataset: str, question: str, context: str) -> list[int]:
    return list(tokenizer.apply_chat_template(messages(dataset, question, context), tokenize=True, add_generation_prompt=True))


def longest_prefix(
    tokenizer: Any, dataset: str, question: str, rendered: str, *, input_budget: int,
    context_target: int | None = None, safety_margin: int = 0,
) -> PackedContext:
    if len(_prompt_ids(tokenizer, dataset, question, "")) + safety_margin > input_budget:
        raise ValueError("Frozen question and prompt cannot fit the complete-chat input budget")
    low, high = 0, len(rendered)
    best = 0
    best_ids: list[int] = []
    while low <= high:
        mid = (low + high) // 2
        candidate = rendered[:mid]
        context_tokens = _context_token_count(tokenizer, candidate)
        ids = _prompt_ids(tokenizer, dataset, question, candidate)
        fits = len(ids) + safety_margin <= input_budget and (context_target is None or context_tokens <= context_target)
        if fits:
            best, best_ids = mid, ids
            low = mid + 1
        else:
            high = mid - 1
    context = rendered[:best]
    count = _context_token_count(tokenizer, context)
    target = count if context_target is None else context_target
    full_prompt_tokens = len(_prompt_ids(tokenizer, dataset, question, rendered))
    return PackedContext(rendered, context, (), tuple(best_ids), full_prompt_tokens, count, target, None if best == len(rendered) else best, ())


def operational_pack(
    tokenizer: Any, dataset: str, question: str, chunks: Sequence[Mapping[str, Any]], input_budget: int,
) -> PackedContext:
    rendered, spans = render_passages(chunks[:4])
    packed = longest_prefix(tokenizer, dataset, question, rendered, input_budget=input_budget)
    per_chunk = tuple(_context_token_count(tokenizer, packed.consumed_context[max(0, span["rendered_start"]):min(len(packed.consumed_context), span["rendered_end"])]) if span["rendered_start"] < len(packed.consumed_context) else 0 for span in spans)
    return PackedContext(rendered, packed.consumed_context, tuple(spans), packed.prompt_token_ids, packed.full_prompt_tokens, packed.context_tokens, packed.target, packed.truncation_location, per_chunk)


def matched_target(
    tokenizer: Any, dataset: str, question: str, rendered_contexts: Sequence[str], input_budget: int,
) -> int:
    nominal = 768 if input_budget == 1024 else 3072
    empty_ids = _prompt_ids(tokenizer, dataset, question, "")
    capacity = max(0, input_budget - len(empty_ids) - 16)
    available = [_context_token_count(tokenizer, context) for context in rendered_contexts]
    if not available:
        raise ValueError("Matched target requires at least one system context")
    return min(nominal, capacity, *available)


def matched_pack(
    tokenizer: Any, dataset: str, question: str, chunks: Sequence[Mapping[str, Any]],
    input_budget: int, target: int,
) -> PackedContext:
    rendered, spans = render_passages(chunks[:16])
    packed = longest_prefix(
        tokenizer, dataset, question, rendered, input_budget=input_budget,
        context_target=target, safety_margin=16,
    )
    if abs(packed.context_tokens - target) > 2:
        raise ValueError(f"Matched context differs from target by more than two tokens: {packed.context_tokens} vs {target}")
    per_chunk = tuple(_context_token_count(tokenizer, packed.consumed_context[max(0, span["rendered_start"]):min(len(packed.consumed_context), span["rendered_end"])]) if span["rendered_start"] < len(packed.consumed_context) else 0 for span in spans)
    return PackedContext(rendered, packed.consumed_context, tuple(spans), packed.prompt_token_ids, packed.full_prompt_tokens, packed.context_tokens, target, packed.truncation_location, per_chunk)
