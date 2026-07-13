"""Deterministic randomized-boundary controls (Specification Section 12 and E1)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .canonical import canonical_json_hash


@dataclass(frozen=True, slots=True)
class JitterResult:
    cuts: tuple[int, ...]
    feasible: int
    changed: int
    generation_hash: str


def jitter_cuts(
    base_cuts: list[int], *, seed: int, policy: str, document_id: str,
    final_short: bool,
) -> JitterResult:
    if len(base_cuts) < 2 or base_cuts[0] != 0:
        raise ValueError("Base cuts must start at zero and include document end")
    total = base_cuts[-1]
    segments = len(base_cuts) - 1
    updated = [0]
    feasible = 0
    changed = 0
    for j, original in enumerate(base_cuts[1:-1], start=1):
        payload = f"chunkrag-jitter-v1\0{seed}\0{policy}\0{document_id}\0{j}".encode("utf-8")
        value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
        delta = value % 97 - 48
        if delta == 0:
            delta = 1 if j % 2 == 0 else -1
        remaining = segments - j
        minimum_remaining_tokens = 64 * remaining - (63 if final_short and remaining else 0)
        lower = max(updated[-1] + 64, total - 254 * remaining)
        upper = min(updated[-1] + 254, total - minimum_remaining_tokens)
        if lower > upper:
            raise ValueError(f"Infeasible jitter interval for {document_id} cut {j}")
        can_change = any(position != original for position in range(lower, upper + 1))
        feasible += int(can_change)
        new_cut = min(max(original + delta, lower), upper)
        changed += int(can_change and new_cut != original)
        updated.append(new_cut)
    updated.append(total)
    lengths = [right - left for left, right in zip(updated, updated[1:])]
    if any(length > 254 for length in lengths):
        raise ValueError("Jitter produced an overlong segment")
    for index, length in enumerate(lengths):
        if length < 64 and not (final_short and index == len(lengths) - 1):
            raise ValueError("Jitter produced an invalid short segment")
    generation_hash = canonical_json_hash({
        "seed": seed, "policy": policy, "document_id": document_id,
        "base_cuts": base_cuts, "new_cuts": updated,
    })
    return JitterResult(tuple(updated), feasible, changed, generation_hash)


def validate_changed_fraction(results: list[JitterResult], minimum: float = 0.8) -> float:
    feasible = sum(result.feasible for result in results)
    changed = sum(result.changed for result in results)
    fraction = 1.0 if feasible == 0 else changed / feasible
    if fraction < minimum:
        raise ValueError(f"Changed-boundary fraction {fraction:.6f} is below {minimum}")
    return fraction
