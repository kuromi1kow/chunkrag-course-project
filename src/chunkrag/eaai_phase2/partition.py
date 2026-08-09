from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable

from chunkrag.eaai_phase2.constants import (
    DEVELOPMENT_SIZE,
    EXPECTED_ELIGIBLE_ROWS,
    HELDOUT_SIZE,
    PARTITION_SALT,
    RESERVE_SIZE,
)
from chunkrag.eaai_phase2.io import canonical_json_bytes, sha256_bytes


@dataclass(frozen=True, slots=True)
class FrozenPartition:
    development: tuple[str, ...]
    heldout_test: tuple[str, ...]
    reserve: tuple[str, ...]
    partition_sha256: str

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "salt": PARTITION_SALT,
            "development": list(self.development),
            "heldout_test": list(self.heldout_test),
            "reserve": list(self.reserve),
            "partition_sha256": self.partition_sha256,
        }

    def ids_for(self, split: str) -> tuple[str, ...]:
        if split == "development":
            return self.development
        if split == "heldout_test":
            return self.heldout_test
        if split == "reserve":
            return self.reserve
        raise ValueError(f"Unsupported Phase 2 split: {split}")


def partition_digest(question_id: str) -> str:
    payload = f"{PARTITION_SALT}\0{question_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def make_frozen_partition(question_ids: Iterable[str]) -> FrozenPartition:
    ids = [str(value) for value in question_ids]
    if len(ids) != EXPECTED_ELIGIBLE_ROWS:
        raise ValueError(f"Expected {EXPECTED_ELIGIBLE_ROWS} eligible IDs, found {len(ids)}")
    if len(set(ids)) != len(ids):
        raise ValueError("TechQA eligible question IDs are not unique")
    ordered = sorted(ids, key=lambda value: (partition_digest(value), value))
    development = tuple(ordered[:DEVELOPMENT_SIZE])
    heldout = tuple(ordered[DEVELOPMENT_SIZE : DEVELOPMENT_SIZE + HELDOUT_SIZE])
    reserve = tuple(ordered[DEVELOPMENT_SIZE + HELDOUT_SIZE :])
    if len(reserve) != RESERVE_SIZE:
        raise AssertionError(f"Expected {RESERVE_SIZE} reserve IDs, found {len(reserve)}")
    public_mapping = {
        "salt": PARTITION_SALT,
        "development": list(development),
        "heldout_test": list(heldout),
        "reserve": list(reserve),
    }
    digest = sha256_bytes(canonical_json_bytes(public_mapping))
    return FrozenPartition(development, heldout, reserve, digest)


def public_partition_summary(partition: FrozenPartition) -> dict[str, object]:
    return {
        "schema_version": 1,
        "algorithm": 'sort sha256("eaai-phase2-techqa-v1\\0" + question_id)',
        "counts": {
            "development": len(partition.development),
            "heldout_test": len(partition.heldout_test),
            "reserve": len(partition.reserve),
        },
        "partition_sha256": partition.partition_sha256,
        "contains_benchmark_ids": False,
    }
