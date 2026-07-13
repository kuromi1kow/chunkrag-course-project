"""Immutable artifact store and hash-chain validation (Specification Sections 23--25)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .canonical import (
    atomic_write_json,
    atomic_write_jsonl,
    canonical_json_hash,
    file_sha256,
    read_json,
    read_jsonl,
)
from .constants import ARTIFACT_SUBDIRECTORIES, PROTOCOL_ID, PROTOCOL_SHA256
from .schemas import validate_record, validate_records


class ArtifactError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    path: str
    sha256: str
    schema: str
    records: int

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "sha256": self.sha256, "schema": self.schema, "records": self.records}


class ArtifactStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def initialize(self) -> None:
        for directory in ARTIFACT_SUBDIRECTORIES:
            (self.root / directory).mkdir(parents=True, exist_ok=True)

    def relative(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

    def write_json(self, relative: str, payload: Mapping[str, Any], schema: str) -> ArtifactRef:
        validate_record(schema, payload)
        path = self.root / relative
        digest = atomic_write_json(path, payload)
        return ArtifactRef(self.relative(path), digest, schema, 1)

    def write_jsonl(
        self,
        relative: str,
        records: Iterable[Mapping[str, Any]],
        schema: str,
        primary_id: str,
    ) -> ArtifactRef:
        rows = list(records)
        validate_records(schema, rows)
        path = self.root / relative
        digest = atomic_write_jsonl(path, rows, primary_id)
        return ArtifactRef(self.relative(path), digest, schema, len(rows))

    def validate_ref(self, reference: ArtifactRef) -> None:
        path = self.root / reference.path
        if not path.is_file():
            raise ArtifactError(f"Missing artifact: {reference.path}")
        actual = file_sha256(path)
        if actual != reference.sha256:
            raise ArtifactError(f"Artifact hash mismatch for {reference.path}: {actual}")
        rows = read_jsonl(path) if path.suffix == ".jsonl" else [read_json(path)]
        if len(rows) != reference.records:
            raise ArtifactError(f"Artifact record count mismatch: {reference.path}")
        validate_records(reference.schema, rows)

    def lock_read_only(self) -> None:
        for path in sorted(self.root.rglob("*")):
            if path.is_file():
                path.chmod(path.stat().st_mode & ~0o222)


def upstream_hash(record: Mapping[str, Any]) -> str:
    value = record.get("upstream_hash")
    if not isinstance(value, str) or len(value) != 64:
        raise ArtifactError("Record lacks an immediate upstream hash")
    return value


def validate_hash_chain(levels: list[tuple[str, list[Mapping[str, Any]]]], root_hash: str) -> None:
    expected = root_hash
    for label, records in levels:
        found = {upstream_hash(row) for row in records}
        if found != {expected}:
            raise ArtifactError(f"Mixed or broken upstream hash at {label}: {sorted(found)}")
        expected = canonical_json_hash(records)


def validate_record_links(
    upstream_records: list[Mapping[str, Any]], downstream_records: list[Mapping[str, Any]],
    *, downstream_field: str = "upstream_hash",
) -> None:
    upstream_hashes = {canonical_json_hash(record) for record in upstream_records}
    missing = {
        str(record[downstream_field]) for record in downstream_records
        if str(record.get(downstream_field)) not in upstream_hashes
    }
    if missing:
        raise ArtifactError(f"Downstream records reference {len(missing)} unknown upstream hashes")


def run_manifest_template(
    *, git_commit: str, source_hash: str, config_hash: str, environment_hash: str,
    planned_counts: Mapping[str, int], hardware: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": PROTOCOL_ID,
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": PROTOCOL_SHA256,
        "git_commit": git_commit,
        "dirty_worktree": False,
        "source_hash": source_hash,
        "config_hash": config_hash,
        "environment_lock_hash": environment_hash,
        "artifact_hashes": {},
        "model_snapshots": {},
        "hardware": dict(hardware),
        "started_utc": None,
        "ended_utc": None,
        "planned_counts": dict(planned_counts),
        "completed_counts": {},
        "shards": [],
        "status": "planned",
        "failures": [],
    }
