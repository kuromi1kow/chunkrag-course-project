"""Hash-validated shard checkpointing and merge (Specification Section 28)."""

from __future__ import annotations

import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from .canonical import (
    atomic_write_json,
    canonical_json_bytes,
    canonical_json_hash,
    read_json,
    read_jsonl,
)
from .constants import PROTOCOL_SHA256, SCHEMA_VERSION
from .schemas import validate_record


class CheckpointError(RuntimeError):
    pass


def shard_question_ids(question_ids: list[str], size: int = 50) -> list[list[str]]:
    ordered = sorted(question_ids)
    if len(ordered) != len(set(ordered)):
        raise CheckpointError("Question IDs must be unique before sharding")
    return [ordered[index:index + size] for index in range(0, len(ordered), size)]


@dataclass(slots=True)
class ShardCheckpoint:
    root: Path
    stage: str
    dataset: str
    condition_id: str
    shard_index: int
    expected_question_ids: list[str]
    config_sha256: str
    environment_hash: str
    schema: str | None = "generation"

    @property
    def stem(self) -> str:
        return f"part-{self.shard_index:03d}"

    @property
    def temp_path(self) -> Path:
        return self.root / f"{self.stem}.jsonl.tmp"

    @property
    def final_path(self) -> Path:
        return self.root / f"{self.stem}.jsonl"

    @property
    def state_path(self) -> Path:
        return self.root / f"{self.stem}.state.json"

    def _state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {
                "schema_version": SCHEMA_VERSION,
                "stage": self.stage,
                "dataset": self.dataset,
                "condition_id": self.condition_id,
                "shard_index": self.shard_index,
                "expected_question_ids": self.expected_question_ids,
                "completed": [],
                "record_hashes": {},
                "protocol_sha256": PROTOCOL_SHA256,
                "config_sha256": self.config_sha256,
                "environment_hash": self.environment_hash,
            }
        state = read_json(self.state_path)
        validate_record("checkpoint", state)
        for key, expected in (
            ("protocol_sha256", PROTOCOL_SHA256),
            ("config_sha256", self.config_sha256),
            ("environment_hash", self.environment_hash),
            ("expected_question_ids", self.expected_question_ids),
        ):
            if state[key] != expected:
                raise CheckpointError(f"Checkpoint mismatch for {key}")
        return state

    def append(self, question_id: str, record: Mapping[str, Any]) -> str:
        if self.final_path.exists():
            raise CheckpointError(f"Shard already finalized: {self.final_path}")
        if question_id not in self.expected_question_ids:
            raise CheckpointError(f"Question {question_id} is outside this shard")
        state = self.validate_partial(lambda row: str(row["question_id"]))
        if self.schema is not None:
            validate_record(self.schema, record)
        record_hash = canonical_json_hash(record)
        if question_id in state["completed"]:
            if state["record_hashes"][question_id] != record_hash:
                raise CheckpointError(f"Conflicting resumed record: {question_id}")
            return record_hash
        self.root.mkdir(parents=True, exist_ok=True)
        with self.temp_path.open("ab") as handle:
            handle.write(canonical_json_bytes(record))
            handle.flush()
            os.fsync(handle.fileno())
        state["completed"].append(question_id)
        state["record_hashes"][question_id] = record_hash
        atomic_write_json(self.state_path, state, overwrite=True)
        return record_hash

    def validate_partial(self, id_getter: Callable[[Mapping[str, Any]], str]) -> dict[str, Any]:
        state = self._state()
        rows: list[dict[str, Any]] = []
        if self.temp_path.exists():
            payload = self.temp_path.read_bytes()
            complete_length = payload.rfind(b"\n") + 1
            if complete_length != len(payload):
                with self.temp_path.open("r+b") as handle:
                    handle.truncate(complete_length)
                    handle.flush()
                    os.fsync(handle.fileno())
                payload = payload[:complete_length]
            for line_number, line in enumerate(payload.splitlines(), start=1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise CheckpointError(f"Invalid completed checkpoint record at line {line_number}") from error
                if not isinstance(row, dict):
                    raise CheckpointError("Checkpoint record is not an object")
                if self.schema is not None:
                    validate_record(self.schema, row)
                rows.append(row)
        observed: dict[str, str] = {}
        for row in rows:
            identifier = id_getter(row)
            if identifier not in self.expected_question_ids or identifier in observed:
                raise CheckpointError(f"Unexpected or duplicate checkpoint ID: {identifier}")
            observed[identifier] = canonical_json_hash(row)
        for identifier, digest in state["record_hashes"].items():
            if observed.get(identifier) != digest:
                raise CheckpointError("Checkpoint state contains a missing or conflicting record")
        if observed != state["record_hashes"]:
            state["completed"] = [qid for qid in self.expected_question_ids if qid in observed]
            state["record_hashes"] = observed
            atomic_write_json(self.state_path, state, overwrite=True)
        return state

    def validate_final(self, id_getter: Callable[[Mapping[str, Any]], str]) -> str:
        if not self.final_path.is_file() or not self.state_path.is_file():
            raise CheckpointError("Final shard and state file must both exist")
        state = self._state()
        rows = read_jsonl(self.final_path)
        observed: dict[str, str] = {}
        for row in rows:
            if self.schema is not None:
                validate_record(self.schema, row)
            identifier = id_getter(row)
            if identifier in observed:
                raise CheckpointError(f"Duplicate final shard ID: {identifier}")
            observed[identifier] = canonical_json_hash(row)
        if observed != state["record_hashes"] or set(observed) != set(self.expected_question_ids):
            raise CheckpointError("Final shard does not match its validated state")
        from .canonical import file_sha256
        return file_sha256(self.final_path)

    def finalize(self, id_getter: Callable[[Mapping[str, Any]], str]) -> Path:
        state = self.validate_partial(id_getter)
        if sorted(state["completed"]) != sorted(self.expected_question_ids):
            missing = sorted(set(self.expected_question_ids) - set(state["completed"]))
            raise CheckpointError(f"Cannot finalize incomplete shard; missing {missing}")
        os.replace(self.temp_path, self.final_path)
        return self.final_path

    def invalidate(self, reason: str) -> tuple[Path | None, Path | None]:
        self.root.mkdir(parents=True, exist_ok=True)
        index = 1
        while (self.root / f"{self.stem}.invalid-{index}.state.json").exists():
            index += 1
        invalid_data = self.root / f"{self.stem}.invalid-{index}.jsonl"
        invalid_state = self.root / f"{self.stem}.invalid-{index}.state.json"
        moved_data = moved_state = None
        if self.temp_path.exists():
            os.replace(self.temp_path, invalid_data)
            moved_data = invalid_data
        if self.state_path.exists():
            state = self._state()
            state["invalidation_reason"] = reason
            from .canonical import atomic_write_json
            atomic_write_json(invalid_state, state)
            self.state_path.unlink()
            moved_state = invalid_state
        return moved_data, moved_state


def merge_shards(
    shard_paths: list[Path], expected_ids: list[str], id_field: str, output_path: Path,
    *, schema: str | None = None, require_state: bool = False,
    expected_config_sha256: str | None = None, expected_environment_hash: str | None = None,
) -> str:
    records: dict[str, dict[str, Any]] = {}
    provenance: set[tuple[str, str, str]] = set()
    for path in sorted(shard_paths):
        if path.suffix != ".jsonl" or path.name.endswith(".tmp"):
            raise CheckpointError(f"Not a finalized shard: {path}")
        for row in read_jsonl(path):
            if schema is not None:
                validate_record(schema, row)
            identifier = str(row[id_field])
            if identifier in records:
                raise CheckpointError(f"Duplicate merged ID: {identifier}")
            records[identifier] = row
        state_path = path.with_name(path.name.removesuffix(".jsonl") + ".state.json")
        if require_state and not state_path.is_file():
            raise CheckpointError(f"Missing shard state: {state_path}")
        if state_path.is_file():
            state = read_json(state_path)
            validate_record("checkpoint", state)
            provenance.add((state["protocol_sha256"], state["config_sha256"], state["environment_hash"]))
            if state["protocol_sha256"] != PROTOCOL_SHA256:
                raise CheckpointError("Merge rejected a noncanonical protocol hash")
            if expected_config_sha256 is not None and state["config_sha256"] != expected_config_sha256:
                raise CheckpointError("Merge rejected a stale configuration hash")
            if expected_environment_hash is not None and state["environment_hash"] != expected_environment_hash:
                raise CheckpointError("Merge rejected a stale environment hash")
            hashes = {str(row[id_field]): canonical_json_hash(row) for row in read_jsonl(path)}
            if hashes != state["record_hashes"]:
                raise CheckpointError(f"Shard state/hash mismatch: {path}")
    if len(provenance) > 1:
        raise CheckpointError("Merge rejected mixed protocol/config/environment provenance")
    if set(records) != set(expected_ids):
        missing = sorted(set(expected_ids) - set(records))
        extra = sorted(set(records) - set(expected_ids))
        raise CheckpointError(f"Merge ID mismatch; missing={missing}, extra={extra}")
    from .canonical import atomic_write_jsonl

    return atomic_write_jsonl(output_path, records.values(), id_field)
