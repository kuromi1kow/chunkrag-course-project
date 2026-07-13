"""Canonical serialization and hashing (Specification Sections 23--25)."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unicodedata
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


class CanonicalizationError(ValueError):
    """Raised when an object cannot be represented by the frozen canonical format."""


def nfc(value: str) -> str:
    return unicodedata.normalize("NFC", value)


def _normalize(value: Any) -> Any:
    if isinstance(value, str):
        return nfc(value)
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise CanonicalizationError("NaN and Infinity are forbidden")
        return value
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise CanonicalizationError("Canonical JSON object keys must be strings")
        return {nfc(key): _normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    raise CanonicalizationError(f"Unsupported canonical JSON type: {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    normalized = _normalize(value)
    text = json.dumps(
        normalized,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return (text + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json_hash(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def canonical_jsonl_bytes(records: Iterable[Mapping[str, Any]], primary_id: str) -> bytes:
    rows = list(records)
    if any(primary_id not in row for row in rows):
        raise CanonicalizationError(f"Missing JSONL primary ID: {primary_id}")
    ordered = sorted(rows, key=lambda row: str(row[primary_id]))
    ids = [str(row[primary_id]) for row in ordered]
    if len(ids) != len(set(ids)):
        raise CanonicalizationError(f"Duplicate JSONL primary IDs: {primary_id}")
    return b"".join(canonical_json_bytes(row) for row in ordered)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(root: Path, paths: Sequence[Path] | None = None) -> str:
    selected = paths if paths is not None else [path for path in root.rglob("*") if path.is_file()]
    digest = hashlib.sha256()
    for path in sorted(selected, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_sha256(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def source_sha256(repo_root: Path, environment_lock: Path) -> str:
    declared = subprocess.run(
        ["git", "ls-files", "-z", "--", "src", "scripts", "configs", "tests", "pyproject.toml", str(environment_lock.relative_to(repo_root))],
        cwd=repo_root, check=True, capture_output=True,
    ).stdout.split(b"\0")
    included = [repo_root / item.decode("utf-8") for item in declared if item]
    if environment_lock not in included:
        raise CanonicalizationError("Resolved environment lock is not tracked by Git")
    if not included or any(not path.is_file() for path in included):
        raise CanonicalizationError("Tracked source set is empty or contains a missing file")
    return tree_sha256(repo_root, included)


def identifier_hash(*parts: str | int) -> str:
    payload = b"\0".join(str(part).encode("utf-8") for part in parts)
    return sha256_bytes(payload)


def atomic_write_bytes(path: Path, payload: bytes, *, overwrite: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        if path.read_bytes() == payload:
            return
        raise FileExistsError(f"Refusing to overwrite immutable artifact: {path}")
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def atomic_write_json(path: Path, value: Any, *, overwrite: bool = False) -> str:
    payload = canonical_json_bytes(value)
    atomic_write_bytes(path, payload, overwrite=overwrite)
    return sha256_bytes(payload)


def atomic_write_jsonl(
    path: Path,
    records: Iterable[Mapping[str, Any]],
    primary_id: str,
    *,
    overwrite: bool = False,
) -> str:
    payload = canonical_jsonl_bytes(records, primary_id)
    atomic_write_bytes(path, payload, overwrite=overwrite)
    return sha256_bytes(payload)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise CanonicalizationError(f"Blank JSONL line {line_number}: {path}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise CanonicalizationError(f"JSONL line is not an object: {path}:{line_number}")
            rows.append(row)
    return rows
