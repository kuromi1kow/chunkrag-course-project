from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable


def canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def write_immutable_bytes(path: str | Path, payload: bytes) -> str:
    output = Path(path)
    expected_hash = sha256_bytes(payload)
    if output.exists():
        actual_hash = sha256_file(output)
        if actual_hash != expected_hash:
            raise FileExistsError(
                f"Immutable artifact conflict at {output}: {actual_hash} != {expected_hash}"
            )
        return actual_hash
    _atomic_write(output, payload)
    return expected_hash


def write_immutable_json(path: str | Path, payload: Any) -> str:
    return write_immutable_bytes(path, canonical_json_bytes(payload))


def write_immutable_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> str:
    payload = b"".join(canonical_json_bytes(row) for row in rows)
    return write_immutable_bytes(path, payload)


def add_row_hash(payload: dict[str, Any]) -> dict[str, Any]:
    if "row_sha256" in payload:
        raise ValueError("row_sha256 must not be supplied before hashing")
    result = dict(payload)
    result["row_sha256"] = sha256_bytes(canonical_json_bytes(payload))
    return result


def validate_row_hash(payload: dict[str, Any]) -> None:
    row_hash = payload.get("row_sha256")
    if not isinstance(row_hash, str):
        raise ValueError("Missing row_sha256")
    unhashed = dict(payload)
    del unhashed["row_sha256"]
    expected = sha256_bytes(canonical_json_bytes(unhashed))
    if row_hash != expected:
        raise ValueError(f"Row hash mismatch: {row_hash} != {expected}")


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"Expected object at {path}:{line_number}")
            yield row
