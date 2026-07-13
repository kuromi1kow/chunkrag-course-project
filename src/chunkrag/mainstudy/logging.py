"""Structured append-only execution logging (Specification Sections 23 and 28)."""

from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path
from typing import Any

from .canonical import canonical_json_bytes


class JsonlLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: str, **fields: Any) -> None:
        row = {
            "event": event,
            "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            **fields,
        }
        with self.path.open("ab") as handle:
            handle.write(canonical_json_bytes(row))
            handle.flush()


def configure_console_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
