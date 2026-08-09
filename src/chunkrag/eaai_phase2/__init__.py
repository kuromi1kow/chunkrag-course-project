"""Prospective EAAI Phase 2 experiment support.

This package is intentionally isolated from the frozen retrospective and
prospective artifact builders.
"""

from chunkrag.eaai_phase2.constants import (
    BASELINE_TREE_SHA256,
    CHUNKERS,
    PROTOCOL_COMMIT,
    RUN_ID,
)

__all__ = ["BASELINE_TREE_SHA256", "CHUNKERS", "PROTOCOL_COMMIT", "RUN_ID"]
