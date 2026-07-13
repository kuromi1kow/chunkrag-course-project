"""Google Colab/Drive execution planning (Specification Section 28)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from .constants import PROTOCOL_SHA256
from .environment import require_canonical_a100
from .protocol import ProtocolError


@dataclass(frozen=True, slots=True)
class ColabInvocation:
    protocol_sha256: str
    git_commit: str
    model: str
    dataset: str
    condition_id: str
    shard_index: int
    question_manifest_hash: str
    upstream_hash: str

    def validate(self) -> None:
        if self.protocol_sha256 != PROTOCOL_SHA256:
            raise ProtocolError("Colab invocation protocol mismatch")
        if self.shard_index < 0:
            raise ProtocolError("Colab shard index must be nonnegative")
        for name in ("git_commit", "model", "dataset", "condition_id", "question_manifest_hash", "upstream_hash"):
            if not getattr(self, name):
                raise ProtocolError(f"Colab invocation missing {name}")


def drive_shard_root(drive_root: str, invocation: ColabInvocation) -> PurePosixPath:
    invocation.validate()
    return PurePosixPath(drive_root) / invocation.git_commit / invocation.model / invocation.dataset / invocation.condition_id


def validate_colab_runtime(
    runtime_manifest: dict[str, Any], *, expected_environment_hash: str, expected_git_commit: str,
) -> None:
    require_canonical_a100(runtime_manifest)
    if runtime_manifest.get("environment_hash") != expected_environment_hash:
        raise ProtocolError("Colab environment lock hash mismatch")
    if runtime_manifest.get("git_commit") != expected_git_commit:
        raise ProtocolError("Colab Git commit mismatch")
    if runtime_manifest.get("protocol_sha256") != PROTOCOL_SHA256:
        raise ProtocolError("Colab protocol hash mismatch")
