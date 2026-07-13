"""Deterministic seed and Torch controls (Specification Sections 20 and 25)."""

from __future__ import annotations

import hashlib
import os
import random
from typing import Any

from .constants import MASTER_SEED


def derived_seed(test_id: str, procedure: str, master_seed: int = MASTER_SEED) -> int:
    payload = f"{master_seed}\0{test_id}\0{procedure}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def configure_determinism(seed: int = MASTER_SEED, *, require_torch: bool = False) -> dict[str, Any]:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    status: dict[str, Any] = {"seed": seed, "torch": False}
    try:
        import numpy as np

        np.random.seed(seed % (2**32))
        status["numpy"] = True
    except ImportError:
        status["numpy"] = False
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        status["torch"] = True
    except ImportError:
        if require_torch:
            raise
    return status
