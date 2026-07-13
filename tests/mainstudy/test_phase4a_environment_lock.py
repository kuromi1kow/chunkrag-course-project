"""Regression coverage for the Phase 4A CUDA transitive-lock failure."""

from __future__ import annotations

import json
from pathlib import Path


CUDA_13_RUNTIME = {
    "cuda-bindings": "13.3.1",
    "cuda-pathfinder": "1.5.6",
    "cuda-toolkit": "13.0.3.0",
    "nvidia-cublas": "13.1.1.3",
    "nvidia-cuda-cupti": "13.0.85",
    "nvidia-cuda-nvrtc": "13.0.88",
    "nvidia-cuda-runtime": "13.0.96",
    "nvidia-cudnn-cu13": "9.20.0.48",
    "nvidia-cufft": "12.0.0.61",
    "nvidia-cufile": "1.15.1.6",
    "nvidia-curand": "10.4.0.35",
    "nvidia-cusolver": "12.0.4.66",
    "nvidia-cusparse": "12.6.3.3",
    "nvidia-cusparselt-cu13": "0.8.1",
    "nvidia-nccl-cu13": "2.29.7",
    "nvidia-nvjitlink": "13.3.33",
    "nvidia-nvshmem-cu13": "3.4.5",
    "nvidia-nvtx": "13.0.85",
    "triton": "3.7.1",
}


def test_resolved_lock_contains_torch_cuda_runtime_closure() -> None:
    root = Path(__file__).resolve().parents[2]
    payload = json.loads((root / "requirements-main-study.transitive.json").read_text(encoding="utf-8"))
    packages = {str(row["name"]).lower(): str(row["version"]) for row in payload["packages"]}

    assert packages["torch"] == "2.13.0"
    assert {name: packages.get(name) for name in CUDA_13_RUNTIME} == CUDA_13_RUNTIME

