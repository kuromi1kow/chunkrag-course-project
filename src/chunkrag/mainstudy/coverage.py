"""Protocol-to-code coverage registry (Specification Section 33 and Phase 3H)."""

from __future__ import annotations

from pathlib import Path
from typing import Any


COVERAGE: dict[str, dict[str, list[str]]] = {
    "1": {"code": ["protocol.py"], "tests": ["test_protocol_plan.py"]},
    "2-6": {"code": ["analysis.py", "experiments.py"], "tests": ["test_protocol_plan.py", "test_statistics.py"]},
    "7": {"code": ["constants.py", "protocol.py", "environment.py"], "tests": ["test_protocol_plan.py"]},
    "8-10": {"code": ["data.py", "gold.py"], "tests": ["test_data_materialization.py"]},
    "11": {"code": ["chunking.py"], "tests": ["test_chunking_controls.py"]},
    "12": {"code": ["controls.py"], "tests": ["test_chunking_controls.py"]},
    "13": {"code": ["retrieval.py"], "tests": ["test_retrieval_packing.py"]},
    "14-15": {"code": ["generation.py", "prompts.py"], "tests": ["test_generation_prompts.py"]},
    "16": {"code": ["packing.py"], "tests": ["test_retrieval_packing.py"]},
    "17": {"code": ["gold.py", "execution.py"], "tests": ["test_data_materialization.py"]},
    "18-19": {"code": ["evaluation.py"], "tests": ["test_evaluation_human.py"]},
    "20": {"code": ["statistics.py", "analysis.py"], "tests": ["test_statistics.py"]},
    "21-22": {"code": ["evaluation.py", "human.py", "statistics.py"], "tests": ["test_evaluation_human.py"]},
    "23-24": {"code": ["schemas.py", "canonical.py", "artifacts.py"], "tests": ["test_canonical_artifacts.py"]},
    "25": {"code": ["determinism.py", "environment.py", "validation.py"], "tests": ["test_protocol_plan.py"]},
    "26": {"code": ["experiments.py", "stages.py", "execution.py"], "tests": ["test_protocol_plan.py"]},
    "27": {"code": ["environment.py", "requirements-main-study.lock"], "tests": ["test_protocol_plan.py"]},
    "28": {"code": ["checkpoint.py", "colab.py", "runner.py"], "tests": ["test_checkpoint.py"]},
    "29": {"code": ["experiments.py", "runner.py", "analysis.py"], "tests": ["test_protocol_plan.py"]},
    "30": {"code": ["paper.py"], "tests": ["test_protocol_plan.py"]},
    "31-33": {"code": ["analysis.py", "validation.py", "coverage.py"], "tests": ["test_protocol_plan.py"]},
}


def validate_coverage(root: Path | None = None) -> dict[str, Any]:
    covered: set[int] = set()
    for section, mapping in COVERAGE.items():
        start, end = (section.split("-", 1) + [section])[:2] if "-" in section else (section, section)
        covered.update(range(int(start), int(end) + 1))
        if not mapping["code"] or not mapping["tests"]:
            raise ValueError(f"Empty protocol coverage entry: {section}")
    missing = sorted(set(range(1, 34)) - covered)
    if missing:
        raise ValueError(f"Missing protocol section coverage: {missing}")
    if root is not None:
        missing_files: list[str] = []
        for mapping in COVERAGE.values():
            for filename in mapping["code"]:
                candidates = [root / "src" / "chunkrag" / "mainstudy" / filename, root / filename]
                if not any(path.is_file() for path in candidates):
                    missing_files.append(filename)
            for filename in mapping["tests"]:
                if not (root / "tests" / "mainstudy" / filename).is_file():
                    missing_files.append(filename)
        if missing_files:
            raise ValueError(f"Protocol coverage references missing files: {sorted(set(missing_files))}")
    return {"sections": 33, "entries": len(COVERAGE), "missing": []}
