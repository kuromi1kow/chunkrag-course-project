"""E0--E7 work-item planning and dependency gates (Specification Sections 26 and 29)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

from .constants import (
    DATASET_ORDER,
    EXPECTED_QUESTION_COUNTS,
    EXPERIMENT_DEPENDENCIES,
    EXPERIMENT_ORDER,
    JITTER_SEEDS,
    POLICY_ORDER,
    STRUCTURED_POLICIES,
)
from .protocol import ProtocolError


@dataclass(frozen=True, slots=True)
class WorkItem:
    experiment: str
    dataset: str | None
    condition_id: str
    shard_index: int | None
    expected_records: int
    prerequisites: tuple[str, ...]

    @property
    def work_id(self) -> str:
        parts = [self.experiment, self.dataset or "global", self.condition_id]
        if self.shard_index is not None:
            parts.append(f"part-{self.shard_index:03d}")
        return "/".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "work_id": self.work_id}


def condition_ids_e1() -> list[str]:
    controls = [f"{policy}-jitter-{seed}" for policy in POLICY_ORDER for seed in JITTER_SEEDS]
    return list(POLICY_ORDER) + controls


def condition_ids_e2() -> list[str]:
    boundary = [policy for policy in STRUCTURED_POLICIES]
    boundary += [f"{policy}-jitter-{seed}" for policy in STRUCTURED_POLICIES for seed in JITTER_SEEDS]
    exposure = [f"{policy}__{packing}" for packing in ("operational-1024", "matched-1024", "operational-4096") for policy in POLICY_ORDER]
    boundary_named = [f"{condition}__matched-4096" for condition in boundary]
    fixed = ["fixed192__matched-4096"]
    result = boundary_named + exposure + fixed
    if len(result) != 31 or len(set(result)) != 31:
        raise ProtocolError("E2 condition registry must contain exactly 31 unique conditions")
    return result


def _shards(dataset: str) -> range:
    count = EXPECTED_QUESTION_COUNTS[dataset]
    return range((count + 49) // 50)


def plan_experiment(experiment: str) -> list[WorkItem]:
    if experiment not in EXPERIMENT_ORDER:
        raise ProtocolError(f"Unknown experiment: {experiment}")
    prerequisites = EXPERIMENT_DEPENDENCIES[experiment]
    items: list[WorkItem] = []
    if experiment == "E0":
        items = [WorkItem("E0", dataset, "materialize", None, EXPECTED_QUESTION_COUNTS[dataset], prerequisites) for dataset in DATASET_ORDER]
        items.append(WorkItem("E0", None, "finalize-manifest", None, 1, prerequisites))
        return items
    if experiment == "E1":
        for dataset in DATASET_ORDER:
            for condition in condition_ids_e1():
                items.append(WorkItem("E1", dataset, condition, None, EXPECTED_QUESTION_COUNTS[dataset], prerequisites))
        return items
    if experiment == "E2":
        for dataset in DATASET_ORDER:
            for condition in condition_ids_e2():
                for shard in _shards(dataset):
                    items.append(WorkItem("E2", dataset, condition, shard, min(50, EXPECTED_QUESTION_COUNTS[dataset] - shard * 50), prerequisites))
        return items
    if experiment == "E3":
        for dataset in DATASET_ORDER:
            for condition in ("gold-1024", "gold-4096"):
                for shard in _shards(dataset):
                    items.append(WorkItem("E3", dataset, condition, shard, min(50, EXPECTED_QUESTION_COUNTS[dataset] - shard * 50), prerequisites))
        return items
    if experiment == "E4":
        for condition in [*condition_ids_e2(), "gold-1024", "gold-4096"]:
            for shard in range(6):
                items.append(WorkItem("E4", "techqa", f"judge__{condition}", shard, 50, prerequisites))
        items.append(WorkItem("E4", "techqa", "human-package", None, 360, prerequisites))
        items.append(WorkItem("E4", "techqa", "human-validation", None, 1, prerequisites))
        return items
    if experiment == "E5":
        for dataset in DATASET_ORDER:
            for embedder in ("bge", "minilm"):
                for retriever in ("dense", "hybrid", "hybrid-rerank"):
                    for policy in POLICY_ORDER:
                        items.append(WorkItem("E5", dataset, f"{embedder}__{retriever}__{policy}", None, EXPECTED_QUESTION_COUNTS[dataset], prerequisites))
        return items
    if experiment == "E6":
        for dataset in ("squad_v2", "hotpot_qa"):
            for condition in POLICY_ORDER:
                for shard in _shards(dataset):
                    items.append(WorkItem("E6", dataset, f"{condition}__matched-4096", shard, 50, prerequisites))
            for shard in _shards(dataset):
                items.append(WorkItem("E6", dataset, "gold-4096", shard, 50, prerequisites))
        return items
    return [WorkItem("E7", None, "audit", None, 1, prerequisites)]


def full_plan() -> list[WorkItem]:
    return [item for experiment in EXPERIMENT_ORDER for item in plan_experiment(experiment)]


def filter_plan(
    items: Iterable[WorkItem], *, dataset: str | None = None, condition_id: str | None = None,
    shard_index: int | None = None,
) -> list[WorkItem]:
    return [
        item for item in items
        if (dataset is None or item.dataset == dataset)
        and (condition_id is None or item.condition_id == condition_id)
        and (shard_index is None or item.shard_index == shard_index)
    ]


def validate_dependencies(experiment: str, completed: Iterable[str]) -> None:
    completed_set = set(completed)
    missing = set(EXPERIMENT_DEPENDENCIES[experiment]) - completed_set
    if missing:
        raise ProtocolError(f"{experiment} missing completed prerequisites: {sorted(missing)}")
