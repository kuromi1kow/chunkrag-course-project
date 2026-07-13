"""Outcome-free sample-size sensitivity simulation (Specification Section 8.5 and E0)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .determinism import derived_seed


def sensitivity_report(
    dataset: str, cluster_sizes: Sequence[int], *, draws: int = 100_000,
) -> dict[str, Any]:
    from scipy.stats import t as student_t

    groups = len(cluster_sizes)
    critical = float(student_t.ppf(1 - (0.05 / 12) / 2, groups - 1))
    results: list[dict[str, Any]] = []
    for standard_deviation in (15.0, 20.0, 25.0):
        for icc in (0.0, 0.1, 0.2):
            test_id = f"design-sensitivity:{dataset}:{standard_deviation:g}:{icc:g}"
            rng = np.random.Generator(np.random.PCG64(derived_seed(test_id, "power-simulation")))
            cluster_sd = standard_deviation * np.sqrt(icc)
            residual_sd = standard_deviation * np.sqrt(1 - icc)
            null_means = np.empty(draws, dtype=np.float64)
            standard_errors = np.empty(draws, dtype=np.float64)
            for draw in range(draws):
                group_effects = rng.normal(0, cluster_sd, size=groups)
                rows: list[float] = []
                labels: list[int] = []
                for group, size in enumerate(cluster_sizes):
                    rows.extend(group_effects[group] + rng.normal(0, residual_sd, size=size))
                    labels.extend([group] * size)
                values = np.asarray(rows)
                null_means[draw] = values.mean()
                cluster_sums = np.array([values[np.asarray(labels) == group].sum() for group in range(groups)])
                cluster_counts = np.asarray(cluster_sizes)
                centered = cluster_sums - null_means[draw] * cluster_counts
                standard_errors[draw] = np.sqrt((groups / (groups - 1)) * np.sum(centered**2)) / len(values)
            minimum_effect = None
            for effect in np.arange(0, 10.0001, 0.25):
                statistics = np.divide(null_means + effect, standard_errors, out=np.zeros_like(null_means), where=standard_errors > 0)
                if float(np.mean(np.abs(statistics) > critical)) >= 0.80:
                    minimum_effect = float(effect)
                    break
            results.append({"standard_deviation": standard_deviation, "icc": icc, "minimum_effect_80_power": minimum_effect})
    return {"dataset": dataset, "draws": draws, "alpha": 0.05 / 12, "cluster_sizes": list(cluster_sizes), "results": results}
