from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class PairedEstimate:
    n: int
    mean_difference: float
    ci_low: float
    ci_high: float
    positive_n: int
    negative_n: int
    tied_n: int
    cohen_dz: float | None
    randomization_p: float | None

    def as_dict(self) -> dict[str, int | float | None]:
        return {
            "n": self.n,
            "mean_difference": self.mean_difference,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "positive_n": self.positive_n,
            "negative_n": self.negative_n,
            "tied_n": self.tied_n,
            "cohen_dz": self.cohen_dz,
            "randomization_p": self.randomization_p,
        }


def paired_bootstrap_ci(
    differences: Sequence[float],
    *,
    draws: int,
    seed: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("Paired bootstrap requires a non-empty one-dimensional vector")
    if draws <= 0:
        raise ValueError("Bootstrap draws must be positive")
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=float)
    batch_size = min(2_000, draws)
    for start in range(0, draws, batch_size):
        stop = min(draws, start + batch_size)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))


def paired_sign_flip_p(
    differences: Sequence[float],
    *,
    draws: int,
    seed: int,
) -> float:
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("Sign-flip test requires a non-empty one-dimensional vector")
    observed = abs(float(values.mean()))
    if observed == 0.0:
        return 1.0
    rng = np.random.default_rng(seed)
    extreme = 0
    batch_size = min(2_000, draws)
    for start in range(0, draws, batch_size):
        batch = min(batch_size, draws - start)
        signs = rng.integers(0, 2, size=(batch, len(values)), dtype=np.int8) * 2 - 1
        randomized = np.abs((signs * values).mean(axis=1))
        extreme += int(np.count_nonzero(randomized >= observed - 1e-15))
    return (extreme + 1) / (draws + 1)


def paired_estimate(
    differences: Sequence[float],
    *,
    bootstrap_draws: int,
    bootstrap_seed: int,
    randomization_draws: int | None = None,
    randomization_seed: int | None = None,
) -> PairedEstimate:
    values = np.asarray(differences, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("Paired differences contain non-finite values")
    low, high = paired_bootstrap_ci(
        values,
        draws=bootstrap_draws,
        seed=bootstrap_seed,
    )
    sample_sd = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    cohen_dz = float(values.mean() / sample_sd) if sample_sd > 0.0 else None
    if randomization_draws is None:
        randomization_p = None
    else:
        if randomization_seed is None:
            raise ValueError("A randomization seed is required when draws are requested")
        randomization_p = paired_sign_flip_p(
            values,
            draws=randomization_draws,
            seed=randomization_seed,
        )
    return PairedEstimate(
        n=len(values),
        mean_difference=float(values.mean()),
        ci_low=low,
        ci_high=high,
        positive_n=int(np.count_nonzero(values > 0.0)),
        negative_n=int(np.count_nonzero(values < 0.0)),
        tied_n=int(np.count_nonzero(values == 0.0)),
        cohen_dz=cohen_dz,
        randomization_p=randomization_p,
    )
