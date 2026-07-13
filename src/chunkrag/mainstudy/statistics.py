"""Frozen cluster-aware inference (Specification Sections 8.5 and 20)."""

from __future__ import annotations

import itertools
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .determinism import derived_seed


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    total = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for index, (test_id, value) in enumerate(ordered):
        if not 0 <= value <= 1:
            raise ValueError(f"Invalid p-value for {test_id}: {value}")
        running = max(running, min(1.0, (total - index) * value))
        adjusted[test_id] = running
    return adjusted


def rank_biserial(contrasts: Sequence[float]) -> float:
    positive = sum(value > 0 for value in contrasts)
    negative = sum(value < 0 for value in contrasts)
    nonzero = positive + negative
    return 0.0 if nonzero == 0 else (positive - negative) / nonzero


def cliffs_delta(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right:
        raise ValueError("Cliff's delta requires two nonempty samples")
    wins = sum(a > b for a in left for b in right)
    losses = sum(a < b for a in left for b in right)
    return (wins - losses) / (len(left) * len(right))


def cluster_bootstrap(
    values: Sequence[float], clusters: Sequence[str], test_id: str, *, draws: int = 20_000,
    interval: float = 0.95,
) -> tuple[float, float]:
    if len(values) != len(clusters) or not values:
        raise ValueError("Bootstrap values and clusters must be nonempty and aligned")
    members: dict[str, list[float]] = defaultdict(list)
    for value, cluster in zip(values, clusters):
        members[str(cluster)].append(float(value))
    keys = sorted(members)
    generator = np.random.Generator(np.random.PCG64(derived_seed(test_id, "cluster-bootstrap")))
    estimates = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        sampled = generator.choice(keys, size=len(keys), replace=True)
        rows = [value for key in sampled for value in members[str(key)]]
        estimates[draw] = float(np.mean(rows))
    tail = (1 - interval) / 2
    return float(np.quantile(estimates, tail)), float(np.quantile(estimates, 1 - tail))


def cluster_bootstrap_difference(
    left: Sequence[float], left_clusters: Sequence[str], right: Sequence[float],
    right_clusters: Sequence[str], test_id: str, *, draws: int = 20_000,
) -> tuple[float, float]:
    def grouped(values: Sequence[float], clusters: Sequence[str]) -> dict[str, list[float]]:
        result: dict[str, list[float]] = defaultdict(list)
        for value, cluster in zip(values, clusters):
            result[str(cluster)].append(float(value))
        return result

    left_groups, right_groups = grouped(left, left_clusters), grouped(right, right_clusters)
    left_keys, right_keys = sorted(left_groups), sorted(right_groups)
    generator = np.random.Generator(np.random.PCG64(derived_seed(test_id, "cluster-bootstrap")))
    estimates = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        sampled_left = generator.choice(left_keys, size=len(left_keys), replace=True)
        sampled_right = generator.choice(right_keys, size=len(right_keys), replace=True)
        left_rows = [value for key in sampled_left for value in left_groups[str(key)]]
        right_rows = [value for key in sampled_right for value in right_groups[str(key)]]
        estimates[draw] = float(np.mean(left_rows) - np.mean(right_rows))
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def cluster_sign_flip(
    values: Sequence[float], clusters: Sequence[str], test_id: str, *, draws: int = 99_999,
) -> float:
    if len(values) != len(clusters) or not values:
        raise ValueError("Sign-flip values and clusters must be nonempty and aligned")
    sums: dict[str, float] = defaultdict(float)
    for value, cluster in zip(values, clusters):
        sums[str(cluster)] += float(value)
    contributions = np.array([sums[key] for key in sorted(sums)], dtype=np.float64)
    observed = abs(float(contributions.sum() / len(values)))
    if len(contributions) <= 20:
        statistics = (
            abs(float(np.dot(contributions, np.array(signs, dtype=np.float64)) / len(values)))
            for signs in itertools.product((-1.0, 1.0), repeat=len(contributions))
        )
        all_values = list(statistics)
        return sum(value >= observed - 1e-15 for value in all_values) / len(all_values)
    generator = np.random.Generator(np.random.PCG64(derived_seed(test_id, "cluster-sign-flip")))
    signs = generator.choice((-1.0, 1.0), size=(draws, len(contributions)))
    randomized = np.abs(signs @ contributions / len(values))
    return (1 + int(np.sum(randomized >= observed - 1e-15))) / (draws + 1)


def _cluster_sandwich(x: np.ndarray, residuals: np.ndarray, clusters: Sequence[str]) -> np.ndarray:
    xtx_inv = np.linalg.inv(x.T @ x)
    meat = np.zeros((x.shape[1], x.shape[1]), dtype=np.float64)
    for cluster in sorted(set(clusters)):
        indices = np.array([index for index, value in enumerate(clusters) if value == cluster])
        score = x[indices].T @ residuals[indices]
        meat += np.outer(score, score)
    n, parameters = x.shape
    groups = len(set(clusters))
    if groups <= 1 or n <= parameters:
        raise ValueError("CR1 requires at least two clusters and residual degrees of freedom")
    correction = groups / (groups - 1) * (n - 1) / (n - parameters)
    return correction * xtx_inv @ meat @ xtx_inv


def cr1_intercept(values: Sequence[float], clusters: Sequence[str]) -> tuple[float, float, int]:
    y = np.asarray(values, dtype=np.float64)
    x = np.ones((len(y), 1), dtype=np.float64)
    beta = np.linalg.solve(x.T @ x, x.T @ y)
    covariance = _cluster_sandwich(x, y - x @ beta, clusters)
    return float(beta[0]), float(math.sqrt(max(0.0, covariance[0, 0]))), len(set(clusters)) - 1


def cr1_dataset_interaction(
    values: Sequence[float], is_squad: Sequence[bool], clusters: Sequence[str],
) -> tuple[float, float, int]:
    y = np.asarray(values, dtype=np.float64)
    x = np.column_stack((np.ones(len(y)), np.asarray(is_squad, dtype=np.float64)))
    beta = np.linalg.solve(x.T @ x, x.T @ y)
    covariance = _cluster_sandwich(x, y - x @ beta, clusters)
    return float(beta[1]), float(math.sqrt(max(0.0, covariance[1, 1]))), len(set(clusters)) - 2


def _normal_cdf(value: float) -> float:
    return 0.5 * (1 + math.erf(value / math.sqrt(2)))


def tost(values: Sequence[float], clusters: Sequence[str], margin: float = 2.0) -> dict[str, float]:
    estimate, standard_error, degrees = cr1_intercept(values, clusters)
    if standard_error == 0:
        p_lower = 0.0 if estimate > -margin else 1.0
        p_upper = 0.0 if estimate < margin else 1.0
    else:
        from scipy.stats import t as student_t

        p_lower = float(student_t.sf((estimate + margin) / standard_error, degrees))
        p_upper = float(student_t.cdf((estimate - margin) / standard_error, degrees))
    return {"estimate": estimate, "standard_error": standard_error, "degrees_freedom": float(degrees), "p_lower": p_lower, "p_upper": p_upper, "p_tost": max(p_lower, p_upper)}


def average_ranks(values: Sequence[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        rank = (index + 1 + end) / 2
        for original, _ in ordered[index:end]:
            ranks[original] = rank
        index = end
    return ranks


def spearman(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("Spearman inputs must be aligned and contain at least two values")
    x = np.asarray(average_ranks(left), dtype=np.float64)
    y = np.asarray(average_ranks(right), dtype=np.float64)
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def quadratic_weighted_kappa(left: Sequence[int], right: Sequence[int]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("Kappa inputs must be aligned and nonempty")
    categories = (0, 1, 2)
    observed = Counter(zip(left, right))
    left_counts, right_counts = Counter(left), Counter(right)
    total = len(left)
    weighted_observed = sum(((a - b) ** 2 / 4) * observed[(a, b)] for a in categories for b in categories) / total
    weighted_expected = sum(((a - b) ** 2 / 4) * left_counts[a] * right_counts[b] for a in categories for b in categories) / (total * total)
    if weighted_expected == 0:
        return float("nan")
    return 1 - weighted_observed / weighted_expected


def ordinal_krippendorff_alpha(left: Sequence[int], right: Sequence[int]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("Alpha inputs must be aligned and nonempty")
    observed = sum((a - b) ** 2 for a, b in zip(left, right)) / len(left)
    pooled = list(left) + list(right)
    if len(pooled) < 2:
        return float("nan")
    expected = sum((a - b) ** 2 for index, a in enumerate(pooled) for b in pooled[index + 1:])
    expected /= len(pooled) * (len(pooled) - 1) / 2
    if expected == 0:
        return float("nan")
    return 1 - observed / expected


def judge_acceptance(
    judge: Mapping[str, Sequence[int]], human: Mapping[str, Sequence[int]], *,
    invalid_fraction_by_condition: Mapping[str, float],
) -> dict[str, Any]:
    thresholds = {"correctness": 0.60, "completeness": 0.60, "groundedness": 0.50}
    result: dict[str, Any] = {"dimensions": {}, "parse_failure_pass": all(value <= 0.01 for value in invalid_fraction_by_condition.values())}
    passed = result["parse_failure_pass"]
    for dimension, threshold in thresholds.items():
        j_values = list(judge[dimension])
        h_values = list(human[dimension])
        correlation = spearman(j_values, h_values)
        kappa = quadratic_weighted_kappa(j_values, h_values)
        bias = float(np.mean(j_values) - np.mean(h_values))
        dimension_pass = (
            not math.isnan(correlation) and not math.isnan(kappa)
            and correlation >= threshold and kappa >= 0.50 and abs(bias) <= 0.25
        )
        result["dimensions"][dimension] = {"spearman": correlation, "quadratic_weighted_kappa": kappa, "mean_bias": bias, "pass": dimension_pass}
        passed = passed and dimension_pass
    result["confirmatory"] = bool(passed)
    return result
