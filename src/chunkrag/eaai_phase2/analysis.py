from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Sequence

import numpy as np

from chunkrag.eaai_phase2.config import load_phase2_config, phase2_paths
from chunkrag.eaai_phase2.constants import (
    BASELINE_TREE_SHA256,
    CHUNKERS,
    CONDITIONS,
    PRIMARY_BOOTSTRAP_DRAWS,
    PRIMARY_BOOTSTRAP_SEED,
    PRIMARY_RANDOMIZATION_DRAWS,
    PRIMARY_RANDOMIZATION_SEED,
    PROTOCOL_COMMIT,
)
from chunkrag.eaai_phase2.gate import gate_probabilities, load_gate
from chunkrag.eaai_phase2.integrity import (
    repository_root,
    require_within,
    verify_baseline,
    verify_clean_paths,
    verify_protocol_commit,
)
from chunkrag.eaai_phase2.io import (
    iter_jsonl,
    read_json,
    sha256_file,
    validate_row_hash,
    write_immutable_json,
)
from chunkrag.eaai_phase2.statistics import paired_estimate


def _generation_index(rows: Sequence[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    index = {
        (str(row["question_id"]), str(row["chunker"]), str(row["condition"])): row
        for row in rows
    }
    if len(index) != len(rows):
        raise RuntimeError("Duplicate generation rows")
    question_ids = sorted({key[0] for key in index})
    if len(question_ids) != 200:
        raise RuntimeError(f"Expected 200 held-out questions, found {len(question_ids)}")
    expected = {
        (question_id, chunker, condition)
        for question_id in question_ids
        for chunker in CHUNKERS
        for condition in CONDITIONS
    }
    if set(index) != expected:
        raise RuntimeError(
            f"Incomplete held-out generation matrix: missing={len(expected - set(index))}, "
            f"extra={len(set(index) - expected)}"
        )
    return index


def _condition_means(
    index: dict[tuple[str, str, str], dict[str, Any]], metric: str
) -> tuple[list[str], dict[str, list[float]]]:
    question_ids = sorted({key[0] for key in index})
    values = {condition: [] for condition in CONDITIONS}
    for question_id in question_ids:
        for condition in CONDITIONS:
            values[condition].append(
                mean(float(index[(question_id, chunker, condition)][metric]) for chunker in CHUNKERS)
            )
    return question_ids, values


def _descriptive_paired(
    differences: Sequence[float], *, seed_offset: int
) -> dict[str, Any]:
    estimate = paired_estimate(
        differences,
        bootstrap_draws=PRIMARY_BOOTSTRAP_DRAWS,
        bootstrap_seed=PRIMARY_BOOTSTRAP_SEED + seed_offset,
    ).as_dict()
    estimate["cohen_dz"] = None
    estimate["randomization_p"] = None
    estimate["classification"] = "secondary_descriptive"
    return estimate


def _primary_analysis(
    rows: Sequence[dict[str, Any]],
    *,
    input_sha256: str,
    gate_sha256: str,
    config_sha256: str,
) -> dict[str, Any]:
    index = _generation_index(rows)
    question_ids, values = _condition_means(index, "f1")
    differences = [
        reranked - hybrid
        for hybrid, reranked in zip(values["hybrid"], values["reranked"], strict=True)
    ]
    estimate = paired_estimate(
        differences,
        bootstrap_draws=PRIMARY_BOOTSTRAP_DRAWS,
        bootstrap_seed=PRIMARY_BOOTSTRAP_SEED,
        randomization_draws=PRIMARY_RANDOMIZATION_DRAWS,
        randomization_seed=PRIMARY_RANDOMIZATION_SEED,
    )
    return {
        "schema_version": 1,
        "study": "eaai_phase2",
        "analysis_classification": "primary_confirmatory",
        "analysis_family": "new_single_test_techqa_reranked_generation",
        "previous_holm_family_modified": False,
        "dataset": "techqa",
        "generator": "qwen",
        "split": "heldout_test",
        "endpoint": "token_f1",
        "sampling_unit": "question",
        "aggregation": "mean over four prespecified chunkers within each question",
        "direction": "reranked_minus_hybrid",
        "condition_means": {
            "hybrid": float(mean(values["hybrid"])),
            "reranked": float(mean(values["reranked"])),
        },
        "estimate": estimate.as_dict(),
        "bootstrap": {
            "method": "paired percentile bootstrap over questions",
            "draws": PRIMARY_BOOTSTRAP_DRAWS,
            "seed": PRIMARY_BOOTSTRAP_SEED,
            "confidence": 0.95,
        },
        "randomization": {
            "method": "two-sided paired sign-flip Monte Carlo test",
            "draws": PRIMARY_RANDOMIZATION_DRAWS,
            "seed": PRIMARY_RANDOMIZATION_SEED,
            "plus_one_correction": True,
        },
        "question_level": [
            {
                "question_id": question_id,
                "hybrid_f1": values["hybrid"][index_value],
                "reranked_f1": values["reranked"][index_value],
                "delta_f1": differences[index_value],
            }
            for index_value, question_id in enumerate(question_ids)
        ],
        "provenance": {
            "input_sha256": input_sha256,
            "gate_sha256": gate_sha256,
            "config_sha256": config_sha256,
            "baseline_tree_sha256": BASELINE_TREE_SHA256,
            "protocol_commit": PROTOCOL_COMMIT,
        },
    }


def _chunker_effects(
    index: dict[tuple[str, str, str], dict[str, Any]], metric: str
) -> dict[str, Any]:
    question_ids = sorted({key[0] for key in index})
    output: dict[str, Any] = {}
    for chunker_index, chunker in enumerate(CHUNKERS):
        hybrid = [float(index[(question_id, chunker, "hybrid")][metric]) for question_id in question_ids]
        reranked = [
            float(index[(question_id, chunker, "reranked")][metric]) for question_id in question_ids
        ]
        differences = [right - left for left, right in zip(hybrid, reranked, strict=True)]
        output[chunker] = {
            "hybrid_mean": float(mean(hybrid)),
            "reranked_mean": float(mean(reranked)),
            "difference": _descriptive_paired(differences, seed_offset=100 + chunker_index),
        }
    return output


def _propagation_analysis(
    index: dict[tuple[str, str, str], dict[str, Any]]
) -> dict[str, Any]:
    tables: dict[str, Any] = {}
    for retrieval_metric in ("all_supporting_docs_found", "answer_string_visible_at_k"):
        counts: Counter[str] = Counter()
        f1_by_retrieval_change: dict[str, list[float]] = defaultdict(list)
        for question_id, chunker, _ in sorted(index):
            if _ != "hybrid":
                continue
            hybrid = index[(question_id, chunker, "hybrid")]
            reranked = index[(question_id, chunker, "reranked")]
            retrieval_delta = float(reranked["retrieval_metrics"][retrieval_metric]) - float(
                hybrid["retrieval_metrics"][retrieval_metric]
            )
            f1_delta = float(reranked["f1"]) - float(hybrid["f1"])
            retrieval_label = "gain" if retrieval_delta > 0 else "loss" if retrieval_delta < 0 else "tie"
            f1_label = "gain" if f1_delta > 0 else "loss" if f1_delta < 0 else "tie"
            counts[f"{retrieval_label}__{f1_label}"] += 1
            f1_by_retrieval_change[retrieval_label].append(f1_delta)
        tables[retrieval_metric] = {
            "cross_tab_counts": dict(sorted(counts.items())),
            "mean_f1_delta_by_retrieval_change": {
                label: float(mean(values)) for label, values in sorted(f1_by_retrieval_change.items())
            },
            "interpretation": "descriptive association; not causal mediation",
        }
    return tables


def _truncation_analysis(
    index: dict[tuple[str, str, str], dict[str, Any]]
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    deltas: dict[str, list[float]] = defaultdict(list)
    for question_id in sorted({key[0] for key in index}):
        for chunker in CHUNKERS:
            hybrid = index[(question_id, chunker, "hybrid")]
            reranked = index[(question_id, chunker, "reranked")]
            label = (
                f"hybrid_{int(bool(hybrid['context_truncated']))}__"
                f"reranked_{int(bool(reranked['context_truncated']))}"
            )
            counts[label] += 1
            deltas[label].append(float(reranked["f1"]) - float(hybrid["f1"]))
    return {
        label: {"n": counts[label], "mean_f1_delta": float(mean(deltas[label]))}
        for label in sorted(counts)
    }


def _adaptive_analysis(
    generation_rows: Sequence[dict[str, Any]],
    retrieval_rows: Sequence[dict[str, Any]],
    *,
    gate_model: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    generation = _generation_index(generation_rows)
    retrieval_index = {
        (str(row["question_id"]), str(row["chunker"])): row for row in retrieval_rows
    }
    if len(retrieval_index) != 200 * len(CHUNKERS):
        raise RuntimeError("Held-out retrieval matrix is incomplete for adaptive analysis")
    ordered_keys = sorted(retrieval_index)
    probabilities = gate_probabilities(
        gate_model,
        [dict(retrieval_index[key]["features"]) for key in ordered_keys],
    )
    decisions: list[dict[str, Any]] = []
    by_question: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    latency_by_question: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    invocation_by_chunker: Counter[str] = Counter()
    for key, probability in zip(ordered_keys, probabilities, strict=True):
        question_id, chunker = key
        invoke = bool(probability >= 0.5)
        selected_condition = "reranked" if invoke else "hybrid"
        hybrid = generation[(question_id, chunker, "hybrid")]
        reranked = generation[(question_id, chunker, "reranked")]
        selected = reranked if invoke else hybrid
        oracle = reranked if float(reranked["f1"]) > float(hybrid["f1"]) else hybrid
        benefit_label = int(float(reranked["f1"]) > float(hybrid["f1"]))
        if invoke:
            invocation_by_chunker[chunker] += 1
        for metric in ("f1", "exact_match"):
            by_question[question_id][f"hybrid_{metric}"].append(float(hybrid[metric]))
            by_question[question_id][f"reranked_{metric}"].append(float(reranked[metric]))
            by_question[question_id][f"adaptive_{metric}"].append(float(selected[metric]))
            by_question[question_id][f"oracle_{metric}"].append(float(oracle[metric]))
        for condition, row in (("hybrid", hybrid), ("reranked", reranked), ("adaptive", selected)):
            latency_by_question[question_id][condition].append(
                float(row["timing_seconds"]["end_to_end_component_sum"])
            )
        decisions.append(
            {
                "question_id": question_id,
                "chunker": chunker,
                "rerank_probability": float(probability),
                "invoke_reranker": invoke,
                "selected_condition": selected_condition,
                "heldout_benefit_label": benefit_label,
                "hybrid_f1": float(hybrid["f1"]),
                "reranked_f1": float(reranked["f1"]),
                "selected_f1": float(selected["f1"]),
            }
        )

    question_ids = sorted(by_question)
    means: dict[str, list[float]] = defaultdict(list)
    latency_means: dict[str, list[float]] = defaultdict(list)
    for question_id in question_ids:
        for system in ("hybrid", "reranked", "adaptive", "oracle"):
            for metric in ("f1", "exact_match"):
                means[f"{system}_{metric}"].append(
                    float(mean(by_question[question_id][f"{system}_{metric}"]))
                )
        for system in ("hybrid", "reranked", "adaptive"):
            latency_means[system].append(float(mean(latency_by_question[question_id][system])))

    adaptive_minus_hybrid = [
        adaptive - hybrid
        for adaptive, hybrid in zip(means["adaptive_f1"], means["hybrid_f1"], strict=True)
    ]
    adaptive_minus_reranked = [
        adaptive - reranked
        for adaptive, reranked in zip(
            means["adaptive_f1"], means["reranked_f1"], strict=True
        )
    ]
    rerank_gain = mean(means["reranked_f1"]) - mean(means["hybrid_f1"])
    adaptive_gain = mean(means["adaptive_f1"]) - mean(means["hybrid_f1"])
    retained_fraction = None if rerank_gain == 0.0 else adaptive_gain / rerank_gain

    labels = np.asarray([row["heldout_benefit_label"] for row in decisions], dtype=int)
    predicted = np.asarray([int(row["invoke_reranker"]) for row in decisions], dtype=int)
    probabilities_array = np.asarray([row["rerank_probability"] for row in decisions], dtype=float)
    classification: dict[str, Any] = {
        "n": len(decisions),
        "positive_n": int(labels.sum()),
        "accuracy": float(np.mean(predicted == labels)),
        "brier_score": float(np.mean((probabilities_array - labels) ** 2)),
    }
    if len(set(labels.tolist())) == 2:
        from sklearn.metrics import (
            balanced_accuracy_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        classification.update(
            {
                "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
                "precision": float(precision_score(labels, predicted, zero_division=0)),
                "recall": float(recall_score(labels, predicted, zero_division=0)),
                "roc_auc": float(roc_auc_score(labels, probabilities_array)),
            }
        )
    else:
        classification.update(
            {
                "balanced_accuracy": None,
                "precision": None,
                "recall": None,
                "roc_auc": None,
            }
        )

    result = {
        "classification": "secondary_and_exploratory",
        "threshold": 0.5,
        "question_level_system_means": {
            key: float(mean(value)) for key, value in sorted(means.items())
        },
        "adaptive_minus_hybrid_f1": _descriptive_paired(
            adaptive_minus_hybrid, seed_offset=500
        ),
        "adaptive_minus_reranked_f1": _descriptive_paired(
            adaptive_minus_reranked, seed_offset=501
        ),
        "reranker_invocation_rate": float(mean(row["invoke_reranker"] for row in decisions)),
        "reranker_invocation_rate_by_chunker": {
            chunker: invocation_by_chunker[chunker] / 200 for chunker in CHUNKERS
        },
        "fraction_of_always_rerank_f1_change_retained": retained_fraction,
        "latency_component_sum_seconds": {
            system: float(mean(values)) for system, values in sorted(latency_means.items())
        },
        "latency_saved_vs_always_rerank_seconds": float(
            mean(latency_means["reranked"]) - mean(latency_means["adaptive"])
        ),
        "oracle_is_unattainable_diagnostic": True,
        "heldout_benefit_classification": classification,
    }
    return result, decisions


def analyze_generator(
    generator_name: str,
    *,
    generation_path: Path,
    retrieval_path: Path,
    gate_model: Any,
    gate_sha256: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    generation_rows = list(iter_jsonl(generation_path))
    retrieval_rows = list(iter_jsonl(retrieval_path))
    for row in [*generation_rows, *retrieval_rows]:
        validate_row_hash(row)
    for row in generation_rows:
        if row.get("generator") != generator_name or row.get("split") != "heldout_test":
            raise RuntimeError("Generation row identity differs from the requested held-out analysis")
        if row.get("gate_sha256") != gate_sha256:
            raise RuntimeError("Generation row was not bound to the requested frozen gate")
    for row in retrieval_rows:
        if row.get("split") != "heldout_test":
            raise RuntimeError("Retrieval row is not from heldout_test")
        if row.get("gate_sha256") != gate_sha256:
            raise RuntimeError("Retrieval row was not bound to the requested frozen gate")
    index = _generation_index(generation_rows)
    _, f1_values = _condition_means(index, "f1")
    _, em_values = _condition_means(index, "exact_match")
    f1_differences = [
        reranked - hybrid
        for hybrid, reranked in zip(f1_values["hybrid"], f1_values["reranked"], strict=True)
    ]
    em_differences = [
        reranked - hybrid
        for hybrid, reranked in zip(em_values["hybrid"], em_values["reranked"], strict=True)
    ]
    adaptive, decisions = _adaptive_analysis(
        generation_rows,
        retrieval_rows,
        gate_model=gate_model,
    )
    return (
        {
            "schema_version": 1,
            "study": "eaai_phase2",
            "analysis_classification": (
                "secondary_prespecified" if generator_name == "qwen" else "secondary_replication"
            ),
            "generator": generator_name,
            "split": "heldout_test",
            "global_f1": {
                "hybrid_mean": float(mean(f1_values["hybrid"])),
                "reranked_mean": float(mean(f1_values["reranked"])),
                "difference": _descriptive_paired(f1_differences, seed_offset=10),
            },
            "global_exact_match": {
                "hybrid_mean": float(mean(em_values["hybrid"])),
                "reranked_mean": float(mean(em_values["reranked"])),
                "difference": _descriptive_paired(em_differences, seed_offset=11),
            },
            "chunker_f1": _chunker_effects(index, "f1"),
            "chunker_exact_match": _chunker_effects(index, "exact_match"),
            "retrieval_to_generation": _propagation_analysis(index),
            "truncation_strata": _truncation_analysis(index),
            "adaptive": adaptive,
            "provenance": {
                "generation_sha256": sha256_file(generation_path),
                "retrieval_sha256": sha256_file(retrieval_path),
                "baseline_tree_sha256": BASELINE_TREE_SHA256,
                "protocol_commit": PROTOCOL_COMMIT,
            },
        },
        decisions,
    )


def run_phase2_analysis(
    config_path: str | Path,
    *,
    repository: str | Path | None = None,
    include_mistral: bool = False,
) -> dict[str, str]:
    repo = Path(repository).resolve() if repository else repository_root()
    verify_baseline(repo)
    verify_protocol_commit(repo)
    from chunkrag.eaai_phase2.experiment import SCIENTIFIC_PATHS

    verify_clean_paths(SCIENTIFIC_PATHS, repo)
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = repo / config_file
    config, config_sha256 = load_phase2_config(config_file)
    paths = phase2_paths(repo, config)
    gate_manifest_path = paths.run_artifacts / "gate" / "gate_manifest.json"
    gate_manifest = read_json(gate_manifest_path)
    gate_model_path = require_within(repo / gate_manifest["model_path"], paths.run_artifacts)
    gate_model = load_gate(gate_model_path, str(gate_manifest["model_sha256"]))
    retrieval_path = paths.run_results / "retrieval_heldout_test.jsonl"
    qwen_path = paths.run_results / "generation_qwen_heldout_test.jsonl"
    if not qwen_path.is_file() or not retrieval_path.is_file():
        raise FileNotFoundError("Complete held-out Qwen retrieval and generation are required")
    qwen_rows = list(iter_jsonl(qwen_path))
    for row in qwen_rows:
        validate_row_hash(row)
        if row.get("gate_sha256") != gate_manifest["model_sha256"]:
            raise RuntimeError("Held-out Qwen row was not bound to the frozen gate")
    primary = _primary_analysis(
        qwen_rows,
        input_sha256=sha256_file(qwen_path),
        gate_sha256=str(gate_manifest["model_sha256"]),
        config_sha256=config_sha256,
    )
    primary_path = paths.run_results / "primary_confirmatory_analysis.json"
    write_immutable_json(primary_path, primary)

    qwen_secondary, qwen_decisions = analyze_generator(
        "qwen",
        generation_path=qwen_path,
        retrieval_path=retrieval_path,
        gate_model=gate_model,
        gate_sha256=str(gate_manifest["model_sha256"]),
    )
    qwen_secondary_path = paths.run_results / "secondary_qwen_analysis.json"
    qwen_decisions_path = paths.run_artifacts / "analysis" / "qwen_gate_decisions.json"
    write_immutable_json(qwen_secondary_path, qwen_secondary)
    write_immutable_json(qwen_decisions_path, qwen_decisions)
    outputs = {
        "primary": str(primary_path),
        "qwen_secondary": str(qwen_secondary_path),
        "qwen_gate_decisions": str(qwen_decisions_path),
    }

    if include_mistral:
        mistral_path = paths.run_results / "generation_mistral_heldout_test.jsonl"
        if not mistral_path.is_file():
            raise FileNotFoundError("include_mistral was requested but Mistral generation is missing")
        mistral_secondary, mistral_decisions = analyze_generator(
            "mistral",
            generation_path=mistral_path,
            retrieval_path=retrieval_path,
            gate_model=gate_model,
            gate_sha256=str(gate_manifest["model_sha256"]),
        )
        mistral_secondary_path = paths.run_results / "secondary_mistral_analysis.json"
        mistral_decisions_path = paths.run_artifacts / "analysis" / "mistral_gate_decisions.json"
        write_immutable_json(mistral_secondary_path, mistral_secondary)
        write_immutable_json(mistral_decisions_path, mistral_decisions)
        outputs["mistral_secondary"] = str(mistral_secondary_path)
        outputs["mistral_gate_decisions"] = str(mistral_decisions_path)

    manifest = {
        "schema_version": 1,
        "status": "complete",
        "include_mistral": include_mistral,
        "outputs": {name: sha256_file(path) for name, path in outputs.items()},
        "gate_sha256": gate_manifest["model_sha256"],
        "config_sha256": config_sha256,
        "baseline_tree_sha256": BASELINE_TREE_SHA256,
        "protocol_commit": PROTOCOL_COMMIT,
    }
    manifest_name = "analysis_with_mistral.json" if include_mistral else "analysis_qwen.json"
    write_immutable_json(paths.run_artifacts / "manifests" / manifest_name, manifest)
    return outputs
