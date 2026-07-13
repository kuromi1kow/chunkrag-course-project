"""Outcome-gated confirmatory analysis assembly (Specification Sections 20, 29--32)."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .canonical import atomic_write_json, canonical_json_hash, file_sha256, read_json, read_jsonl
from .constants import EXPERIMENT_ORDER, PROTOCOL_ID, PROTOCOL_SHA256
from .statistics import (
    cliffs_delta, cluster_bootstrap, cluster_bootstrap_difference, cluster_sign_flip,
    cr1_dataset_interaction, holm_adjust, rank_biserial, tost,
)
from .experiments import condition_ids_e2
from .completion import completed_stages


class AnalysisGateError(RuntimeError):
    pass


def require_completed_experiments(
    completion_manifest: Mapping[str, Any], *, artifact_root: Path, completion_path: Path,
) -> None:
    expected_path = artifact_root / "audit" / "completion.json"
    if completion_path.resolve() != expected_path.resolve():
        raise AnalysisGateError("Confirmatory analysis requires the canonical completion manifest path")
    if completion_manifest.get("protocol_sha256") != PROTOCOL_SHA256:
        raise AnalysisGateError("Completion manifest protocol mismatch")
    completed = completion_manifest.get("completed_experiments", [])
    if list(completed) != list(EXPERIMENT_ORDER):
        raise AnalysisGateError("Confirmatory analysis requires completed E0--E7 in frozen order")
    if not completion_manifest.get("artifacts_locked_read_only", False):
        raise AnalysisGateError("Confirmatory analysis requires read-only result artifacts")
    for stage in completion_manifest.get("stage_markers", []):
        path = artifact_root / stage["path"]
        if not path.is_file() or file_sha256(path) != stage["sha256"]:
            raise AnalysisGateError(f"Invalid stage marker: {stage.get('path')}")
        if path.stat().st_mode & 0o222:
            raise AnalysisGateError(f"Stage marker is still writable: {stage.get('path')}")
    provenance = {
        "git_commit": completion_manifest.get("git_commit"),
        "config_sha256": completion_manifest.get("config_sha256"),
        "environment_hash": completion_manifest.get("environment_hash"),
    }
    if not all(isinstance(value, str) and value for value in provenance.values()):
        raise AnalysisGateError("Completion manifest lacks canonical provenance")
    try:
        observed_stages = completed_stages(artifact_root, **provenance)
    except (ValueError, KeyError) as error:
        raise AnalysisGateError("Completion work/stage validation failed") from error
    if observed_stages != list(EXPERIMENT_ORDER):
        raise AnalysisGateError("Completion manifest does not correspond to validated E0--E7 work markers")
    artifacts = completion_manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise AnalysisGateError("Completion manifest contains no artifact inventory")
    for reference in artifacts:
        path = artifact_root / reference["path"]
        if not path.is_file() or path.stat().st_size != int(reference["bytes"]):
            raise AnalysisGateError(f"Missing or resized locked artifact: {reference['path']}")
        if file_sha256(path) != reference["sha256"]:
            raise AnalysisGateError(f"Locked artifact hash mismatch: {reference['path']}")
        if path.stat().st_mode & 0o222:
            raise AnalysisGateError(f"Result artifact is still writable: {reference['path']}")
    lock_path = artifact_root / "audit" / "analysis.lock.json"
    if completion_path.stat().st_mode & 0o222:
        raise AnalysisGateError("Completion manifest is still writable")
    if lock_path.exists():
        raise AnalysisGateError("Confirmatory analysis has already been executed")


def paired_contrast(
    left: Mapping[str, float], right: Mapping[str, float], question_order: Sequence[str],
) -> list[float]:
    if set(left) != set(question_order) or set(right) != set(question_order):
        raise ValueError("Paired contrast question sets do not match the frozen order")
    return [float(left[item]) - float(right[item]) for item in question_order]


def analyze_contrast(
    *, test_id: str, contrasts: Sequence[float], clusters: Sequence[str], equivalence_margin: float | None = None,
) -> dict[str, Any]:
    low, high = cluster_bootstrap(contrasts, clusters, test_id)
    cluster_values: dict[str, list[float]] = defaultdict(list)
    for value, cluster in zip(contrasts, clusters):
        cluster_values[str(cluster)].append(float(value))
    cluster_means = [sum(values) / len(values) for _, values in sorted(cluster_values.items())]
    result = {
        "test_id": test_id, "n": len(contrasts), "clusters": len(set(clusters)),
        "mean_difference": sum(contrasts) / len(contrasts), "ci95_low": low, "ci95_high": high,
        "rank_biserial": rank_biserial(contrasts),
        "raw_p": cluster_sign_flip(contrasts, clusters, test_id),
        "cluster_symmetry_diagnostic": {"cluster_means": cluster_means, "positive": sum(value > 0 for value in cluster_means), "negative": sum(value < 0 for value in cluster_means), "zero": sum(value == 0 for value in cluster_means)},
    }
    if equivalence_margin is not None:
        result["tost"] = tost(contrasts, clusters, equivalence_margin)
        result["ci90"] = cluster_bootstrap(contrasts, clusters, test_id, interval=0.90)
    return result


def adjust_family(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    adjusted = holm_adjust({row["test_id"]: row["raw_p"] for row in results})
    return [{**row, "holm_p": adjusted[row["test_id"]]} for row in results]


def write_analysis(path: Path, payload: Mapping[str, Any]) -> str:
    body = {"schema_version": PROTOCOL_ID, **dict(payload)}
    return atomic_write_json(path, body)


def _condition_scores(artifact_root: Path, dataset: str, condition: str) -> dict[str, dict[str, float]]:
    generation_root = artifact_root / "generation" / "mistral" / dataset / condition
    evaluation_root = artifact_root / "evaluation" / "automatic" / "mistral" / dataset / condition
    generation_by_id: dict[str, str] = {}
    for path in sorted(generation_root.glob("part-*.jsonl")):
        for row in read_jsonl(path):
            generation_by_id[row["generation_id"]] = row["question_id"]
    values: dict[str, dict[str, float]] = {}
    for path in sorted(evaluation_root.glob("part-*.jsonl")):
        for row in read_jsonl(path):
            question_id = generation_by_id[row["generation_id"]]
            values[question_id] = {key: float(value) for key, value in row["metrics"].items() if isinstance(value, (int, float))}
    return values


def analyze_primary_families(artifact_root: Path) -> dict[str, Any]:
    primary: list[dict[str, Any]] = []
    budget: list[dict[str, Any]] = []
    mechanism: list[dict[str, Any]] = []
    exposure_rows: list[dict[str, Any]] = []
    boundary_vectors: dict[tuple[str, str], tuple[list[float], list[str]]] = {}
    for dataset in ("squad_v2", "hotpot_qa"):
        questions = read_jsonl(artifact_root / "manifests" / "questions" / f"{dataset}.jsonl")
        order = [row["question_id"] for row in questions]
        clusters = [row["cluster_id"] for row in questions]
        cache: dict[str, dict[str, dict[str, float]]] = {}

        def scores(condition: str) -> dict[str, dict[str, float]]:
            cache.setdefault(condition, _condition_scores(artifact_root, dataset, condition))
            return cache[condition]

        fixed_op1024 = scores("fixed192__operational-1024")
        fixed_matched1024 = scores("fixed192__matched-1024")
        fixed_op4096 = scores("fixed192__operational-4096")
        for policy in ("recursive192", "sentence192", "semantic192"):
            exact = scores(f"{policy}__matched-4096")
            jitters = [scores(f"{policy}-jitter-{seed}__matched-4096") for seed in (1103, 2207, 3301, 4409, 5519)]
            h1 = [exact[q]["f1"] * 100 - sum(item[q]["f1"] * 100 for item in jitters) / 5 for q in order]
            boundary_vectors[(dataset, policy)] = (h1, clusters)
            primary.append(analyze_contrast(test_id=f"H1:{dataset}:{policy}", contrasts=h1, clusters=clusters, equivalence_margin=2.0))
            op1024 = scores(f"{policy}__operational-1024")
            matched1024 = scores(f"{policy}__matched-1024")
            op4096 = scores(f"{policy}__operational-4096")
            h2 = [((op1024[q]["f1"] - fixed_op1024[q]["f1"]) - (matched1024[q]["f1"] - fixed_matched1024[q]["f1"])) * 100 for q in order]
            primary.append(analyze_contrast(test_id=f"H2:{dataset}:{policy}", contrasts=h2, clusters=clusters))
            h3 = [((op1024[q]["f1"] - fixed_op1024[q]["f1"]) - (op4096[q]["f1"] - fixed_op4096[q]["f1"])) * 100 for q in order]
            budget.append(analyze_contrast(test_id=f"H3:{dataset}:{policy}", contrasts=h3, clusters=clusters))
            evidence_h2 = [
                (op1024[q]["consumed_gold_evidence_fraction"] - fixed_op1024[q]["consumed_gold_evidence_fraction"])
                - (matched1024[q]["consumed_gold_evidence_fraction"] - fixed_matched1024[q]["consumed_gold_evidence_fraction"])
                for q in order
            ]
            evidence_h3 = [
                (op1024[q]["consumed_gold_evidence_fraction"] - fixed_op1024[q]["consumed_gold_evidence_fraction"])
                - (op4096[q]["consumed_gold_evidence_fraction"] - fixed_op4096[q]["consumed_gold_evidence_fraction"])
                for q in order
            ]
            mechanism.extend([
                {"dataset": dataset, "policy": policy, "contrast": "exposure", "mean": float(sum(evidence_h2) / len(evidence_h2))},
                {"dataset": dataset, "policy": policy, "contrast": "budget", "mean": float(sum(evidence_h3) / len(evidence_h3))},
            ])
            for label, policy_scores, fixed_scores in (
                ("operational-1024", op1024, fixed_op1024),
                ("matched-1024", matched1024, fixed_matched1024),
                ("operational-4096", op4096, fixed_op4096),
            ):
                answer_values = [(policy_scores[q]["f1"] - fixed_scores[q]["f1"]) * 100 for q in order]
                evidence_values = [policy_scores[q]["consumed_gold_evidence_fraction"] - fixed_scores[q]["consumed_gold_evidence_fraction"] for q in order]
                answer_low, answer_high = cluster_bootstrap(answer_values, clusters, f"figure3-answer:{dataset}:{policy}:{label}")
                evidence_low, evidence_high = cluster_bootstrap(evidence_values, clusters, f"figure3-evidence:{dataset}:{policy}:{label}")
                exposure_rows.append({
                    "dataset": dataset, "policy": policy, "condition": label,
                    "answer_mean": float(sum(answer_values) / len(answer_values)),
                    "answer_ci_low": answer_low, "answer_ci_high": answer_high,
                    "evidence_mean": float(sum(evidence_values) / len(evidence_values)),
                    "evidence_ci_low": evidence_low, "evidence_ci_high": evidence_high,
                })
    heterogeneity: list[dict[str, Any]] = []
    for policy in ("recursive192", "sentence192", "semantic192"):
        left, left_clusters = boundary_vectors[("squad_v2", policy)]
        right, right_clusters = boundary_vectors[("hotpot_qa", policy)]
        values = [*left, *right]
        indicators = [True] * len(left) + [False] * len(right)
        prefixed_clusters = [f"squad_v2:{item}" for item in left_clusters] + [f"hotpot_qa:{item}" for item in right_clusters]
        estimate, standard_error, degrees = cr1_dataset_interaction(values, indicators, prefixed_clusters)
        from scipy.stats import t as student_t

        raw_p = float(2 * student_t.sf(abs(estimate / standard_error), degrees)) if standard_error else (1.0 if estimate == 0 else 0.0)
        low, high = cluster_bootstrap_difference(left, left_clusters, right, right_clusters, f"H4:squad_v2-minus-hotpot_qa:{policy}")
        heterogeneity.append({
            "test_id": f"H4:squad_v2-minus-hotpot_qa:{policy}", "mean_difference": estimate,
            "standard_error": standard_error, "degrees_freedom": degrees,
            "ci95_low": low, "ci95_high": high, "cliffs_delta": cliffs_delta(left, right),
            "raw_p": raw_p,
        })
    adjusted_h4 = holm_adjust({row["test_id"]: row["raw_p"] for row in heterogeneity})
    heterogeneity = [{**row, "holm_p": adjusted_h4[row["test_id"]]} for row in heterogeneity]
    adjusted_primary = adjust_family(primary)
    tost_adjusted = holm_adjust({row["test_id"]: row["tost"]["p_tost"] for row in adjusted_primary if row["test_id"].startswith("H1:")})
    for row in adjusted_primary:
        if row["test_id"] in tost_adjusted:
            row["tost"]["holm_p"] = tost_adjusted[row["test_id"]]
    return {"primary": adjusted_primary, "budget": adjust_family(budget), "heterogeneity": heterogeneity, "mechanism": mechanism, "exposure_rows": exposure_rows}


def dataset_summary(artifact_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in ("squad_v2", "hotpot_qa", "techqa"):
        questions_path = artifact_root / "manifests" / "questions" / f"{dataset}.jsonl"
        corpus_path = artifact_root / "manifests" / "corpora" / f"{dataset}.jsonl"
        clusters_path = artifact_root / "manifests" / "clusters" / f"{dataset}.jsonl"
        rows.append({
            "dataset": dataset, "questions": len(read_jsonl(questions_path)),
            "documents": len(read_jsonl(corpus_path)), "clusters": len(read_jsonl(clusters_path)),
            "question_hash": file_sha256(questions_path), "corpus_hash": file_sha256(corpus_path),
        })
    return rows


def _techqa_scores(artifact_root: Path, condition: str) -> dict[str, dict[str, float]]:
    generation_root = artifact_root / "generation" / "mistral" / "techqa" / condition
    generation_by_id: dict[str, str] = {}
    for path in sorted(generation_root.glob("part-*.jsonl")):
        for row in read_jsonl(path):
            generation_by_id[row["generation_id"]] = row["question_id"]
    answer: dict[str, float] = {}
    judge_root = artifact_root / "evaluation" / "judge" / "techqa" / condition
    for path in sorted(judge_root.glob("part-*.jsonl")):
        for row in read_jsonl(path):
            answer[generation_by_id[row["generation_id"]]] = float(row["judge"]["parsed"]["semantic_utility"])
    evidence = _condition_scores(artifact_root, "techqa", condition)
    return {question_id: {"utility": utility, "consumed_gold_evidence_fraction": evidence[question_id]["consumed_gold_evidence_fraction"]} for question_id, utility in answer.items()}


def analyze_techqa_family(artifact_root: Path) -> dict[str, Any]:
    questions = read_jsonl(artifact_root / "manifests" / "questions" / "techqa.jsonl")
    order = [row["question_id"] for row in questions]
    clusters = [row["cluster_id"] for row in questions]
    validation_path = artifact_root / "evaluation" / "human" / "judge-validation.json"
    validation_payload = read_json(validation_path) if validation_path.is_file() else {}
    validated = bool(validation_payload.get("confirmatory"))
    remove_from_main = bool(validation_payload.get("remove_from_main"))
    cache: dict[str, dict[str, dict[str, float]]] = {}

    def scores(condition: str) -> dict[str, dict[str, float]]:
        cache.setdefault(condition, _techqa_scores(artifact_root, condition))
        return cache[condition]

    fixed_op = scores("fixed192__operational-1024")
    fixed_matched = scores("fixed192__matched-1024")
    results: list[dict[str, Any]] = []
    mechanism: list[dict[str, Any]] = []
    exposure_rows: list[dict[str, Any]] = []
    for policy in ("recursive192", "sentence192", "semantic192"):
        exact = scores(f"{policy}__matched-4096")
        jitters = [scores(f"{policy}-jitter-{seed}__matched-4096") for seed in (1103, 2207, 3301, 4409, 5519)]
        h1 = [exact[q]["utility"] - sum(item[q]["utility"] for item in jitters) / 5 for q in order]
        results.append(analyze_contrast(test_id=f"H1:techqa:{policy}", contrasts=h1, clusters=clusters, equivalence_margin=0.05))
        op = scores(f"{policy}__operational-1024")
        matched = scores(f"{policy}__matched-1024")
        h2 = [(op[q]["utility"] - fixed_op[q]["utility"]) - (matched[q]["utility"] - fixed_matched[q]["utility"]) for q in order]
        results.append(analyze_contrast(test_id=f"H2:techqa:{policy}", contrasts=h2, clusters=clusters))
        evidence = [(op[q]["consumed_gold_evidence_fraction"] - fixed_op[q]["consumed_gold_evidence_fraction"]) - (matched[q]["consumed_gold_evidence_fraction"] - fixed_matched[q]["consumed_gold_evidence_fraction"]) for q in order]
        mechanism.append({"dataset": "techqa", "policy": policy, "contrast": "exposure", "mean": float(sum(evidence) / len(evidence))})
        op4096 = scores(f"{policy}__operational-4096")
        fixed_op4096 = scores("fixed192__operational-4096")
        for label, policy_scores, fixed_scores in (
            ("operational-1024", op, fixed_op),
            ("matched-1024", matched, fixed_matched),
            ("operational-4096", op4096, fixed_op4096),
        ):
            answer_values = [policy_scores[q]["utility"] - fixed_scores[q]["utility"] for q in order]
            evidence_values = [policy_scores[q]["consumed_gold_evidence_fraction"] - fixed_scores[q]["consumed_gold_evidence_fraction"] for q in order]
            answer_low, answer_high = cluster_bootstrap(answer_values, clusters, f"figure3-answer:techqa:{policy}:{label}")
            evidence_low, evidence_high = cluster_bootstrap(evidence_values, clusters, f"figure3-evidence:techqa:{policy}:{label}")
            exposure_rows.append({"dataset": "techqa", "policy": policy, "condition": label, "answer_mean": float(sum(answer_values) / len(answer_values)), "answer_ci_low": answer_low, "answer_ci_high": answer_high, "evidence_mean": float(sum(evidence_values) / len(evidence_values)), "evidence_ci_low": evidence_low, "evidence_ci_high": evidence_high})
    adjusted = adjust_family(results) if validated else [{**row, "holm_p": None} for row in results]
    gold_gaps = []
    for budget in (1024, 4096):
        gold = scores(f"gold-{budget}")
        conditions = [condition for condition in condition_ids_e2() if condition.endswith(str(budget))]
        for condition in conditions:
            system = scores(condition)
            values = [gold[q]["utility"] - system[q]["utility"] for q in order]
            gold_gaps.append({"condition_id": condition, "budget": budget, "mean_gold_gap_semantic_utility": float(sum(values) / len(values)), "n": len(values)})
    return {"validated": validated, "remove_from_main": remove_from_main, "results": adjusted, "mechanism": mechanism, "exposure_rows": exposure_rows, "gold_semantic_gaps": gold_gaps, "human_subset_results": validation_payload.get("human_subset_results", [])}


def gold_techqa_summary(artifact_root: Path) -> dict[str, Any]:
    gold_rows: list[dict[str, Any]] = []
    for dataset in ("squad_v2", "hotpot_qa", "techqa"):
        for condition in ("gold-1024", "gold-4096"):
            scores = _condition_scores(artifact_root, dataset, condition)
            values = [row["f1"] for row in scores.values()]
            gold_rows.append({
                "dataset": dataset, "condition": condition, "metric": "f1",
                "value": float(sum(values) / len(values)),
            })
    judge_values: list[float] = []
    groundedness: list[float] = []
    judge_root = artifact_root / "evaluation" / "judge" / "techqa"
    for path in sorted(judge_root.glob("**/part-*.jsonl")):
        for row in read_jsonl(path):
            parsed = row.get("judge", {}).get("parsed", {})
            if "semantic_utility" in parsed:
                judge_values.append(float(parsed["semantic_utility"]))
                groundedness.append(float(parsed["groundedness"]) / 2)
    validation_path = artifact_root / "evaluation" / "human" / "judge-validation.json"
    validation = read_json(validation_path) if validation_path.is_file() else {"confirmatory": False, "status": "missing"}
    return {
        "gold": gold_rows,
        "techqa_semantic_utility": float(sum(judge_values) / len(judge_values)) if judge_values else None,
        "techqa_groundedness": float(sum(groundedness) / len(groundedness)) if groundedness else None,
        "judge_validation": validation,
    }


def regenerate_analysis(artifact_root: Path, completion_manifest: Path, output_path: Path) -> str:
    completion = read_json(completion_manifest)
    from .environment import require_clean_git
    from .protocol import repo_root
    from .canonical import source_sha256
    repository = repo_root()
    state = require_clean_git(repository)
    if state["commit"] != completion.get("git_commit"):
        raise AnalysisGateError("Confirmatory analysis Git commit differs from the canonical run")
    if source_sha256(repository, repository / "requirements-main-study.transitive.json") != completion.get("source_hash"):
        raise AnalysisGateError("Confirmatory analysis source hash differs from the canonical run")
    canonical_output = artifact_root / "analysis" / "confirmatory.json"
    if output_path.resolve() != canonical_output.resolve():
        raise AnalysisGateError("Confirmatory analysis output path is immutable")
    require_completed_experiments(completion, artifact_root=artifact_root, completion_path=completion_manifest)
    payload = analyze_primary_families(artifact_root)
    payload["dataset_summary"] = dataset_summary(artifact_root)
    techqa = analyze_techqa_family(artifact_root)
    payload["techqa"] = techqa
    payload["mechanism"].extend(techqa["mechanism"])
    payload["exposure_rows"].extend(techqa["exposure_rows"])
    payload["gold_techqa"] = gold_techqa_summary(artifact_root)
    payload["completion_manifest_hash"] = canonical_json_hash(completion)
    digest = write_analysis(output_path, payload)
    lock = {
        "schema_version": PROTOCOL_ID, "protocol_sha256": PROTOCOL_SHA256,
        "completion_manifest_hash": canonical_json_hash(completion),
        "analysis_path": output_path.relative_to(artifact_root).as_posix(), "analysis_sha256": digest,
    }
    atomic_write_json(artifact_root / "audit" / "analysis.lock.json", lock)
    output_path.chmod(output_path.stat().st_mode & ~0o222)
    analysis_lock = artifact_root / "audit" / "analysis.lock.json"
    analysis_lock.chmod(analysis_lock.stat().st_mode & ~0o222)
    return digest
