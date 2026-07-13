"""Frozen derived outputs required by Experiments E1, E3, E4, E5, and E6."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np

from .canonical import atomic_write_json, atomic_write_jsonl, read_jsonl
from .constants import PROTOCOL_ID
from .evaluation import document_metrics, document_ranking, interval_fully_covered
from .statistics import cluster_bootstrap
from .environment import hardware_manifest


def _scores_by_question(root: Path, role: str, dataset: str, condition: str, metric: str) -> dict[str, float]:
    generation_by_id: dict[str, str] = {}
    for path in sorted((root / "generation" / role / dataset / condition).glob("part-*.jsonl")):
        for row in read_jsonl(path): generation_by_id[row["generation_id"]] = row["question_id"]
    result: dict[str, float] = {}
    for path in sorted((root / "evaluation" / "automatic" / role / dataset / condition).glob("part-*.jsonl")):
        for row in read_jsonl(path): result[generation_by_id[row["generation_id"]]] = float(row["metrics"][metric])
    return result


def maybe_write_gold_gaps(root: Path, dataset: str, expected_question_ids: Sequence[str]) -> str | None:
    from .experiments import condition_ids_e2
    order = sorted(expected_question_ids)
    gold = {budget: _scores_by_question(root, "mistral", dataset, f"gold-{budget}", "f1") for budget in (1024, 4096)}
    if any(set(rows) != set(order) for rows in gold.values()):
        return None
    gaps = []
    for condition in condition_ids_e2():
        budget = int(condition.rsplit("-", 1)[1])
        system = _scores_by_question(root, "mistral", dataset, condition, "f1")
        if set(system) != set(order):
            return None
        values = [(gold[budget][qid] - system[qid]) * 100 for qid in order]
        gaps.append({"condition_id": condition, "budget": budget, "mean_gold_gap_f1_points": mean(values), "n": len(values)})
    return atomic_write_json(root / "analysis" / "e3" / f"{dataset}-gold-gaps.json", {"schema_version": PROTOCOL_ID, "dataset": dataset, "gaps": gaps})


def retrieval_metric_rows(
    dataset: str, questions: Sequence[Mapping[str, Any]], chunks: Mapping[str, Mapping[str, Any]],
    traces: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    question_by_id = {str(row["question_id"]): row for row in questions}
    rows: list[dict[str, Any]] = []
    for trace in traces:
        question = question_by_id[str(trace["question_id"])]
        ranked_chunks = list(trace["top16_chunk_ids"])
        documents = document_ranking(ranked_chunks, chunks)
        row: dict[str, Any] = {
            "schema_version": PROTOCOL_ID, "retrieval_id": trace["retrieval_id"],
            "question_id": trace["question_id"], "dataset": dataset,
            "condition_id": trace["condition_id"], "mrr": document_metrics(documents, question["gold_document_ids"], 16)["mrr"],
            "ndcg_at_8": document_metrics(documents, question["gold_document_ids"], 8)["ndcg"],
        }
        for depth in (4, 8):
            metrics = document_metrics(documents, question["gold_document_ids"], depth)
            row[f"doccov_at_{depth}"] = metrics["doc_coverage"]
            row[f"allhit_at_{depth}"] = metrics["all_hit"]
            intervals: dict[str, list[tuple[int, int]]] = defaultdict(list)
            for chunk_id in ranked_chunks[:depth]:
                chunk = chunks[chunk_id]
                intervals[str(chunk["document_id"])].append((int(chunk["char_start"]), int(chunk["char_end"])))
            if dataset == "squad_v2":
                row[f"answer_span_at_{depth}"] = float(any(
                    interval_fully_covered(int(span["char_start"]), int(span["char_end"]), intervals.get(str(span["document_id"]), []))
                    for span in question["gold_spans"]
                ))
                row[f"gold_document_hit_at_{depth}"] = row[f"allhit_at_{depth}"]
            elif dataset == "hotpot_qa":
                facts = question["supporting_facts"]
                covered = sum(interval_fully_covered(
                    int(fact["char_start"]), int(fact["char_end"]), intervals.get(str(fact["document_id"]), []),
                ) for fact in facts)
                row[f"supporting_sentence_fraction_at_{depth}"] = covered / len(facts) if facts else 0.0
                row[f"all_supporting_sentences_at_{depth}"] = float(covered == len(facts))
        rows.append(row)
    return rows


def cost_trace(
    *, dataset: str, condition_id: str, chunks: Sequence[Mapping[str, Any]],
    traces: Sequence[Mapping[str, Any]], build_audit: Mapping[str, Any], warmup_questions: int,
) -> dict[str, Any]:
    lengths = sorted(int(row["token_count"]) for row in chunks)
    return {
        "schema_version": PROTOCOL_ID, "dataset": dataset, "condition_id": condition_id,
        "chunk_count": len(lengths), "chunk_tokens_mean": mean(lengths),
        "chunk_tokens_median": median(lengths), "chunk_tokens_p10": float(np.quantile(lengths, 0.10)),
        "chunk_tokens_p90": float(np.quantile(lengths, 0.90)), "chunk_tokens_max": max(lengths),
        "index_build_seconds": float(build_audit["index_build_seconds"]),
        "embedding_tokens": int(build_audit["embedding_tokens"]), "index_bytes": int(build_audit["index_bytes"]),
        "index_vectors": int(build_audit["index_vectors"]), "embedding_dtype": build_audit["embedding_dtype"],
        "warmup_questions": warmup_questions, "measured_questions": len(traces),
        "hardware": hardware_manifest(), "batch_size": {"embedding": 64, "reranker": 32},
        "retrieval_seconds": sum(float(row["latency"].get("dense_seconds", 0)) + float(row["latency"].get("sparse_seconds", 0)) for row in traces),
        "reranker_seconds": sum(float(row["latency"].get("reranker_seconds", 0)) for row in traces),
    }


def write_retrieval_outputs(
    root: Path, *, namespace: str, dataset: str, condition_id: str,
    questions: Sequence[Mapping[str, Any]], chunks: Sequence[Mapping[str, Any]],
    traces: Sequence[Mapping[str, Any]], build_audit: Mapping[str, Any], warmup_questions: int = 5,
) -> list[str]:
    chunk_by_id = {str(row["chunk_id"]): row for row in chunks}
    metric_rows = retrieval_metric_rows(dataset, questions, chunk_by_id, traces)
    metric_hash = atomic_write_jsonl(
        root / "analysis" / "retrieval" / namespace / dataset / f"{condition_id}.jsonl",
        metric_rows, "retrieval_id",
    )
    cost_hash = atomic_write_json(
        root / "audit" / "cost" / namespace / dataset / f"{condition_id}.json",
        cost_trace(dataset=dataset, condition_id=condition_id, chunks=chunks, traces=traces, build_audit=build_audit, warmup_questions=warmup_questions),
    )
    exposure_hash = atomic_write_json(
        root / "audit" / "encoder-exposure" / namespace / dataset / f"{condition_id}.json",
        {"schema_version": PROTOCOL_ID, "dataset": dataset, "condition_id": condition_id,
         "native_token_counts": list(build_audit["dense_token_counts"]), "truncated": 0,
         "embedding_dtype": build_audit["embedding_dtype"]},
    )
    return [metric_hash, cost_hash, exposure_hash]


def maybe_write_e5_effect_table(root: Path, dataset: str, embedder: str, stack: str) -> str | None:
    paths = [root / "analysis" / "retrieval" / "secondary" / dataset / f"{embedder}__{stack}__{policy}.jsonl" for policy in ("fixed192", "recursive192", "sentence192", "semantic192")]
    if not all(path.is_file() for path in paths):
        return None
    by_policy = {policy: {row["question_id"]: row for row in read_jsonl(path)} for policy, path in zip(("fixed192", "recursive192", "sentence192", "semantic192"), paths)}
    order = sorted(by_policy["fixed192"])
    rows = []
    for policy in ("recursive192", "sentence192", "semantic192"):
        for metric in ("mrr", "ndcg_at_8", "doccov_at_4", "doccov_at_8", "allhit_at_4", "allhit_at_8"):
            values = [float(by_policy[policy][qid][metric]) - float(by_policy["fixed192"][qid][metric]) for qid in order]
            rows.append({"policy": policy, "metric": metric, "mean_paired_effect": mean(values), "n": len(values)})
    return atomic_write_json(root / "analysis" / "retrieval" / "secondary" / dataset / f"{embedder}__{stack}__paired-effects.json", {"schema_version": PROTOCOL_ID, "dataset": dataset, "embedder": embedder, "retriever": stack, "effects": rows})


def qwen_effect_summary(root: Path, dataset: str, clusters: Mapping[str, str]) -> dict[str, Any] | None:
    base = root / "evaluation" / "automatic" / "qwen" / dataset
    conditions = [f"{policy}__matched-4096" for policy in ("fixed192", "recursive192", "sentence192", "semantic192")] + ["gold-4096"]
    if not all((base / condition).is_dir() for condition in conditions):
        return None
    values: dict[str, dict[str, float]] = {}
    for condition in conditions:
        values[condition] = {}
        for path in sorted((base / condition).glob("part-*.jsonl")):
            for row in read_jsonl(path):
                values[condition][row["generation_id"]] = float(row["metrics"]["f1"])
    generation_to_question = {}
    for condition in conditions:
        for path in sorted((root / "generation" / "qwen" / dataset / condition).glob("part-*.jsonl")):
            for row in read_jsonl(path): generation_to_question[row["generation_id"]] = row["question_id"]
    by_condition = {condition: {generation_to_question[gid]: score for gid, score in scores.items()} for condition, scores in values.items()}
    if any(set(rows) != set(clusters) for rows in by_condition.values()):
        return None
    order = sorted(by_condition["fixed192__matched-4096"])
    rows = []
    for policy in ("recursive192", "sentence192", "semantic192"):
        contrasts = [(by_condition[f"{policy}__matched-4096"][qid] - by_condition["fixed192__matched-4096"][qid]) * 100 for qid in order]
        cluster_ids = [clusters[qid] for qid in order]
        low, high = cluster_bootstrap(contrasts, cluster_ids, f"E6:{dataset}:{policy}")
        rows.append({"policy": policy, "mean_difference": mean(contrasts), "ci95_low": low, "ci95_high": high, "n": len(order), "clusters": len(set(cluster_ids))})
    return {"schema_version": PROTOCOL_ID, "dataset": dataset, "effects": rows}
