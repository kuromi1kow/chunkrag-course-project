"""Concrete E1--E7 handlers (Specification Sections 11--30 and E1--E7)."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .artifacts import ArtifactStore
from .canonical import canonical_json_hash, file_sha256, read_jsonl, tree_sha256
from .checkpoint import ShardCheckpoint
from .chunking import (
    TokenizedSource, chunk_records, fixed_cuts, recursive_cuts, semantic_cuts,
    sentence_cuts,
)
from .constants import JITTER_SEEDS, POLICY_ORDER, PROTOCOL_ID, PROTOCOL_SHA256, STRUCTURED_POLICIES
from .controls import jitter_cuts, validate_changed_fraction
from .evaluation import (
    best_answer_metrics, build_evaluation_record, parse_judge_json, techqa_judge_messages,
)
from .experiments import WorkItem, condition_ids_e2
from .generation import LocalGenerator, build_generation_record
from .human import agreement_report, blindness_scan, build_blinded_package, build_training_package, validate_label_rows
from .packing import matched_pack, matched_target, operational_pack
from .prompts import render_passages
from .protocol import ProtocolError, repo_root
from .retrieval import PrimaryRetriever, build_retrieval_record


_TOKENIZERS: dict[str, Any] = {}
_ENCODERS: dict[str, Any] = {}
_GENERATORS: dict[str, LocalGenerator] = {}


def _root(store: ArtifactStore) -> Path:
    return store.root


def _load_manifest(store: ArtifactStore, kind: str, dataset: str) -> list[dict[str, Any]]:
    path = _root(store) / "manifests" / kind / f"{dataset}.jsonl"
    if not path.is_file():
        raise ProtocolError(f"Missing prerequisite manifest: {path}")
    return read_jsonl(path)


def _load_tokenizer(config: Mapping[str, Any], role: str = "canonical") -> Any:
    spec = config["models"][role]
    key = f"{spec['repository']}@{spec['revision']}"
    if key not in _TOKENIZERS:
        from transformers import AutoTokenizer

        _TOKENIZERS[key] = AutoTokenizer.from_pretrained(spec["repository"], revision=spec["revision"], local_files_only=True)
    return _TOKENIZERS[key]


def _load_encoder(config: Mapping[str, Any], role: str, device: str = "cuda") -> Any:
    spec = config["models"][role]
    key = f"{spec['repository']}@{spec['revision']}@{device}"
    if key not in _ENCODERS:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(spec["repository"], revision=spec["revision"], device=device, local_files_only=True)
        model.eval()
        _ENCODERS[key] = model
    return _ENCODERS[key]


def _condition_base(condition_id: str) -> tuple[str, int | None]:
    if "-jitter-" in condition_id:
        policy, seed = condition_id.rsplit("-jitter-", 1)
        return policy, int(seed)
    return condition_id, None


def _gpu_memory() -> dict[str, int]:
    try:
        import torch

        return {"peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated())} if torch.cuda.is_available() else {"peak_gpu_memory_bytes": 0}
    except ImportError:
        return {"peak_gpu_memory_bytes": 0}


def _cuts_for_policy(source: TokenizedSource, policy: str, config: Mapping[str, Any]) -> list[int]:
    if policy == "fixed192":
        return fixed_cuts(source.tokens)
    if policy == "recursive192":
        return recursive_cuts(source)
    if policy == "sentence192":
        return sentence_cuts(source)
    if policy == "semantic192":
        encoder = _load_encoder(config, "canonical")
        return semantic_cuts(source, lambda texts: encoder.encode(texts, batch_size=64, normalize_embeddings=True))
    raise ProtocolError(f"Unknown frozen policy: {policy}")


def execute_e1(item: WorkItem, config: Mapping[str, Any], store: ArtifactStore) -> list[str]:
    assert item.dataset is not None
    corpus = _load_manifest(store, "corpora", item.dataset)
    questions = _load_manifest(store, "questions", item.dataset)
    tokenizer = _load_tokenizer(config, "canonical")
    policy, control_seed = _condition_base(item.condition_id)
    chunks: list[dict[str, Any]] = []
    jitter_results = []
    for document in corpus:
        source = TokenizedSource.build(document["text"], tokenizer)
        base = _cuts_for_policy(source, policy, config)
        generation_hash = "0" * 64
        cuts = base
        if control_seed is not None:
            jitter = jitter_cuts(
                base, seed=control_seed, policy=policy, document_id=document["document_id"],
                final_short=(base[-1] - base[-2] < 64),
            )
            cuts = list(jitter.cuts)
            generation_hash = jitter.generation_hash
            jitter_results.append(jitter)
        chunks.extend(chunk_records(
            document, source, policy, cuts,
            config["models"]["canonical"]["repository"], config["models"]["canonical"]["revision"],
            condition_id=item.condition_id, control_seed=control_seed,
            boundary_generation_hash=generation_hash,
        ))
    if control_seed is not None:
        validate_changed_fraction(jitter_results, config["chunking"]["required_changed_fraction"])
    chunk_ref = store.write_jsonl(
        f"chunks/{item.dataset}/{item.condition_id}.jsonl", chunks, "chunk", "chunk_id",
    )
    dense = config["models"]["dense"]
    reranker = config["models"]["reranker"]
    engine = PrimaryRetriever(
        chunks, dense["repository"], dense["revision"], reranker["repository"],
        reranker["revision"], config["retrieval"]["query_prefix"], "cuda",
    )
    engine.build()
    question_hash = file_sha256(_root(store) / "manifests" / "questions" / f"{item.dataset}.jsonl")
    corpus_hash = file_sha256(_root(store) / "manifests" / "corpora" / f"{item.dataset}.jsonl")
    retrieval_config_hash = canonical_json_hash({"retrieval": config["retrieval"], "models": {"dense": dense, "reranker": reranker}})
    upstream = canonical_json_hash([question_hash, corpus_hash, chunk_ref.sha256])
    traces: list[dict[str, Any]] = []
    for question in questions:
        dense_rows, sparse_rows, fused, reranked, latency = engine.query(question["question"])
        traces.append(build_retrieval_record(
            question_id=question["question_id"], condition_id=item.condition_id,
            question_manifest_hash=question_hash, corpus_manifest_hash=corpus_hash,
            dense=dense_rows, sparse=sparse_rows, fused=fused, reranked=reranked,
            config_hash=retrieval_config_hash, upstream_hash=upstream,
            latency=latency, memory=_gpu_memory(),
        ))
    trace_ref = store.write_jsonl(
        f"retrieval/primary/{item.dataset}/{item.condition_id}.jsonl",
        traces, "retrieval", "retrieval_id",
    )
    return [chunk_ref.sha256, trace_ref.sha256]


def _questions_for_shard(store: ArtifactStore, dataset: str, shard: int) -> list[dict[str, Any]]:
    questions = sorted(_load_manifest(store, "questions", dataset), key=lambda row: row["question_id"])
    selected = questions[shard * 50:(shard + 1) * 50]
    if not selected:
        raise ProtocolError(f"Empty question shard: {dataset}/{shard}")
    return selected


def _chunks_by_id(store: ArtifactStore, dataset: str, condition: str) -> dict[str, dict[str, Any]]:
    path = _root(store) / "chunks" / dataset / f"{condition}.jsonl"
    return {row["chunk_id"]: row for row in read_jsonl(path)}


def _retrieval_by_question(store: ArtifactStore, dataset: str, condition: str, secondary: str = "primary") -> dict[str, dict[str, Any]]:
    path = _root(store) / "retrieval" / secondary / dataset / f"{condition}.jsonl"
    return {row["question_id"]: row for row in read_jsonl(path)}


def _attach_titles(chunks: Sequence[dict[str, Any]], corpus: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [{**chunk, "title": corpus[chunk["document_id"]]["title"]} for chunk in chunks]


def _retrieved_chunks(store: ArtifactStore, dataset: str, condition: str, question_id: str) -> tuple[list[dict[str, Any]], str]:
    trace = _retrieval_by_question(store, dataset, condition)[question_id]
    chunks = _chunks_by_id(store, dataset, condition)
    corpus = {row["document_id"]: row for row in _load_manifest(store, "corpora", dataset)}
    selected = _attach_titles([chunks[item] for item in trace["top16_chunk_ids"]], corpus)
    return selected, canonical_json_hash(trace)


def _matched_target_for_question(store: ArtifactStore, config: Mapping[str, Any], dataset: str, question: Mapping[str, Any], budget: int, tokenizer: Any, *, qwen: bool = False) -> int:
    contexts: list[str] = []
    conditions = list(POLICY_ORDER) if qwen else ["fixed192", *STRUCTURED_POLICIES, *[f"{policy}-jitter-{seed}" for policy in STRUCTURED_POLICIES for seed in JITTER_SEEDS]]
    for condition in conditions:
        chunks, _ = _retrieved_chunks(store, dataset, condition, question["question_id"])
        rendered, _ = render_passages(chunks[:16])
        contexts.append(rendered)
    return matched_target(tokenizer, dataset, question["question"], contexts, budget)


def _generator(config: Mapping[str, Any], role: str) -> LocalGenerator:
    spec = config["models"][role]
    key = role
    if key not in _GENERATORS:
        value = LocalGenerator(spec["repository"], spec["revision"])
        value.load()
        _GENERATORS[key] = value
    return _GENERATORS[key]


def _snapshot_hash(model: LocalGenerator) -> str:
    from huggingface_hub import snapshot_download

    path = Path(snapshot_download(model.repository, revision=model.revision, local_files_only=True))
    if not path.is_dir():
        raise ProtocolError(f"Pinned local model snapshot is unavailable: {model.repository}@{model.revision}")
    return tree_sha256(path)


def _generate_with_retries(generator: LocalGenerator, prompt_ids: list[int], max_new_tokens: int) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    attempts: list[dict[str, Any]] = []
    raw = ""
    trace: dict[str, Any] = {"generated_tokens": 0, "stopping_reason": "failed", "latency_seconds": 0.0, "peak_gpu_memory_bytes": 0}
    for attempt in range(1, 4):
        try:
            raw, trace = generator.generate(prompt_ids, max_new_tokens)
            attempts.append({"attempt": attempt, "status": "success"})
            break
        except (RuntimeError, OSError) as error:
            attempts.append({"attempt": attempt, "status": "infrastructure_failure", "error_type": type(error).__name__})
    return raw, trace, attempts


def _generate_shard(
    item: WorkItem, config: Mapping[str, Any], store: ArtifactStore, *, role: str,
    source_condition: str, packing_id: str,
) -> list[str]:
    assert item.dataset is not None and item.shard_index is not None
    questions = _questions_for_shard(store, item.dataset, item.shard_index)
    generator = _generator(config, role)
    tokenizer = generator.tokenizer
    snapshot_hash = _snapshot_hash(generator)
    budget = int(packing_id.rsplit("-", 1)[1])
    checkpoint = ShardCheckpoint(
        _root(store) / "generation" / role / item.dataset / item.condition_id,
        item.experiment, item.dataset, item.condition_id, item.shard_index,
        [row["question_id"] for row in questions], config["config_sha256"],
        file_sha256(repo_root() / "requirements-main-study.lock"),
    )
    if checkpoint.final_path.is_file():
        evaluation_ref = _evaluate_generation_shard(store, item.dataset, item.condition_id, checkpoint.final_path)
        return [file_sha256(checkpoint.final_path), evaluation_ref]
    for question in questions:
        chunks, upstream = _retrieved_chunks(store, item.dataset, source_condition, question["question_id"])
        if packing_id.startswith("operational"):
            packed = operational_pack(tokenizer, item.dataset, question["question"], chunks, budget)
        else:
            target = _matched_target_for_question(store, config, item.dataset, question, budget, tokenizer, qwen=role == "qwen")
            packed = matched_pack(tokenizer, item.dataset, question["question"], chunks, budget, target)
        raw, trace, attempts = _generate_with_retries(
            generator, list(packed.prompt_token_ids), config["generation"]["max_new_tokens"][item.dataset],
        )
        record = build_generation_record(
            question=question, condition_id=source_condition,
            control_seed=_condition_base(source_condition)[1], packing_id=packing_id, budget=budget,
            packed=packed, model_repository=generator.repository, model_revision=generator.revision,
            model_snapshot_hash=snapshot_hash, retrieval_or_gold_hash=upstream,
            prompt_version_hash=canonical_json_hash({"dataset": item.dataset, "packing": packing_id}),
            raw_output=raw, generated_tokens=trace["generated_tokens"],
            stopping_reason=trace["stopping_reason"], latency={"generation_seconds": trace["latency_seconds"]},
            attempt_history=attempts, hardware={"peak_gpu_memory_bytes": trace.get("peak_gpu_memory_bytes", 0)},
        )
        checkpoint.append(question["question_id"], record)
    partial_rows = read_jsonl(checkpoint.temp_path)
    failed = sum(row["stopping_reason"] == "failed" for row in partial_rows)
    if failed / len(partial_rows) > 0.01:
        checkpoint.invalidate(f"unresolved infrastructure failures {failed}/{len(partial_rows)} exceed 1%")
        raise ProtocolError("Generation shard invalidated because unresolved failures exceed 1%")
    finalized = checkpoint.finalize(lambda row: str(row["question_id"]))
    evaluation_ref = _evaluate_generation_shard(store, item.dataset, item.condition_id, finalized)
    return [file_sha256(finalized), evaluation_ref]


def execute_e2(item: WorkItem, config: Mapping[str, Any], store: ArtifactStore) -> list[str]:
    source_condition, packing_id = item.condition_id.split("__", 1)
    return _generate_shard(item, config, store, role="mistral", source_condition=source_condition, packing_id=packing_id)


def _gold_chunks(
    store: ArtifactStore, config: Mapping[str, Any], dataset: str, question: Mapping[str, Any],
    generator_tokenizer: Any, target: int,
) -> list[dict[str, Any]]:
    from .evaluation import token_f1

    corpus = {row["document_id"]: row for row in _load_manifest(store, "corpora", dataset)}
    chunks: list[dict[str, Any]] = []
    if dataset == "squad_v2":
        doc = corpus[question["gold_document_ids"][0]]
        chosen = min(question["gold_spans"], key=lambda row: (row["char_start"], -(row["char_end"] - row["char_start"]), row["text"]))
        source = TokenizedSource.build(doc["text"], generator_tokenizer)
        answer_tokens = [index for index, (start, end) in enumerate(source.offsets) if end > chosen["char_start"] and start < chosen["char_end"]]
        left = min(answer_tokens) if answer_tokens else 0
        right = max(answer_tokens) + 1 if answer_tokens else min(1, source.tokens)
        best = (left, right)
        prefer_left = True
        while left > 0 or right < source.tokens:
            candidate_left, candidate_right = left, right
            if prefer_left and left > 0:
                candidate_left -= 1
            elif right < source.tokens:
                candidate_right += 1
            elif left > 0:
                candidate_left -= 1
            char_start, char_end = source.char_span(candidate_left, candidate_right)
            candidate = {"chunk_id": None, "document_id": doc["document_id"], "title": doc["title"], "text": doc["text"][char_start:char_end], "char_start": char_start, "char_end": char_end}
            rendered, _ = render_passages([candidate])
            if len(generator_tokenizer(rendered, add_special_tokens=False, truncation=False)["input_ids"]) > target:
                if not prefer_left:
                    break
            else:
                left, right, best = candidate_left, candidate_right, (candidate_left, candidate_right)
            prefer_left = not prefer_left
        char_start, char_end = source.char_span(*best)
        chunks.append({"chunk_id": None, "document_id": doc["document_id"], "title": doc["title"], "text": doc["text"][char_start:char_end], "char_start": char_start, "char_end": char_end})
    elif dataset == "hotpot_qa":
        facts = sorted(question["supporting_facts"], key=lambda row: (row["document_index"], row["sentence_index"]))
        ordered: list[tuple[str, int, int, int]] = []
        seen: set[tuple[str, int]] = set()
        for fact in facts:
            key = (fact["document_id"], fact["sentence_index"])
            if key not in seen:
                seen.add(key)
                ordered.append((fact["document_id"], fact["sentence_index"], fact["char_start"], fact["char_end"]))
        max_distance = max((len(corpus[fact["document_id"]]["source_provenance"][0]["sentence_spans"]) for fact in facts), default=0)
        for distance in range(1, max_distance + 1):
            for fact in facts:
                spans = corpus[fact["document_id"]]["source_provenance"][0]["sentence_spans"]
                for index in (fact["sentence_index"] - distance, fact["sentence_index"] + distance):
                    if 0 <= index < len(spans) and (fact["document_id"], index) not in seen:
                        span = spans[index]
                        seen.add((fact["document_id"], index))
                        ordered.append((fact["document_id"], index, span["char_start"], span["char_end"]))
        for document_id, _, char_start, char_end in ordered:
            fact = {"document_id": document_id, "char_start": char_start, "char_end": char_end}
            doc = corpus[fact["document_id"]]
            chunks.append({"chunk_id": None, "document_id": doc["document_id"], "title": doc["title"], "text": doc["text"][fact["char_start"]:fact["char_end"]], "char_start": fact["char_start"], "char_end": fact["char_end"]})
    else:
        canonical_tokenizer = _load_tokenizer(config, "canonical")
        candidates: list[tuple[float, str, int, dict[str, Any]]] = []
        for doc_id in question["gold_document_ids"]:
            doc = corpus[doc_id]
            source = TokenizedSource.build(doc["text"], canonical_tokenizer)
            cuts = fixed_cuts(source.tokens)
            for token_start, token_end in zip(cuts, cuts[1:]):
                char_start, char_end = source.char_span(token_start, token_end)
                text = doc["text"][char_start:char_end]
                candidate = {"chunk_id": None, "document_id": doc_id, "title": doc["title"], "text": text, "char_start": char_start, "char_end": char_end}
                score = max(token_f1(text, reference) for reference in question["references"])
                candidates.append((-score, doc["title"], char_start, candidate))
        chunks = [item[3] for item in sorted(candidates, key=lambda item: (item[0], item[1], item[2]))]
    return chunks


def _execute_gold(item: WorkItem, config: Mapping[str, Any], store: ArtifactStore, *, role: str) -> list[str]:
    assert item.dataset is not None and item.shard_index is not None
    # Gold ordering is materialized here; the same matched target as retrieved systems is
    # used. The packing helper preserves the target and records any gold shortfall.
    questions = _questions_for_shard(store, item.dataset, item.shard_index)
    generator = _generator(config, role)
    tokenizer = generator.tokenizer
    budget = int(item.condition_id.rsplit("-", 1)[1])
    checkpoint = ShardCheckpoint(
        _root(store) / "generation" / role / item.dataset / item.condition_id,
        item.experiment, item.dataset, item.condition_id, item.shard_index,
        [row["question_id"] for row in questions], config["config_sha256"],
        file_sha256(repo_root() / "requirements-main-study.lock"),
    )
    if checkpoint.final_path.is_file():
        evaluation_ref = _evaluate_generation_shard(store, item.dataset, item.condition_id, checkpoint.final_path)
        return [file_sha256(checkpoint.final_path), evaluation_ref]
    gold_by_question = {row["question_id"]: row for row in _load_manifest(store, "gold", item.dataset)}
    for question in questions:
        target = _matched_target_for_question(store, config, item.dataset, question, budget, tokenizer, qwen=role == "qwen")
        chunks = _gold_chunks(store, config, item.dataset, question, tokenizer, target)
        rendered, _ = render_passages(chunks)
        available = len(tokenizer(rendered, add_special_tokens=False, truncation=False)["input_ids"])
        effective = min(target, available)
        packed = matched_pack(tokenizer, item.dataset, question["question"], chunks, budget, effective)
        raw, trace, attempts = _generate_with_retries(
            generator, list(packed.prompt_token_ids), config["generation"]["max_new_tokens"][item.dataset],
        )
        upstream = canonical_json_hash(gold_by_question[question["question_id"]])
        record = build_generation_record(
            question=question, condition_id="gold", control_seed=None,
            packing_id=item.condition_id, budget=budget, packed=packed,
            model_repository=generator.repository, model_revision=generator.revision,
            model_snapshot_hash=_snapshot_hash(generator), retrieval_or_gold_hash=upstream,
            prompt_version_hash=canonical_json_hash({"dataset": item.dataset, "gold": True}),
            raw_output=raw, generated_tokens=trace["generated_tokens"],
            stopping_reason=trace["stopping_reason"], latency={"generation_seconds": trace["latency_seconds"]},
            attempt_history=attempts, hardware={"peak_gpu_memory_bytes": trace.get("peak_gpu_memory_bytes", 0)},
        )
        checkpoint.append(question["question_id"], record)
    partial_rows = read_jsonl(checkpoint.temp_path)
    failed = sum(row["stopping_reason"] == "failed" for row in partial_rows)
    if failed / len(partial_rows) > 0.01:
        checkpoint.invalidate(f"unresolved infrastructure failures {failed}/{len(partial_rows)} exceed 1%")
        raise ProtocolError("Gold generation shard invalidated because unresolved failures exceed 1%")
    finalized = checkpoint.finalize(lambda row: str(row["question_id"]))
    evaluation_ref = _evaluate_generation_shard(store, item.dataset, item.condition_id, finalized)
    return [file_sha256(finalized), evaluation_ref]


def _consumed_intervals(generation: Mapping[str, Any]) -> dict[str, list[tuple[int, int]]]:
    consumed_end = len(generation["consumed_context"])
    intervals: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for span in generation["ranked_source_spans"]:
        source_start = span.get("source_char_start")
        source_end = span.get("source_char_end")
        if source_start is None or source_end is None or consumed_end <= span["text_rendered_start"]:
            continue
        retained_chars = min(consumed_end, span["text_rendered_end"]) - span["text_rendered_start"]
        intervals[span["document_id"]].append((int(source_start), min(int(source_end), int(source_start) + retained_chars)))
    return intervals


def _automatic_metrics(generation: Mapping[str, Any], question: Mapping[str, Any]) -> dict[str, Any]:
    from .evaluation import interval_fully_covered

    metrics: dict[str, Any] = best_answer_metrics(generation["normalized_output"], question["references"])
    intervals = _consumed_intervals(generation)
    if question["dataset"] == "squad_v2":
        metrics["consumed_gold_evidence_fraction"] = float(any(interval_fully_covered(span["char_start"], span["char_end"], intervals.get(span["document_id"], [])) for span in question["gold_spans"]))
    elif question["dataset"] == "hotpot_qa":
        facts = question["supporting_facts"]
        covered = sum(interval_fully_covered(fact["char_start"], fact["char_end"], intervals.get(fact["document_id"], [])) for fact in facts)
        metrics["consumed_gold_evidence_fraction"] = covered / len(facts) if facts else 0.0
    else:
        represented = sum(bool(intervals.get(doc_id)) for doc_id in set(question["gold_document_ids"]))
        metrics["consumed_gold_evidence_fraction"] = represented / len(set(question["gold_document_ids"])) if question["gold_document_ids"] else 0.0
    return metrics


def _evaluate_generation_shard(store: ArtifactStore, dataset: str, condition: str, generation_path: Path) -> str:
    questions = {row["question_id"]: row for row in _load_manifest(store, "questions", dataset)}
    evaluator_hash = canonical_json_hash({"version": "automatic-v1", "dataset": dataset})
    evaluations = [
        build_evaluation_record(row, questions[row["question_id"]], _automatic_metrics(row, questions[row["question_id"]]), evaluator_hash)
        for row in read_jsonl(generation_path)
    ]
    role = "qwen" if "/qwen/" in generation_path.as_posix() else "mistral"
    output = _root(store) / "evaluation" / "automatic" / role / dataset / condition / generation_path.name
    from .canonical import atomic_write_jsonl

    return atomic_write_jsonl(output, evaluations, "evaluation_id")


def execute_e3(item: WorkItem, config: Mapping[str, Any], store: ArtifactStore) -> list[str]:
    return _execute_gold(item, config, store, role="mistral")


def _generation_shard_path(store: ArtifactStore, role: str, dataset: str, condition: str, shard: int) -> Path:
    return _root(store) / "generation" / role / dataset / condition / f"part-{shard:03d}.jsonl"


def execute_e4(item: WorkItem, config: Mapping[str, Any], store: ArtifactStore) -> list[str]:
    assert item.dataset == "techqa"
    if item.condition_id == "human-package":
        questions = _load_manifest(store, "questions", "techqa")
        generations: dict[tuple[str, str], Mapping[str, Any]] = {}
        mapping = {
            "fixed192": "fixed192__matched-4096", "recursive192": "recursive192__matched-4096",
            "sentence192": "sentence192__matched-4096", "semantic192": "semantic192__matched-4096",
            "semantic192-jitter-1103": "semantic192-jitter-1103__matched-4096", "gold": "gold-4096",
        }
        for label, condition in mapping.items():
            base = _root(store) / "generation" / "mistral" / "techqa" / condition
            for path in sorted(base.glob("part-*.jsonl")):
                for row in read_jsonl(path):
                    generations[(row["question_id"], label)] = row
        package = build_blinded_package(questions, generations)
        blindness_scan(package)
        path = _root(store) / "evaluation" / "human" / "techqa-package.json"
        from .canonical import atomic_write_json
        digest = atomic_write_json(path, {"schema_version": PROTOCOL_ID, "records": package})
        training = build_training_package(questions, generations)
        training_digest = atomic_write_json(_root(store) / "evaluation" / "human" / "techqa-training.json", {"schema_version": PROTOCOL_ID, "records": training})
        from .canonical import identifier_hash
        linkage: list[dict[str, Any]] = []
        package_ids = {row["annotation_record_id"] for row in package}
        for (question_id, label), generation in generations.items():
            annotation_id = identifier_hash("human", question_id, canonical_json_hash(generation))
            if annotation_id in package_ids:
                linkage.append({"annotation_record_id": annotation_id, "generation_id": generation["generation_id"], "condition": label})
        linkage_digest = atomic_write_json(_root(store) / "evaluation" / "human" / "techqa-linkage-private.json", {"schema_version": PROTOCOL_ID, "records": linkage})
        return [digest, training_digest, linkage_digest]
    if item.condition_id == "human-validation":
        from .canonical import read_json, atomic_write_json
        from .statistics import judge_acceptance

        human_root = _root(store) / "evaluation" / "human"
        package = read_json(human_root / "techqa-package.json")["records"]
        expected_ids = {row["annotation_record_id"] for row in package}
        labels_a = read_jsonl(human_root / "human-labels-a.jsonl")
        labels_b = read_jsonl(human_root / "human-labels-b.jsonl")
        adjudicated = read_jsonl(human_root / "human-adjudicated.jsonl")
        validate_label_rows(labels_a, expected_ids, adjudicated=False)
        validate_label_rows(labels_b, expected_ids, adjudicated=False)
        validate_label_rows(adjudicated, expected_ids, adjudicated=True)
        linkage = {row["annotation_record_id"]: row for row in read_json(human_root / "techqa-linkage-private.json")["records"]}
        judge_by_generation: dict[str, Mapping[str, Any]] = {}
        invalid_by_condition: dict[str, list[bool]] = defaultdict(list)
        judge_root = _root(store) / "evaluation" / "judge" / "techqa"
        for path in sorted(judge_root.glob("**/part-*.jsonl")):
            condition = path.parent.name
            for row in read_jsonl(path):
                judge_by_generation[row["generation_id"]] = row["judge"]
                invalid_by_condition[condition].append(not any(attempt["status"] == "success" for attempt in row["judge"]["attempts"]))
        human_values: dict[str, list[int]] = {key: [] for key in ("correctness", "completeness", "groundedness")}
        judge_values: dict[str, list[int]] = {key: [] for key in human_values}
        for row in adjudicated:
            link = linkage[row["annotation_record_id"]]
            judge_row = judge_by_generation[link["generation_id"]]
            parsed = judge_row["parsed"]
            for dimension in human_values:
                if row.get(dimension) is not None:
                    human_values[dimension].append(int(row[dimension]))
                    judge_values[dimension].append(int(parsed[dimension]))
        validation = judge_acceptance(
            judge_values, human_values,
            invalid_fraction_by_condition={key: sum(values) / len(values) for key, values in invalid_by_condition.items()},
        )
        validation["agreement"] = agreement_report(labels_a, labels_b)
        labels_b_by_id = {row["annotation_record_id"]: row for row in labels_b}
        jointly_unassessable_answers = sum(bool(row.get("cannot_assess_reason")) and bool(labels_b_by_id[row["annotation_record_id"]].get("cannot_assess_reason")) for row in labels_a)
        grounded_ids = {row["annotation_record_id"] for row in package if row["groundedness_subset"]}
        jointly_unassessable_grounded = sum(bool(row.get("cannot_assess_reason")) and bool(labels_b_by_id[row["annotation_record_id"]].get("cannot_assess_reason")) for row in labels_a if row["annotation_record_id"] in grounded_ids)
        validation["cannot_assess"] = {"answer_records": jointly_unassessable_answers, "groundedness_records": jointly_unassessable_grounded}
        validation["remove_from_main"] = jointly_unassessable_answers > 36 or jointly_unassessable_grounded > 6
        if validation["remove_from_main"]:
            validation["confirmatory"] = False
        validation["schema_version"] = PROTOCOL_ID
        digest = atomic_write_json(human_root / "judge-validation.json", validation)
        return [digest]
    assert item.shard_index is not None
    source_condition = item.condition_id.removeprefix("judge__")
    path = _generation_shard_path(store, "mistral", "techqa", source_condition, item.shard_index)
    generations = read_jsonl(path)
    questions = {row["question_id"]: row for row in _load_manifest(store, "questions", "techqa")}
    judge = _generator(config, "qwen")
    records: list[dict[str, Any]] = []
    evaluator_hash = canonical_json_hash({"prompt": "techqa-judge-v1", "model": config["models"]["qwen"]})
    for generation in generations:
        question = questions[generation["question_id"]]
        judge_messages = techqa_judge_messages(question["question"], question["references"][0], generation["consumed_context"], generation["normalized_output"])
        ids = list(judge.tokenizer.apply_chat_template(judge_messages, tokenize=True, add_generation_prompt=True))
        if len(ids) > 8192:
            raise ProtocolError("TechQA judge input exceeds frozen 8192-token limit")
        if generation["stopping_reason"] == "failed":
            parsed = {"correctness": 0, "completeness": 0, "groundedness": 0, "reason": "permanently failed generation", "semantic_utility": 0.0}
            records.append(build_evaluation_record(
                generation, question, {"exact_match": 0.0, "f1": 0.0}, evaluator_hash,
                judge={"raw": "", "parsed": parsed, "attempts": [{"attempt": 0, "status": "generation_failed"}]},
            ))
            continue
        parsed: dict[str, Any] | None = None
        attempts: list[dict[str, Any]] = []
        raw = ""
        for attempt in (1, 2):
            raw, _ = judge.generate(ids, 256)
            try:
                parsed = parse_judge_json(raw)
                attempts.append({"attempt": attempt, "status": "success"})
                break
            except (json.JSONDecodeError, ValueError):
                attempts.append({"attempt": attempt, "status": "invalid_json"})
        if parsed is None:
            parsed = {"correctness": 0, "completeness": 0, "groundedness": 0, "reason": "invalid JSON after retry", "semantic_utility": 0.0}
        records.append(build_evaluation_record(
            generation, question, best_answer_metrics(generation["normalized_output"], question["references"]),
            evaluator_hash, judge={"raw": raw, "parsed": parsed, "attempts": attempts},
        ))
    ref = store.write_jsonl(
        f"evaluation/judge/techqa/{source_condition}/part-{item.shard_index:03d}.jsonl",
        records, "evaluation", "evaluation_id",
    )
    return [ref.sha256]


def execute_e5(item: WorkItem, config: Mapping[str, Any], store: ArtifactStore) -> list[str]:
    assert item.dataset is not None
    embedder, stack, policy = item.condition_id.split("__")
    chunks = list(_chunks_by_id(store, item.dataset, policy).values())
    dense_role = "dense" if embedder == "bge" else "canonical"
    dense = config["models"][dense_role]
    reranker = config["models"]["reranker"]
    engine = PrimaryRetriever(
        chunks, dense["repository"], dense["revision"], reranker["repository"], reranker["revision"],
        config["retrieval"]["query_prefix"] if embedder == "bge" else "", "cuda",
    )
    engine.build()
    questions = _load_manifest(store, "questions", item.dataset)
    question_hash = file_sha256(_root(store) / "manifests" / "questions" / f"{item.dataset}.jsonl")
    corpus_hash = file_sha256(_root(store) / "manifests" / "corpora" / f"{item.dataset}.jsonl")
    config_hash = canonical_json_hash({"embedder": embedder, "stack": stack, "policy": policy})
    upstream = canonical_json_hash([question_hash, corpus_hash, file_sha256(_root(store) / "chunks" / item.dataset / f"{policy}.jsonl")])
    traces: list[dict[str, Any]] = []
    for question in questions:
        dense_rows, sparse_rows, fused, reranked, latency = engine.query(question["question"], stack=stack)
        selected = dense_rows if stack == "dense" else fused if stack == "hybrid" else reranked
        normalized_reranked = [{**row, "reranker_score": row["reranker_score"] if "reranker_score" in row else row["fused_score"] if "fused_score" in row else row["score"]} for row in selected]
        traces.append(build_retrieval_record(
            question_id=question["question_id"], condition_id=item.condition_id,
            question_manifest_hash=question_hash, corpus_manifest_hash=corpus_hash,
            dense=dense_rows, sparse=sparse_rows, fused=fused,
            reranked=normalized_reranked, config_hash=config_hash, upstream_hash=upstream,
            latency=latency, memory=_gpu_memory(),
        ))
    ref = store.write_jsonl(f"retrieval/secondary/{item.dataset}/{item.condition_id}.jsonl", traces, "retrieval", "retrieval_id")
    return [ref.sha256]


def execute_e6(item: WorkItem, config: Mapping[str, Any], store: ArtifactStore) -> list[str]:
    if item.condition_id == "gold-4096":
        return _execute_gold(item, config, store, role="qwen")
    source_condition, packing_id = item.condition_id.split("__", 1)
    return _generate_shard(item, config, store, role="qwen", source_condition=source_condition, packing_id=packing_id)


def execute_e7(item: WorkItem, config: Mapping[str, Any], store: ArtifactStore) -> list[str]:
    from .validation import validate_repository
    from .artifacts import validate_record_links
    from .reproducibility import compare_generation, compare_metrics, compare_retrieval
    from .canonical import atomic_write_jsonl

    report = validate_repository(repo_root())
    report.update({"schema_version": PROTOCOL_ID, "protocol_sha256": PROTOCOL_SHA256, "experiment": "E7", "work_id": item.work_id, "datasets": {}})
    audit_root = _root(store) / "audit" / "recomputed"
    audit_store = ArtifactStore(audit_root)
    audit_store.initialize()
    for dataset in config["dataset_order"]:
        questions = sorted(_load_manifest(store, "questions", dataset), key=lambda row: (row["selection_hash"], row["question_id"]))[:25]
        corpus = _load_manifest(store, "corpora", dataset)
        clusters = [row for row in _load_manifest(store, "clusters", dataset) if any(question_id in {q["question_id"] for q in questions} for question_id in row["question_ids"])]
        gold = [row for row in _load_manifest(store, "gold", dataset) if row["question_id"] in {q["question_id"] for q in questions}]
        audit_store.write_jsonl(f"manifests/questions/{dataset}.jsonl", questions, "question", "question_id")
        audit_store.write_jsonl(f"manifests/corpora/{dataset}.jsonl", corpus, "corpus", "document_id")
        audit_store.write_jsonl(f"manifests/clusters/{dataset}.jsonl", clusters, "cluster", "cluster_id")
        audit_store.write_jsonl(f"manifests/gold/{dataset}.jsonl", gold, "gold", "gold_id")
        for condition in [*POLICY_ORDER, *[f"{policy}-jitter-{seed}" for policy in POLICY_ORDER for seed in JITTER_SEEDS]]:
            audit_item = WorkItem("E1", dataset, condition, None, 25, ("E0",))
            execute_e1(audit_item, config, audit_store)
            original = _retrieval_by_question(store, dataset, condition)
            recomputed = _retrieval_by_question(audit_store, dataset, condition)
            for question in questions:
                compare_retrieval(original[question["question_id"]], recomputed[question["question_id"]])
        for condition in condition_ids_e2():
            audit_item = WorkItem("E2", dataset, condition, 0, 25, ("E1",))
            execute_e2(audit_item, config, audit_store)
        for condition in ("gold-1024", "gold-4096"):
            audit_item = WorkItem("E3", dataset, condition, 0, 25, ("E0",))
            execute_e3(audit_item, config, audit_store)
        for condition in [*condition_ids_e2(), "gold-1024", "gold-4096"]:
            original_generation_root = _root(store) / "generation" / "mistral" / dataset / condition
            original_generations = {row["question_id"]: row for path in sorted(original_generation_root.glob("part-*.jsonl")) for row in read_jsonl(path) if row["question_id"] in {q["question_id"] for q in questions}}
            recomputed_path = audit_root / "generation" / "mistral" / dataset / condition / "part-000.jsonl"
            recomputed_generations = {row["question_id"]: row for row in read_jsonl(recomputed_path)}
            original_eval_root = _root(store) / "evaluation" / "automatic" / "mistral" / dataset / condition
            original_eval_by_generation = {row["generation_id"]: row for path in sorted(original_eval_root.glob("part-*.jsonl")) for row in read_jsonl(path)}
            recomputed_eval_path = audit_root / "evaluation" / "automatic" / "mistral" / dataset / condition / "part-000.jsonl"
            recomputed_eval_by_generation = {row["generation_id"]: row for row in read_jsonl(recomputed_eval_path)}
            for question in questions:
                qid = question["question_id"]
                compare_generation(original_generations[qid], recomputed_generations[qid])
                compare_metrics(original_eval_by_generation[original_generations[qid]["generation_id"]], recomputed_eval_by_generation[recomputed_generations[qid]["generation_id"]])
        report["datasets"][dataset] = {"questions": [row["question_id"] for row in questions], "status": "exact"}
    retrieval_rows = [row for path in (_root(store) / "retrieval").glob("**/*.jsonl") for row in read_jsonl(path)]
    generation_rows = [row for path in (_root(store) / "generation").glob("**/part-*.jsonl") for row in read_jsonl(path)]
    report["cost_summary"] = {
        "storage_bytes": sum(path.stat().st_size for path in _root(store).rglob("*") if path.is_file()),
        "retrieval_records": len(retrieval_rows), "generation_records": len(generation_rows),
        "retrieval_seconds": sum(sum(float(value) for value in row["latency"].values()) for row in retrieval_rows),
        "generation_seconds": sum(sum(float(value) for value in row["latency"].values()) for row in generation_rows),
        "prompt_input_tokens": sum(int(row["used_input_tokens"]) for row in generation_rows),
        "generated_tokens": sum(int(row["generated_tokens"]) for row in generation_rows),
        "index_bytes": sum(path.stat().st_size for path in (_root(store) / "chunks").glob("**/*.jsonl")),
        "peak_gpu_memory_bytes": max(
            [int(value) for row in retrieval_rows for value in row.get("memory", {}).values() if isinstance(value, int)]
            + [int(row.get("hardware", {}).get("peak_gpu_memory_bytes", 0)) for row in generation_rows]
            + [0]
        ),
    }
    primary_retrieval_rows = [row for path in (_root(store) / "retrieval" / "primary").glob("**/*.jsonl") for row in read_jsonl(path)]
    gold_rows = [row for dataset in config["dataset_order"] for row in _load_manifest(store, "gold", dataset)]
    validate_record_links([*primary_retrieval_rows, *gold_rows], generation_rows)
    evaluation_rows = [row for path in (_root(store) / "evaluation").glob("**/part-*.jsonl") for row in read_jsonl(path)]
    validate_record_links(generation_rows, evaluation_rows)
    report["hash_chain"] = {"generation_links": len(generation_rows), "evaluation_links": len(evaluation_rows), "status": "valid"}
    report["status"] = "complete"
    path = _root(store) / "audit" / "e7-repository-validation.json"
    from .canonical import atomic_write_json

    return [atomic_write_json(path, report)]


HANDLERS = {
    "E1": execute_e1, "E2": execute_e2, "E3": execute_e3, "E4": execute_e4,
    "E5": execute_e5, "E6": execute_e6, "E7": execute_e7,
}
