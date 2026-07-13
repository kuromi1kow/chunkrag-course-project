"""Pinned local generation and trace construction (Specification Sections 14, 16, 23.4)."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import asdict
from typing import Any

from .canonical import canonical_json_hash, identifier_hash
from .constants import PROTOCOL_ID
from .packing import PackedContext
from .prompts import messages, normalize_generated_answer, prompt_version
from .schemas import validate_record


class GenerationError(RuntimeError):
    pass


class LocalGenerator:
    """Lazy Transformers adapter; model construction and ``generate`` are Phase-4 actions."""

    def __init__(self, repository: str, revision: str, *, device: str = "cuda") -> None:
        self.repository = repository
        self.revision = revision
        self.device = device
        self.tokenizer: Any = None
        self.model: Any = None

    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.repository, revision=self.revision, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.repository, revision=self.revision, dtype=torch.float16, local_files_only=True,
        ).to(self.device)
        self.model.eval()

    def generate(self, prompt_token_ids: list[int], max_new_tokens: int) -> tuple[str, dict[str, Any]]:
        import torch

        if self.model is None or self.tokenizer is None:
            raise GenerationError("Local generator must be loaded before inference")
        input_ids = torch.tensor([prompt_token_ids], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(
                input_ids=input_ids, attention_mask=attention_mask,
                max_new_tokens=max_new_tokens, do_sample=False,
                num_beams=1, repetition_penalty=1.0, no_repeat_ngram_size=0,
                length_penalty=1.0, use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        elapsed = time.perf_counter() - started
        generated = output[0, input_ids.shape[1]:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        stopped_eos = bool(len(generated) and int(generated[-1]) == int(self.tokenizer.eos_token_id))
        return text, {
            "generated_tokens": int(len(generated)), "latency_seconds": elapsed,
            "stopping_reason": "eos" if stopped_eos else "max_new_tokens",
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
        }


def build_generation_record(
    *, question: Mapping[str, Any], condition_id: str, control_seed: int | None,
    packing_id: str, budget: int, packed: PackedContext, model_repository: str,
    model_revision: str, model_snapshot_hash: str, retrieval_or_gold_hash: str,
    prompt_version_hash: str, raw_output: str, generated_tokens: int, stopping_reason: str,
    latency: Mapping[str, float], attempt_history: list[dict[str, Any]], hardware: Mapping[str, Any],
) -> dict[str, Any]:
    generation_id = identifier_hash(
        retrieval_or_gold_hash, question["question_id"], model_snapshot_hash, packing_id, prompt_version_hash,
    )
    record = {
        "schema_version": PROTOCOL_ID, "generation_id": generation_id,
        "question_id": question["question_id"], "retrieval_or_gold_hash": retrieval_or_gold_hash,
        "condition_id": condition_id, "control_seed": control_seed, "packing_id": packing_id,
        "budget": budget, "ranked_source_spans": list(packed.spans),
        "rendered_context": packed.rendered_context, "consumed_context": packed.consumed_context,
        "per_chunk_consumed_tokens": list(packed.per_chunk_consumed_tokens), "prompt_version": prompt_version(question["dataset"]),
        "prompt_version_hash": prompt_version_hash,
        "messages": messages(question["dataset"], question["question"], packed.consumed_context),
        "prompt_token_ids": list(packed.prompt_token_ids), "full_input_tokens": packed.full_prompt_tokens,
        "used_input_tokens": len(packed.prompt_token_ids), "context_target": packed.target,
        "truncation_location": packed.truncation_location, "model_repository": model_repository,
        "model_revision": model_revision, "model_snapshot_hash": model_snapshot_hash,
        "dtype": "float16", "hardware": dict(hardware), "raw_output": raw_output,
        "normalized_output": normalize_generated_answer(raw_output), "generated_tokens": generated_tokens,
        "stopping_reason": stopping_reason, "latency": dict(latency),
        "attempt_history": attempt_history, "upstream_hash": retrieval_or_gold_hash,
    }
    record["record_hash"] = canonical_json_hash(record)
    validate_record("generation", record)
    return record


def resolved_generation_config(max_new_tokens: int, tokenizer: Any) -> dict[str, Any]:
    return {
        "max_new_tokens": max_new_tokens, "do_sample": False, "num_beams": 1,
        "temperature": None, "repetition_penalty": 1.0, "no_repeat_ngram_size": 0,
        "length_penalty": 1.0, "use_cache": True,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
    }
