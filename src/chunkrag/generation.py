from __future__ import annotations

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def resolve_torch_dtype(torch_dtype: str | None):
    if torch_dtype is None:
        return None
    if torch_dtype == "auto":
        return "auto"
    if not hasattr(torch, torch_dtype):
        raise ValueError(f"Unsupported torch dtype: {torch_dtype}")
    return getattr(torch, torch_dtype)


class QAGenerator:
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        max_input_tokens: int = 768,
        max_new_tokens: int = 32,
        torch_dtype: str | None = None,
        use_device_map: bool = False,
    ) -> None:
        self.device = resolve_device(device)
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        config = AutoConfig.from_pretrained(model_name)
        self.is_encoder_decoder = bool(config.is_encoder_decoder)
        model_kwargs = {}
        resolved_dtype = resolve_torch_dtype(torch_dtype)
        if resolved_dtype is not None:
            model_kwargs["torch_dtype"] = resolved_dtype
        if use_device_map:
            model_kwargs["device_map"] = "auto"
        if self.is_encoder_decoder:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **model_kwargs)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        if not use_device_map:
            self.model.to(self.device)
        self.model.eval()
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens

    def _build_prompt(self, question: str, context: str | None) -> str:
        if context:
            return (
                "Answer the question using only the provided context. "
                "If the answer is not supported by the context, answer with 'unanswerable'.\n\n"
                f"Question: {question}\n\n"
                f"Context:\n{context}\n\n"
                "Answer:"
            )
        return (
            "Answer the question as briefly as possible.\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )

    @torch.inference_mode()
    def answer(self, question: str, context: str | None = None) -> str:
        prompt = self._build_prompt(question, context)
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        ).to(self.device)
        generated = self.model.generate(
            **encoded,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        if self.is_encoder_decoder:
            text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        else:
            prompt_length = encoded["input_ids"].shape[1]
            text = self.tokenizer.decode(generated[0][prompt_length:], skip_special_tokens=True)
        return text.strip()
