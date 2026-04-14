from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)


@runtime_checkable
class Generator(Protocol):
    def answer(self, question: str, context: str | None = None) -> str:
        ...

    def chat(self, messages: list[dict[str, str]]) -> str:
        ...


def build_qa_prompt(question: str, context: str | None) -> str:
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


def build_chat_prompt(messages: list[dict[str, str]]) -> str:
    lines = [
        "You are a helpful, concise assistant.",
        "Respond naturally to greetings and short chat messages.",
        "If the user's message is unclear, ask a brief clarifying question instead of saying unanswerable.",
        "",
        "Conversation:",
    ]
    for message in messages:
        role = message["role"].strip().capitalize()
        content = message["content"].strip()
        if not content:
            continue
        lines.append(f"{role}: {content}")
    lines.append("Assistant:")
    return "\n".join(lines)


def build_openai_qa_messages(question: str, context: str | None) -> list[dict[str, str]]:
    if context:
        return [
            {
                "role": "system",
                "content": (
                    "You are an extractive question answering assistant. "
                    "Use only the provided context. "
                    "Return the shortest answer span supported by the context. "
                    "Do not explain your reasoning. "
                    "If the answer is not fully supported, reply with exactly 'unanswerable'."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Answer the following question using only the context.\n\n"
                    f"Question: {question}\n\n"
                    "Context passages:\n"
                    f"{context}\n\n"
                    "Return only the answer text."
                ),
            },
        ]
    return [
        {
            "role": "system",
            "content": (
                "You are a concise question answering assistant. "
                "Return only a short answer phrase with no explanation."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {question}\n\n"
                "Return only the answer text."
            ),
        },
    ]


def normalize_qa_response(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        return cleaned

    cleaned = cleaned.splitlines()[0].strip()
    cleaned = re.sub(r"^\s*(answer|final answer)\s*:\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^\s*the answer is\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.strip().strip("\"'`")

    lowered = cleaned.lower()
    if "unanswerable" in lowered or "not answerable" in lowered or "not supported" in lowered:
        return "unanswerable"
    return cleaned


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

    @torch.inference_mode()
    def generate_from_prompt(self, prompt: str, max_new_tokens: int | None = None) -> str:
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        ).to(self.device)
        generated = self.model.generate(
            **encoded,
            max_new_tokens=max_new_tokens or self.max_new_tokens,
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

    @torch.inference_mode()
    def answer(self, question: str, context: str | None = None) -> str:
        prompt = build_qa_prompt(question, context)
        return self.generate_from_prompt(prompt)

    def chat(self, messages: list[dict[str, str]]) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                prompt = build_chat_prompt(messages)
        else:
            prompt = build_chat_prompt(messages)
        return self.generate_from_prompt(prompt)


class ExtractiveFallbackGenerator:
    def __init__(self, max_sentences: int = 3, max_characters: int = 360) -> None:
        self.max_sentences = max_sentences
        self.max_characters = max_characters

    def answer(self, question: str, context: str | None = None) -> str:
        if not context:
            return "I do not have grounded context yet."

        cleaned_lines: list[str] = []
        for line in context.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.lower().startswith("matched queries:"):
                continue
            if stripped.lower().startswith("title:"):
                continue
            if re.match(r"^\[\d+\]\s+title:", stripped, flags=re.IGNORECASE):
                continue
            stripped = re.sub(r"^#+\s*", "", stripped)
            stripped = re.sub(r"^[-*]\s*", "", stripped)
            cleaned_lines.append(stripped)
        cleaned_context = ". ".join(cleaned_lines) if cleaned_lines else context
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+|\n+", cleaned_context)
            if sentence.strip()
        ]
        if not sentences:
            return cleaned_context.strip()[: self.max_characters]

        query_terms = {term for term in re.findall(r"\w+", question.lower()) if len(term) > 2}
        scored: list[tuple[int, float, str]] = []
        for index, sentence in enumerate(sentences):
            sentence_terms = set(re.findall(r"\w+", sentence.lower()))
            if len(sentence_terms) < 4:
                continue
            overlap = len(query_terms & sentence_terms)
            score = overlap + (0.2 if any(term in sentence.lower() for term in query_terms) else 0.0)
            scored.append((index, score, sentence))

        if not scored:
            scored = [
                (index, 0.0, sentence)
                for index, sentence in enumerate(sentences)
            ]

        selected = sorted(scored, key=lambda item: (item[1], -item[0]), reverse=True)[: self.max_sentences]
        selected_indices = {index for index, _, _ in selected}
        ordered_sentences = [sentence for index, _, sentence in scored if index in selected_indices]
        answer = " ".join(ordered_sentences).strip()
        if not answer:
            answer = sentences[0]
        return answer[: self.max_characters].strip()

    def chat(self, messages: list[dict[str, str]]) -> str:
        user_messages = [message["content"] for message in messages if message.get("role") == "user" and message.get("content")]
        if not user_messages:
            return "I do not have a user message yet."
        return self.answer(user_messages[-1], context=None)


class OpenAICompatibleGenerator:
    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_key: str,
        tokenizer_name: str | None = None,
        max_input_tokens: int = 1_024,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("Install the demo extras to use OpenAI-compatible endpoints.") from exc

        self.model_name = model_name
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or model_name)
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def answer(self, question: str, context: str | None = None) -> str:
        messages = build_openai_qa_messages(question, context)
        prompt = "\n\n".join(message["content"] for message in messages)
        input_ids = self.tokenizer.encode(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_input_tokens,
        )
        truncated_prompt = self.tokenizer.decode(input_ids, skip_special_tokens=True)
        truncated_messages = [dict(message) for message in messages]
        truncated_messages[-1]["content"] = truncated_prompt
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=truncated_messages,
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
        )
        message = response.choices[0].message
        return normalize_qa_response(message.content or "")

    def chat(self, messages: list[dict[str, str]]) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
        )
        message = response.choices[0].message
        return (message.content or "").strip()
