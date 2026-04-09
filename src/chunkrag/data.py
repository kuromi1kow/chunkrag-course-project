from __future__ import annotations

import random
from collections import defaultdict

from datasets import load_dataset

from chunkrag.schemas import Document, QAExample


def load_squad_documents_and_examples(
    split: str,
    max_examples: int,
    candidate_pool_size: int,
    seed: int = 42,
    answerable_only: bool = True,
) -> tuple[list[Document], list[QAExample]]:
    raw = load_dataset("squad_v2", split=split)
    if answerable_only:
        raw = raw.filter(lambda row: len(row["answers"]["text"]) > 0)

    if candidate_pool_size < len(raw):
        pooled = raw.shuffle(seed=seed).select(range(candidate_pool_size))
    else:
        pooled = raw.shuffle(seed=seed)

    title_to_contexts: dict[str, list[str]] = defaultdict(list)
    title_to_seen: dict[str, set[str]] = defaultdict(set)
    title_to_rows: dict[str, list[dict]] = defaultdict(list)
    for row in pooled:
        title = row["title"]
        context = row["context"]
        if context not in title_to_seen[title]:
            title_to_contexts[title].append(context)
            title_to_seen[title].add(context)
        title_to_rows[title].append(row)

    examples: list[QAExample] = []
    selected_titles: set[str] = set()
    title_order = list(title_to_rows)
    random.Random(seed).shuffle(title_order)
    row_indices = {title: 0 for title in title_order}
    while len(examples) < max_examples:
        made_progress = False
        for title in title_order:
            idx = row_indices[title]
            rows = title_to_rows[title]
            if idx >= len(rows):
                continue
            row = rows[idx]
            row_indices[title] += 1
            selected_titles.add(title)
            examples.append(
                QAExample(
                    example_id=row["id"],
                    dataset="squad_v2",
                    question=row["question"],
                    answers=row["answers"]["text"],
                    relevant_doc_ids=[f"squad::{title}"],
                    metadata={"title": title},
                )
            )
            made_progress = True
            if len(examples) >= max_examples:
                break
        if not made_progress:
            break

    documents = [
        Document(
            doc_id=f"squad::{title}",
            title=title,
            text="\n\n".join(title_to_contexts[title]),
            dataset="squad_v2",
        )
        for title in sorted(selected_titles)
    ]
    return documents, examples


def load_hotpot_documents_and_examples(
    split: str,
    max_examples: int,
    config_name: str = "distractor",
    seed: int = 42,
) -> tuple[list[Document], list[QAExample]]:
    raw = load_dataset("hotpot_qa", config_name, split=split)
    raw = raw.shuffle(seed=seed).select(range(min(len(raw), max_examples)))

    documents: dict[str, Document] = {}
    examples: list[QAExample] = []
    for row in raw:
        titles = row["context"]["title"]
        sentence_lists = row["context"]["sentences"]
        doc_ids: list[str] = []
        for title, sentences in zip(titles, sentence_lists):
            doc_id = f"hotpot::{title}"
            doc_ids.append(doc_id)
            text = " ".join(sentences)
            if doc_id not in documents:
                documents[doc_id] = Document(
                    doc_id=doc_id,
                    title=title,
                    text=text,
                    dataset="hotpot_qa",
                )

        supporting_titles = set(row["supporting_facts"]["title"])
        examples.append(
            QAExample(
                example_id=row["id"],
                dataset="hotpot_qa",
                question=row["question"],
                answers=[row["answer"]],
                relevant_doc_ids=[f"hotpot::{title}" for title in supporting_titles],
                metadata={"candidate_doc_ids": doc_ids, "supporting_titles": sorted(supporting_titles)},
            )
        )
    return list(documents.values()), examples
