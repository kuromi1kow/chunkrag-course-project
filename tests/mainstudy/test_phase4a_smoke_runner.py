from __future__ import annotations

from pathlib import Path

import pytest

from chunkrag.mainstudy.artifacts import ArtifactStore
from chunkrag.mainstudy.canonical import atomic_write_jsonl
from chunkrag.mainstudy.execution import _gold_chunks
from chunkrag.mainstudy.protocol import ProtocolError
from scripts.run_phase4a_smoke import QUESTION_COUNT, _subset_corpus, _validate_output_paths


def _document(document_id: str, row_index: int) -> dict[str, object]:
    return {
        "document_id": document_id,
        "source_provenance": [{"row_index": row_index}],
    }


def test_phase4a_output_paths_reject_canonical_and_nonempty_roots(tmp_path: Path) -> None:
    with pytest.raises(ProtocolError, match="canonical artifact root"):
        _validate_output_paths(tmp_path / "chunkrag-main-v1" / "smoke", tmp_path / "paper")
    artifact_root = tmp_path / "phase4a"
    artifact_root.mkdir()
    (artifact_root / "stale.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ProtocolError, match="empty artifact directory"):
        _validate_output_paths(artifact_root, tmp_path / "paper")


def test_phase4a_fixture_keeps_gold_and_retrieval_depth_documents() -> None:
    corpus = [_document(f"doc-{index:03d}", index) for index in range(80)]
    questions = [{"question_id": "q-gold", "gold_document_ids": ["doc-079"]}]
    selected = _subset_corpus("squad_v2", corpus, questions, [])
    selected_ids = {row["document_id"] for row in selected}
    assert len(selected_ids) == 61
    assert "doc-079" in selected_ids
    assert {f"doc-{index:03d}" for index in range(60)} <= selected_ids
    assert QUESTION_COUNT == 5


def test_phase4a_hotpot_fixture_keeps_all_selected_row_documents() -> None:
    corpus = [_document(f"doc-{index:03d}", index) for index in range(80)]
    rows = [{"id": f"q-{index:03d}"} for index in range(80)]
    questions = [{"question_id": "q-079", "gold_document_ids": ["doc-079"]}]
    selected = _subset_corpus("hotpot_qa", corpus, questions, rows)
    assert "doc-079" in {row["document_id"] for row in selected}


def test_hotpot_gold_packing_omits_unavailable_source_sentence(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    store.initialize()
    document = {
        "document_id": "d", "title": "D", "text": "zero one",
        "source_provenance": [{
            "document_index": 0,
            "sentence_spans": [
                {"sentence_index": 0, "char_start": 0, "char_end": 4},
                {"sentence_index": 1, "char_start": 5, "char_end": 8},
            ],
        }],
    }
    atomic_write_jsonl(tmp_path / "manifests/corpora/hotpot_qa.jsonl", [document], "document_id")
    question = {
        "dataset": "hotpot_qa",
        "supporting_facts": [
            {"document_id": "d", "document_index": 0, "sentence_index": 0, "char_start": 0, "char_end": 4},
            {"document_id": "d", "document_index": 0, "sentence_index": 902, "char_start": None, "char_end": None},
        ],
    }
    chunks = _gold_chunks(store, {}, "hotpot_qa", question, None, 128)
    assert [row["text"] for row in chunks] == ["zero", "one"]
