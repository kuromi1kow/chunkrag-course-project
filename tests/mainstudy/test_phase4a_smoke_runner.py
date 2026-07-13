from __future__ import annotations

from pathlib import Path

import pytest

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
