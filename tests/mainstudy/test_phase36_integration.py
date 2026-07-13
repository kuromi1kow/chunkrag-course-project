"""Small synthetic integration tests for canonical outputs and fail-closed E4 execution."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from chunkrag.mainstudy.artifacts import ArtifactStore
from chunkrag.mainstudy.canonical import read_json, read_jsonl
from chunkrag.mainstudy.canonical import file_sha256
from chunkrag.mainstudy.completion import mark_work_complete
from chunkrag.mainstudy.execution import execute_e4
from chunkrag.mainstudy.experiments import WorkItem
from chunkrag.mainstudy.outputs import write_retrieval_outputs
from chunkrag.mainstudy.protocol import ProtocolError
from chunkrag.mainstudy.runner import main as runner_main


class Phase36IntegrationTests(unittest.TestCase):
    def test_retrieval_metrics_and_cost_outputs_are_materialized(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            question = {"question_id": "q", "gold_document_ids": ["d"], "gold_spans": [{"document_id": "d", "char_start": 1, "char_end": 3}], "supporting_facts": []}
            chunk = {"chunk_id": "c", "document_id": "d", "char_start": 0, "char_end": 4, "token_count": 4}
            trace = {"retrieval_id": "r", "question_id": "q", "condition_id": "fixed192", "top16_chunk_ids": ["c"], "latency": {"dense_seconds": 1, "sparse_seconds": 2, "reranker_seconds": 3}}
            hashes = write_retrieval_outputs(root, namespace="primary", dataset="squad_v2", condition_id="fixed192", questions=[question], chunks=[chunk], traces=[trace], build_audit={"index_build_seconds": 4, "embedding_tokens": 4, "dense_token_counts": [4], "index_bytes": 8, "index_vectors": 1, "embedding_dtype": "float32"})
            self.assertEqual(len(hashes), 3)
            metrics = read_jsonl(root / "analysis/retrieval/primary/squad_v2/fixed192.jsonl")[0]
            self.assertEqual(metrics["answer_span_at_4"], 1.0)
            cost = read_json(root / "audit/cost/primary/squad_v2/fixed192.json")
            self.assertEqual(cost["index_bytes"], 8)
            self.assertEqual(cost["warmup_questions"], 5)

    def test_judge_cannot_run_before_human_package_marker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(Path(directory)); store.initialize()
            item = WorkItem("E4", "techqa", "judge__gold-4096", 0, 50, ("E2", "E3"))
            with self.assertRaises(ProtocolError): execute_e4(item, {}, store)

    def test_judge_cannot_run_before_human_labels_are_collected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(Path(directory)); store.initialize()
            package = store.root / "evaluation" / "human" / "techqa-package.json"
            package.parent.mkdir(parents=True, exist_ok=True); package.write_text('{"records":[],"schema_version":"chunkrag-main-v1"}\n')
            package_item = WorkItem("E4", "techqa", "human-package", None, 360, ("E2", "E3"))
            mark_work_complete(store.root, package_item, [file_sha256(package)], git_commit="g", config_sha256="c" * 64, environment_hash="e" * 64)
            judge_item = WorkItem("E4", "techqa", "judge__gold-4096", 0, 50, ("E2", "E3"))
            with self.assertRaisesRegex(ProtocolError, "human labels"):
                execute_e4(judge_item, {}, store)

    def test_artifact_lock_removes_write_permissions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(Path(directory)); store.initialize()
            path = store.root / "generation" / "x"; path.parent.mkdir(parents=True, exist_ok=True); path.write_text("x")
            store.lock_read_only()
            self.assertEqual(path.stat().st_mode & 0o222, 0)
            self.assertEqual(path.parent.stat().st_mode & 0o222, 0)
            for candidate in (store.root / "generation", path.parent): candidate.chmod(0o755)
            path.chmod(0o644)

    def test_runner_activates_determinism_and_runtime_gate_before_handler(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            events = []
            config = {"artifact_root": "unused", "config_sha256": "c" * 64}
            with patch("chunkrag.mainstudy.runner.load_protocol_config", return_value=config), \
                 patch("chunkrag.mainstudy.runner.require_clean_git", return_value={"commit": "g", "dirty": False}), \
                 patch("chunkrag.mainstudy.runner.configure_determinism", side_effect=lambda **_: events.append("determinism")), \
                 patch("chunkrag.mainstudy.runner.environment_manifest", side_effect=lambda *_, **__: events.append("environment") or {"lock_sha256": "e" * 64, "hardware": {}, "packages": []}), \
                 patch("chunkrag.mainstudy.runner.require_canonical_runtime", side_effect=lambda *_, **__: events.append("runtime")), \
                 patch("chunkrag.mainstudy.runner.completed_stages", return_value=[]), \
                 patch("chunkrag.mainstudy.runner.mark_work_complete", return_value="m" * 64), \
                 patch("chunkrag.mainstudy.runner.finalize_stage", return_value=None), \
                 patch("chunkrag.mainstudy.runner.source_sha256", return_value="s" * 64), \
                 patch("chunkrag.mainstudy.stages.execute_work_item", side_effect=lambda *args: events.append("handler") or []):
                result = runner_main(["--experiment", "E0", "--mode", "run", "--dataset", "squad_v2", "--artifact-root", directory])
            self.assertEqual(result, 0)
            self.assertLess(events.index("determinism"), events.index("environment"))
            self.assertLess(events.index("runtime"), events.index("handler"))


if __name__ == "__main__": unittest.main()
