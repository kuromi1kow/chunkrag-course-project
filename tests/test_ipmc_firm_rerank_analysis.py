from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "analyze_ipmc_firm_rerank.py"
)
SPEC = importlib.util.spec_from_file_location("analyze_ipmc_firm_rerank", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class FirmRerankAnalysisTests(unittest.TestCase):
    def test_archive_excludes_identity_fields_and_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "run"
            archive = Path(tmpdir) / "archive"
            write_json(root / "experiment_config.json", {"seeds": [13, 21, 34]})
            write_json(root / "all_results.json", [{"dataset": "squad_v2"}])
            write_json(root / "aggregate_results.json", [{"dataset": "squad_v2"}])
            write_json(
                root / "run_manifest.json",
                {
                    "status": "complete",
                    "git_commit": "deadbeef",
                    "git_worktree_dirty_at_run": False,
                    "source_tree_sha256": "a" * 64,
                },
            )
            write_json(root / "seed_13" / "raw_predictions.json", [{"text": "passage"}])

            MODULE.archive_summary_artifacts(root, archive)

            manifest = json.loads((archive / "run_manifest.json").read_text())
            self.assertNotIn("git_commit", manifest)
            self.assertNotIn("git_worktree_dirty_at_run", manifest)
            self.assertEqual(manifest["status"], "complete")
            self.assertFalse(any(archive.rglob("*predictions.json")))

    def test_analyze_requires_and_compares_the_complete_paired_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            write_json(
                root / "run_manifest.json",
                {"status": "complete", "source_tree_sha256": "a" * 64},
            )
            summaries = []
            for dataset in MODULE.DATASETS:
                for retriever in MODULE.RETRIEVERS:
                    for chunker in MODULE.CHUNKERS:
                        for seed in MODULE.SEEDS:
                            summaries.append(
                                {
                                    "dataset": dataset,
                                    "retriever": retriever,
                                    "chunker": chunker,
                                    "seed": seed,
                                    "avg_retrieval_latency_s": (
                                        0.02 if retriever == "hybrid_rerank" else 0.01
                                    ),
                                }
                            )
                            prediction = {
                                "example_id": "q1",
                                "all_supporting_docs_found": (
                                    1.0 if retriever == "hybrid_rerank" else 0.0
                                ),
                                "answer_string_visible_at_k": 1.0,
                            }
                            write_json(
                                MODULE.prediction_path(
                                    root,
                                    seed,
                                    dataset,
                                    retriever,
                                    chunker,
                                ),
                                [prediction],
                            )
            write_json(root / "all_results.json", summaries)

            report = MODULE.analyze(root)

            self.assertEqual(len(report["cells"]), 12)
            first = report["cells"][0]
            self.assertEqual(
                first["all_supporting_docs_found"]["paired_delta_mean"],
                1.0,
            )
            self.assertEqual(first["answer_string_visible_at_k"]["paired_delta_mean"], 0.0)
            self.assertEqual(first["latency"]["ratio"], 2.0)
            self.assertEqual(first["flips"]["allhit_gain"], 3)


if __name__ == "__main__":
    unittest.main()
