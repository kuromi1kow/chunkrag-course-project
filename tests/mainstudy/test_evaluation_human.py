"""Specification Sections 18--22: metrics, judge schema, and blinding."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from chunkrag.mainstudy.artifacts import ArtifactStore
from chunkrag.mainstudy.canonical import atomic_write_json, atomic_write_jsonl
from chunkrag.mainstudy.execution import execute_e4
from chunkrag.mainstudy.evaluation import (
    best_answer_metrics, document_metrics, interval_fully_covered, parse_judge_json,
    supporting_fact_fully_covered,
)
from chunkrag.mainstudy.experiments import WorkItem
from chunkrag.mainstudy.human import HUMAN_CONDITIONS, blindness_scan, build_blinded_package, build_training_package


class EvaluationHumanTests(unittest.TestCase):
    def test_answer_and_evidence_metrics(self) -> None:
        metrics = best_answer_metrics("The Boston.", ["Boston"])
        self.assertEqual(metrics, {"exact_match": 1.0, "f1": 1.0})
        self.assertTrue(interval_fully_covered(2, 8, [(0, 4), (4, 9)]))
        self.assertEqual(document_metrics(["x", "g"], ["g"], 2)["mrr"], 0.5)

    def test_unavailable_supporting_fact_is_uncovered(self) -> None:
        missing = {"document_id": "d", "sentence_index": 902, "char_start": None, "char_end": None}
        valid = {"document_id": "d", "sentence_index": 1, "char_start": 2, "char_end": 8}
        intervals = {"d": [(0, 4), (4, 9)]}
        self.assertFalse(supporting_fact_fully_covered(missing, intervals))
        self.assertTrue(supporting_fact_fully_covered(valid, intervals))

    def test_judge_json_strict(self) -> None:
        parsed = parse_judge_json('{"correctness":2,"completeness":1,"groundedness":2,"reason":"ok"}')
        self.assertEqual(parsed["semantic_utility"], 0.75)
        with self.assertRaises(ValueError):
            parse_judge_json('{"correctness":3,"completeness":1,"groundedness":2,"reason":"x"}')

    def test_human_package_is_blinded_and_sized(self) -> None:
        questions = [{"question_id": f"q{i:03d}", "question": "Q", "references": ["R"]} for i in range(70)]
        generations = {}
        for question in questions:
            for condition in HUMAN_CONDITIONS:
                generations[(question["question_id"], condition)] = {
                    "generation_id": f"{question['question_id']}-{condition}",
                    "normalized_output": "A", "consumed_context": "C",
                }
        package = build_blinded_package(questions, generations)
        self.assertEqual(len(package), 360)
        blindness_scan(package)
        self.assertEqual(sum(row["groundedness_subset"] for row in package), 60)
        self.assertEqual(len(build_training_package(questions, generations)), 20)

    def test_e4_judge_branch_reads_frozen_package_without_local_scope_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = ArtifactStore(root)
            store.initialize()
            human_root = root / "evaluation" / "human"
            atomic_write_json(human_root / "techqa-package.json", {"records": []})
            for name in ("human-labels-a.jsonl", "human-labels-b.jsonl", "human-adjudicated.jsonl"):
                atomic_write_jsonl(human_root / name, [], "annotation_record_id")
            atomic_write_jsonl(root / "manifests" / "questions" / "techqa.jsonl", [], "question_id")
            condition = "fixed192__matched-4096"
            atomic_write_jsonl(
                root / "generation" / "mistral" / "techqa" / condition / "part-000.jsonl",
                [], "generation_id",
            )
            fake_judge = type("FakeJudge", (), {"repository": "qwen", "revision": "rev"})()
            item = WorkItem("E4", "techqa", f"judge__{condition}", 0, 0, ("E3",))
            config = {"models": {"qwen": {"repository": "qwen", "revision": "rev"}}}
            with (
                patch("chunkrag.mainstudy.completion.completed_work_ids", return_value={"E4/techqa/human-package"}),
                patch("chunkrag.mainstudy.execution._generator", return_value=fake_judge),
                patch("chunkrag.mainstudy.execution._snapshot_hash", return_value="a" * 64),
            ):
                hashes = execute_e4(item, config, store)
            self.assertEqual(len(hashes), 1)
            self.assertTrue(
                (root / "evaluation" / "judge" / "techqa" / condition / "part-000.jsonl").is_file()
            )


if __name__ == "__main__":
    unittest.main()
