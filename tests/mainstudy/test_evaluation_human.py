"""Specification Sections 18--22: metrics, judge schema, and blinding."""

import unittest

from chunkrag.mainstudy.evaluation import (
    best_answer_metrics, document_metrics, interval_fully_covered, parse_judge_json,
)
from chunkrag.mainstudy.human import HUMAN_CONDITIONS, blindness_scan, build_blinded_package, build_training_package


class EvaluationHumanTests(unittest.TestCase):
    def test_answer_and_evidence_metrics(self) -> None:
        metrics = best_answer_metrics("The Boston.", ["Boston"])
        self.assertEqual(metrics, {"exact_match": 1.0, "f1": 1.0})
        self.assertTrue(interval_fully_covered(2, 8, [(0, 4), (4, 9)]))
        self.assertEqual(document_metrics(["x", "g"], ["g"], 2)["mrr"], 0.5)

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


if __name__ == "__main__":
    unittest.main()
