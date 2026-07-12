from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.analyze_reviewer_robustness import (
    answer_visibility_is_applicable,
    apply_global_holm_family,
    config_hash,
    validate_completed_root,
    validate_generation_trace,
)


class RobustnessAnalysisTests(unittest.TestCase):
    @staticmethod
    def _write_completed_run(root: Path, config: dict, *, rows: list[dict] | None = None) -> None:
        rows = [] if rows is None else rows
        (root / "experiment_config.json").write_text(json.dumps(config), encoding="utf-8")
        (root / "all_results.json").write_text(json.dumps(rows), encoding="utf-8")
        (root / "run_manifest.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "config_sha256": config_hash(config),
                    "source_tree_sha256": "a" * 64,
                    "num_summary_rows": len(rows),
                }
            ),
            encoding="utf-8",
        )

    def test_completed_root_requires_matching_config_hash(self) -> None:
        config = {"seed": 7, "datasets": [], "chunkers": []}
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "run"
            root.mkdir()
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            self._write_completed_run(root, config)
            (root / "run_manifest.json").write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "config_sha256": "wrong",
                        "source_tree_sha256": "a" * 64,
                        "num_summary_rows": 0,
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Config hash mismatch"):
                validate_completed_root(root, config_path)

            self._write_completed_run(root, config)
            self.assertEqual(validate_completed_root(root, config_path), config)

    def test_completed_root_rejects_manifest_row_count_mismatch(self) -> None:
        config = {"seed": 7, "datasets": [], "chunkers": []}
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "run"
            root.mkdir()
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            self._write_completed_run(root, config, rows=[{"system": "x"}])
            manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
            manifest["num_summary_rows"] = 2
            (root / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "summary-row count mismatch"):
                validate_completed_root(root, config_path)

    def test_answer_visibility_excludes_hotpot_binary_labels(self) -> None:
        self.assertFalse(answer_visibility_is_applicable("hotpot_qa", ["Yes"]))
        self.assertFalse(answer_visibility_is_applicable("hotpot_qa", ["no"]))
        self.assertTrue(answer_visibility_is_applicable("hotpot_qa", ["Boston"]))
        self.assertTrue(answer_visibility_is_applicable("squad_v2", ["yes"]))

    def test_generation_trace_enforces_configured_budgets(self) -> None:
        trace = {
            "full_prompt_tokens": 1600,
            "used_prompt_tokens": 1536,
            "generated_tokens": 96,
            "generation_max_new_tokens": 96,
            "context_truncated": True,
            "refinement_applied": False,
            "generation_length_capped": True,
        }
        validate_generation_trace(
            trace,
            expected_input_limit=1536,
            expected_output_limit=96,
            label="fixture",
        )
        trace["generated_tokens"] = 97
        with self.assertRaisesRegex(ValueError, "Generated tokens exceed"):
            validate_generation_trace(
                trace,
                expected_input_limit=1536,
                expected_output_limit=96,
                label="fixture",
            )

    def test_global_holm_family_covers_both_generators(self) -> None:
        def generation_shell() -> dict:
            return {
                "paired_f1_against_recursive_254": {
                    "datasets": {
                        dataset: {comparator: {} for comparator in ("fixed_128", "fixed_254", "sentence_254")}
                        for dataset in ("squad_v2", "hotpot_qa", "techqa")
                    }
                }
            }

        generations = {"Qwen": generation_shell(), "Mistral": generation_shell()}
        raw = {
            label: {
                f"{dataset}::{comparator}": 0.01 + index * 0.001
                for index, (dataset, comparator) in enumerate(
                    (d, c)
                    for d in ("squad_v2", "hotpot_qa", "techqa")
                    for c in ("fixed_128", "fixed_254", "sentence_254")
                )
            }
            for label in generations
        }

        metadata = apply_global_holm_family(generations, raw)

        self.assertEqual(metadata["family_size"], 18)
        for generation in generations.values():
            self.assertEqual(
                generation["paired_f1_against_recursive_254"]["primary_holm_family_size"],
                18,
            )
            for dataset in ("squad_v2", "hotpot_qa", "techqa"):
                for comparator in ("fixed_128", "fixed_254", "sentence_254"):
                    self.assertIn(
                        "randomization_p_holm_global",
                        generation["paired_f1_against_recursive_254"]["datasets"][dataset][comparator],
                    )


if __name__ == "__main__":
    unittest.main()
