"""Specification Section 30: all frozen main paper artifacts render from synthetic data."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from chunkrag.mainstudy.paper import _load_pyplot, regenerate_paper_artifacts


class PaperArtifactTests(unittest.TestCase):
    def test_headless_backend_ignores_colab_inline_ambient_value(self) -> None:
        with patch.dict("os.environ", {"MPLBACKEND": "module://matplotlib_inline.backend_inline"}):
            pyplot = _load_pyplot()
            self.assertEqual(pyplot.get_backend().lower(), "agg")

    def test_three_tables_and_three_figures_render(self) -> None:
        primary = []
        exposure = []
        for dataset in ("squad_v2", "hotpot_qa", "techqa"):
            for policy in ("recursive192", "sentence192", "semantic192"):
                primary.append({"test_id": f"H1:{dataset}:{policy}", "mean_difference": 0.1, "ci95_low": -0.1, "ci95_high": 0.3, "rank_biserial": 0.1, "raw_p": 0.5, "holm_p": 1.0})
                for condition in ("operational-1024", "matched-1024", "operational-4096"):
                    exposure.append({"dataset": dataset, "policy": policy, "condition": condition, "answer_mean": 0.1, "answer_ci_low": -0.1, "answer_ci_high": 0.3, "evidence_mean": 0.01, "evidence_ci_low": -0.02, "evidence_ci_high": 0.04})
        analysis = {
            "primary": primary, "budget": [], "exposure_rows": exposure,
            "dataset_summary": [{"dataset": name, "questions": 1, "documents": 1, "clusters": 1, "question_hash": "a" * 64, "corpus_hash": "b" * 64} for name in ("squad_v2", "hotpot_qa", "techqa")],
            "gold_techqa": {"gold": [], "techqa_semantic_utility": 0.5, "techqa_groundedness": 0.5},
            "techqa": {"validated": True, "remove_from_main": False},
        }
        with tempfile.TemporaryDirectory() as directory:
            outputs = regenerate_paper_artifacts(analysis, Path(directory))
            self.assertEqual(len([path for path in outputs if path.suffix == ".tex"]), 3)
            self.assertEqual(len([path for path in outputs if path.suffix == ".pdf"]), 3)
            self.assertTrue(all(path.stat().st_size > 0 for path in outputs))


if __name__ == "__main__":
    unittest.main()
