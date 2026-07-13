"""Specification Sections 23 and 29: hashed stage completion gates."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from chunkrag.mainstudy.completion import completed_stages, finalize_stage, mark_work_complete
from chunkrag.mainstudy.experiments import WorkItem
from chunkrag.mainstudy.canonical import file_sha256


class CompletionTests(unittest.TestCase):
    def test_stage_finalizes_only_after_all_markers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            one = WorkItem("E0", "d", "one", None, 1, ())
            two = WorkItem("E0", None, "two", None, 1, ())
            first = root / "first.json"
            second = root / "second.json"
            first.write_text('{"one":1}\n', encoding="utf-8")
            second.write_text('{"two":2}\n', encoding="utf-8")
            with patch("chunkrag.mainstudy.completion.plan_experiment", return_value=[one, two]):
                mark_work_complete(root, one, [file_sha256(first)], git_commit="g", config_sha256="c" * 64, environment_hash="e" * 64)
                provenance = {"git_commit": "g", "config_sha256": "c" * 64, "environment_hash": "e" * 64}
                self.assertIsNone(finalize_stage(root, "E0", **provenance))
                mark_work_complete(root, two, [file_sha256(second)], git_commit="g", config_sha256="c" * 64, environment_hash="e" * 64)
                self.assertIsNotNone(finalize_stage(root, "E0", **provenance))


if __name__ == "__main__":
    unittest.main()
