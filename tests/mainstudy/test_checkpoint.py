"""Specification Section 28: interruption, resume, finalization, and merge."""

import tempfile
import unittest
from pathlib import Path

from chunkrag.mainstudy.checkpoint import CheckpointError, ShardCheckpoint, merge_shards, shard_question_ids
from chunkrag.mainstudy.canonical import read_jsonl


class CheckpointTests(unittest.TestCase):
    def test_sharding_is_sorted_and_fixed(self) -> None:
        self.assertEqual(shard_question_ids(["c", "a", "b"], 2), [["a", "b"], ["c"]])

    def test_resume_and_finalize(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = ShardCheckpoint(Path(directory), "E2", "d", "c", 0, ["q1", "q2"], "a" * 64, "b" * 64, schema=None)
            one = {"question_id": "q1", "value": 1}
            two = {"question_id": "q2", "value": 2}
            checkpoint.append("q1", one)
            checkpoint.append("q1", one)
            with self.assertRaises(CheckpointError):
                checkpoint.append("q1", {"question_id": "q1", "value": 9})
            checkpoint.append("q2", two)
            final = checkpoint.finalize(lambda row: row["question_id"])
            self.assertEqual([row["question_id"] for row in read_jsonl(final)], ["q1", "q2"])

    def test_merge_rejects_missing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "part-000.jsonl").write_text('{"question_id":"q1"}\n', encoding="utf-8")
            with self.assertRaises(CheckpointError):
                merge_shards([root / "part-000.jsonl"], ["q1", "q2"], "question_id", root / "merged.jsonl")


if __name__ == "__main__":
    unittest.main()
