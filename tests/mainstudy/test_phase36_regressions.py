"""Regression reproductions for the packing and checkpoint failures from the audit."""

import json
import os
import tempfile
import unittest
from pathlib import Path

from chunkrag.mainstudy.canonical import canonical_json_bytes
from chunkrag.mainstudy.checkpoint import CheckpointError, ShardCheckpoint
from chunkrag.mainstudy.packing import longest_prefix


class NonMonotonicTokenizer:
    table = {0: 0, 1: 1, 2: 2, 3: 20, 4: 20, 5: 20, 6: 20, 7: 20, 8: 3, 9: 4, 10: 20}
    def __call__(self, text, **kwargs):
        n = text.count("§")
        result = {"input_ids": list(range(self.table[n]))}
        if kwargs.get("return_offsets_mapping"):
            result["offset_mapping"] = [(index, index + 1) for index in range(len(text))]
        return result
    def apply_chat_template(self, messages, **kwargs):
        n = sum(message["content"].count("§") for message in messages)
        return list(range(10 + self.table[n]))


class Phase36RegressionTests(unittest.TestCase):
    def test_nonmonotonic_tokenizer_returns_true_longest_native_prefix(self) -> None:
        packed = longest_prefix(NonMonotonicTokenizer(), "squad_v2", "q", "§" * 10, input_budget=15)
        self.assertEqual(len(packed.consumed_context), 9)

    def test_crash_after_record_fsync_recovers_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = ShardCheckpoint(Path(directory), "E2", "d", "c", 0, ["q1", "q2"], "a" * 64, "b" * 64, schema=None)
            checkpoint.append("q1", {"question_id": "q1", "value": 1})
            with checkpoint.temp_path.open("ab") as handle:
                handle.write(canonical_json_bytes({"question_id": "q2", "value": 2}))
                handle.flush(); os.fsync(handle.fileno())
            state = checkpoint.validate_partial(lambda row: row["question_id"])
            self.assertEqual(state["completed"], ["q1", "q2"])
            checkpoint.finalize(lambda row: row["question_id"])

    def test_incomplete_tail_is_discarded_but_completed_records_survive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = ShardCheckpoint(Path(directory), "E2", "d", "c", 0, ["q1", "q2"], "a" * 64, "b" * 64, schema=None)
            checkpoint.append("q1", {"question_id": "q1", "value": 1})
            with checkpoint.temp_path.open("ab") as handle: handle.write(b'{"question_id":"q2"')
            state = checkpoint.validate_partial(lambda row: row["question_id"])
            self.assertEqual(state["completed"], ["q1"])

    def test_final_shard_tampering_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = ShardCheckpoint(Path(directory), "E2", "d", "c", 0, ["q1"], "a" * 64, "b" * 64, schema=None)
            checkpoint.append("q1", {"question_id": "q1", "value": 1})
            final = checkpoint.finalize(lambda row: row["question_id"])
            final.chmod(0o644); final.write_text('{"question_id":"q1","value":2}\n')
            with self.assertRaises(CheckpointError): checkpoint.validate_final(lambda row: row["question_id"])


if __name__ == "__main__": unittest.main()
