"""Adversarial provenance, environment, completion, and merge tests."""

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from chunkrag.mainstudy.analysis import AnalysisGateError, require_completed_experiments
from chunkrag.mainstudy.canonical import file_sha256, source_sha256
from chunkrag.mainstudy.checkpoint import CheckpointError, merge_shards
from chunkrag.mainstudy.constants import EXPERIMENT_ORDER, PROTOCOL_ID, PROTOCOL_SHA256
from chunkrag.mainstudy.environment import environment_manifest, freeze_transitive_environment
from chunkrag.mainstudy.protocol import ProtocolError


class Phase36AdversarialTests(unittest.TestCase):
    def test_source_hash_uses_only_tracked_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.email", "a@b.c"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.name", "t"], cwd=root, check=True)
            (root / "src").mkdir(); (root / "src" / "a.py").write_text("x=1\n")
            lock = root / "lock.json"; lock.write_text("{}\n")
            subprocess.run(["git", "add", "src/a.py", "lock.json"], cwd=root, check=True)
            subprocess.run(["git", "commit", "-qm", "x"], cwd=root, check=True)
            first = source_sha256(root, lock)
            (root / "src" / "__pycache__").mkdir(); (root / "src" / "__pycache__" / "a.pyc").write_bytes(b"ignored")
            self.assertEqual(first, source_sha256(root, lock))
            (root / "src" / "a.py").write_text("x=2\n")
            self.assertNotEqual(first, source_sha256(root, lock))

    def test_environment_lock_rejects_package_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); direct = root / "requirements-main-study.lock"; direct.write_text("x\n")
            resolved = root / "requirements-main-study.transitive.json"
            with patch("chunkrag.mainstudy.environment.verify_direct_versions", return_value={}):
                freeze_transitive_environment(resolved, direct)
                environment_manifest(resolved, check_installed=True)
                payload = json.loads(resolved.read_text()); payload["packages"] = []
                resolved.write_text(json.dumps(payload))
                with self.assertRaises(ProtocolError): environment_manifest(resolved, check_installed=True)

    def test_merge_rejects_mixed_environments(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, env in enumerate(("a" * 64, "b" * 64)):
                data = root / f"part-{index:03d}.jsonl"; data.write_text(json.dumps({"question_id": f"q{index}"}) + "\n")
                state = {"schema_version": PROTOCOL_ID, "stage": "E2", "dataset": "d", "condition_id": "c", "shard_index": index, "expected_question_ids": [f"q{index}"], "completed": [f"q{index}"], "record_hashes": {f"q{index}": __import__('chunkrag.mainstudy.canonical', fromlist=['canonical_json_hash']).canonical_json_hash({"question_id": f"q{index}"})}, "protocol_sha256": PROTOCOL_SHA256, "config_sha256": "c" * 64, "environment_hash": env}
                (root / f"part-{index:03d}.state.json").write_text(json.dumps(state))
            with self.assertRaises(CheckpointError):
                merge_shards(sorted(root.glob("part-*.jsonl")), ["q0", "q1"], "question_id", root / "out.jsonl", require_state=True)

    def test_fabricated_or_writable_completion_cannot_unlock_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / "audit").mkdir(); locked = root / "raw.json"; locked.write_text("{}\n")
            stages = []
            for experiment in EXPERIMENT_ORDER:
                path = root / "audit" / f"{experiment}.json"; path.write_text("{}\n")
                stages.append({"experiment": experiment, "path": f"audit/{experiment}.json", "sha256": file_sha256(path)})
            completion_path = root / "audit" / "completion.json"
            completion = {"protocol_sha256": PROTOCOL_SHA256, "git_commit": "g", "config_sha256": "c" * 64, "environment_hash": "e" * 64, "completed_experiments": list(EXPERIMENT_ORDER), "artifacts_locked_read_only": True, "stage_markers": stages, "artifacts": [{"path": "raw.json", "sha256": file_sha256(locked), "bytes": locked.stat().st_size}]}
            completion_path.write_text(json.dumps(completion))
            with self.assertRaises(AnalysisGateError):
                require_completed_experiments(completion, artifact_root=root, completion_path=completion_path)
            locked.chmod(0o444)
            completion_path.chmod(0o444)
            for stage in stages: (root / stage["path"]).chmod(0o444)
            with self.assertRaises(AnalysisGateError):
                require_completed_experiments(completion, artifact_root=root, completion_path=completion_path)
            with self.assertRaises(AnalysisGateError):
                require_completed_experiments(completion, artifact_root=root, completion_path=root / "fake.json")


if __name__ == "__main__": unittest.main()
