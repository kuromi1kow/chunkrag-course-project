"""Phase 3.6 unit compliance tests for immutable identities and ordering."""

from copy import deepcopy
import os
import unittest

from chunkrag.mainstudy.canonical import canonical_json_hash, identifier_hash
from chunkrag.mainstudy.constants import EXPERIMENT_DEPENDENCIES, PROTOCOL_SHA256
from chunkrag.mainstudy.determinism import configure_determinism
from chunkrag.mainstudy.environment import require_canonical_runtime
from chunkrag.mainstudy.experiments import plan_experiment
from chunkrag.mainstudy.generation import build_generation_record
from chunkrag.mainstudy.packing import PackedContext
from chunkrag.mainstudy.prompts import prompt_template_hash
from chunkrag.mainstudy.protocol import ProtocolError, load_protocol_config, validate_protocol_config
from chunkrag.mainstudy.retrieval import build_retrieval_record
from chunkrag.mainstudy.schemas import SchemaError, validate_record
from chunkrag.mainstudy.validation import validate_statistical_primitives


class Phase36UnitTests(unittest.TestCase):
    def test_every_protocol_config_mutation_is_rejected(self) -> None:
        base = load_protocol_config()
        base.pop("config_sha256")
        mutations = [
            (("master_seed",), 999),
            (("models", "mistral", "revision"), "wrong"),
            (("generation", "max_new_tokens", "techqa"), 1),
            (("retrieval", "dense_weight"), 0.1),
        ]
        for path, value in mutations:
            candidate = deepcopy(base)
            cursor = candidate
            for key in path[:-1]: cursor = cursor[key]
            cursor[path[-1]] = value
            with self.subTest(path=path), self.assertRaises(ProtocolError):
                validate_protocol_config(candidate)

    def test_retrieval_id_uses_protocol_sha(self) -> None:
        record = build_retrieval_record(
            question_id="q", condition_id="fixed192", question_manifest_hash="a" * 64,
            corpus_manifest_hash="b" * 64, dense=[], sparse=[], fused=[], reranked=[],
            config_hash="c" * 64, upstream_hash="d" * 64, latency={}, memory={},
        )
        self.assertEqual(record["retrieval_id"], identifier_hash(PROTOCOL_SHA256, "q", "fixed192", "c" * 64))

    def test_generation_self_hash_and_prompt_hash_are_enforced(self) -> None:
        packed = PackedContext("", "", (), (), 1, 0, 0, None, ())
        question = {"question_id": "q", "dataset": "squad_v2", "question": "Q"}
        record = build_generation_record(
            question=question, condition_id="fixed192", control_seed=None,
            packing_id="matched-4096", budget=4096, packed=packed,
            model_repository="m", model_revision="r", model_snapshot_hash="a" * 64,
            retrieval_or_gold_hash="b" * 64, prompt_version_hash=prompt_template_hash("squad_v2"),
            raw_output="x", generated_tokens=1, stopping_reason="eos", latency={},
            attempt_history=[], hardware={},
        )
        payload = dict(record); declared = payload.pop("record_hash")
        self.assertEqual(declared, canonical_json_hash(payload))
        record["raw_output"] = "tampered"
        with self.assertRaises(SchemaError): validate_record("generation", record)

    def test_canonical_order_and_determinism_are_active(self) -> None:
        e4 = plan_experiment("E4")
        self.assertEqual(e4[0].condition_id, "human-package")
        self.assertTrue(all(item.condition_id.startswith("judge__") for item in e4[1:-1]))
        self.assertIn("E2", EXPERIMENT_DEPENDENCIES["E3"])
        self.assertIn("E4", EXPERIMENT_DEPENDENCIES["E5"])
        status = configure_determinism()
        self.assertEqual(os.environ["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")
        self.assertEqual(status["seed"], 8677)
        with self.assertRaises(ProtocolError):
            require_canonical_runtime({"hardware": {"gpus": [], "cuda_build": None}}, gpu_required=True)
        self.assertEqual(validate_statistical_primitives()["status"], "valid")


if __name__ == "__main__": unittest.main()
