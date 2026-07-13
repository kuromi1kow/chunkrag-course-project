"""Specification Sections 23--25: canonical bytes, schemas, immutable artifacts."""

import json
import tempfile
import unittest
from pathlib import Path

from chunkrag.mainstudy.artifacts import ArtifactError, ArtifactStore, validate_record_links
from chunkrag.mainstudy.canonical import (
    CanonicalizationError, canonical_json_bytes, canonical_json_hash, canonical_jsonl_bytes,
)
from chunkrag.mainstudy.constants import PROTOCOL_ID
from chunkrag.mainstudy.schemas import SchemaError, validate_record


class CanonicalTests(unittest.TestCase):
    def test_canonical_json_is_order_and_unicode_invariant(self) -> None:
        left = {"b": "e\u0301", "a": 1}
        right = {"a": 1, "b": "é"}
        self.assertEqual(canonical_json_bytes(left), canonical_json_bytes(right))
        self.assertEqual(canonical_json_hash(left), canonical_json_hash(right))

    def test_jsonl_sorts_and_rejects_duplicates(self) -> None:
        payload = canonical_jsonl_bytes([{"id": "b"}, {"id": "a"}], "id")
        self.assertEqual([json.loads(line)["id"] for line in payload.splitlines()], ["a", "b"])
        with self.assertRaises(CanonicalizationError):
            canonical_jsonl_bytes([{"id": "a"}, {"id": "a"}], "id")

    def test_nan_rejected(self) -> None:
        with self.assertRaises(CanonicalizationError):
            canonical_json_bytes({"x": float("nan")})

    def test_immutable_store_rejects_changed_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(Path(directory))
            store.initialize()
            record = {
                "schema_version": PROTOCOL_ID, "cluster_id": "c", "dataset": "d",
                "question_ids": ["q"], "size": 1,
            }
            store.write_jsonl("manifests/c.jsonl", [record], "cluster", "cluster_id")
            changed = {**record, "size": 2}
            with self.assertRaises(FileExistsError):
                store.write_jsonl("manifests/c.jsonl", [changed], "cluster", "cluster_id")

    def test_schema_rejects_unknown_field(self) -> None:
        record = {
            "schema_version": PROTOCOL_ID, "cluster_id": "c", "dataset": "d",
            "question_ids": [], "size": 0, "extra": True,
        }
        with self.assertRaises(SchemaError):
            validate_record("cluster", record)

    def test_record_links_reject_unknown_hash(self) -> None:
        upstream = [{"id": "a"}]
        from chunkrag.mainstudy.canonical import canonical_json_hash
        validate_record_links(upstream, [{"upstream_hash": canonical_json_hash(upstream[0])}])
        with self.assertRaises(ArtifactError):
            validate_record_links(upstream, [{"upstream_hash": "0" * 64}])


if __name__ == "__main__":
    unittest.main()
