"""Specification Sections 1, 26--30: authority, counts, DAG, and output mapping."""

import unittest

from chunkrag.mainstudy.constants import EXPECTED_E2_RECORDS, EXPECTED_E3_RECORDS, EXPERIMENT_ORDER
from chunkrag.mainstudy.coverage import validate_coverage
from chunkrag.mainstudy.experiments import condition_ids_e1, condition_ids_e2, full_plan, plan_experiment
from chunkrag.mainstudy.paper import output_assignment_manifest, validate_output_assignments
from chunkrag.mainstudy.protocol import load_protocol_config, verify_protocol


class ProtocolPlanTests(unittest.TestCase):
    def test_frozen_protocol_and_config(self) -> None:
        self.assertEqual(len(verify_protocol()), 64)
        config = load_protocol_config()
        self.assertEqual(config["experiment_order"], list(EXPERIMENT_ORDER))

    def test_condition_counts(self) -> None:
        self.assertEqual(len(condition_ids_e1()), 24)
        self.assertEqual(len(condition_ids_e2()), 31)
        e2 = plan_experiment("E2")
        counts = {dataset: sum(item.expected_records for item in e2 if item.dataset == dataset) for dataset in EXPECTED_E2_RECORDS}
        self.assertEqual(counts, EXPECTED_E2_RECORDS)
        e3 = plan_experiment("E3")
        counts3 = {dataset: sum(item.expected_records for item in e3 if item.dataset == dataset) for dataset in EXPECTED_E3_RECORDS}
        self.assertEqual(counts3, EXPECTED_E3_RECORDS)

    def test_full_plan_covers_all_experiments(self) -> None:
        self.assertEqual(sorted(set(item.experiment for item in full_plan())), list(EXPERIMENT_ORDER))

    def test_paper_assignment(self) -> None:
        validate_output_assignments()
        manifest = output_assignment_manifest()
        self.assertEqual(len(manifest["main_figures"]), 3)
        self.assertEqual(len(manifest["main_tables"]), 3)

    def test_protocol_coverage_registry(self) -> None:
        self.assertEqual(validate_coverage()["missing"], [])


if __name__ == "__main__":
    unittest.main()
