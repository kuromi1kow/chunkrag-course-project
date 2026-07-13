"""Frozen identifiers and experiment registry (Specification Sections 7, 23, 26--30)."""

from __future__ import annotations

PROTOCOL_ID = "chunkrag-main-v1"
PROTOCOL_SHA256 = "567b652fc403e7ff7e00e349de86357f9a293cac77e7e7f4d3612284eb2c89bf"
SCHEMA_VERSION = PROTOCOL_ID
MASTER_SEED = 8677

DATASET_ORDER = ("squad_v2", "hotpot_qa", "techqa")
POLICY_ORDER = ("fixed192", "recursive192", "sentence192", "semantic192")
STRUCTURED_POLICIES = POLICY_ORDER[1:]
JITTER_SEEDS = (1103, 2207, 3301, 4409, 5519)

PACKING_IDS = (
    "operational-1024",
    "operational-4096",
    "matched-1024",
    "matched-4096",
    "gold-1024",
    "gold-4096",
)

EXPERIMENT_ORDER = ("E0", "E1", "E2", "E3", "E4", "E5", "E6", "E7")
EXPERIMENT_DEPENDENCIES = {
    "E0": (),
    "E1": ("E0",),
    "E2": ("E1",),
    "E3": ("E0",),
    "E4": ("E2", "E3"),
    "E5": ("E1",),
    "E6": ("E1", "E3"),
    "E7": ("E0", "E1", "E2", "E3", "E4", "E5", "E6"),
}

EXPECTED_QUESTION_COUNTS = {"squad_v2": 500, "hotpot_qa": 500, "techqa": 300}
EXPECTED_E2_CONDITIONS = 31
EXPECTED_E2_RECORDS = {name: count * 31 for name, count in EXPECTED_QUESTION_COUNTS.items()}
EXPECTED_E3_RECORDS = {name: count * 2 for name, count in EXPECTED_QUESTION_COUNTS.items()}
EXPECTED_E4_JUDGE_RECORDS = 9_900
EXPECTED_E6_RECORDS = {"squad_v2": 2_500, "hotpot_qa": 2_500}

ARTIFACT_SUBDIRECTORIES = (
    "manifests",
    "chunks",
    "retrieval",
    "generation",
    "evaluation",
    "analysis",
    "audit",
)

MAIN_FIGURES = {
    "figure1": ("E1", "E2", "E3"),
    "figure2": ("E2",),
    "figure3": ("E2", "E4"),
}
MAIN_TABLES = {
    "table1": ("E0",),
    "table2": ("E2",),
    "table3": ("E3", "E4"),
}

HASH_RE = r"^[0-9a-f]{64}$"
