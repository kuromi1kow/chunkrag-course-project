from __future__ import annotations

RUN_ID = "techqa_adaptive_v1"
PROTOCOL_COMMIT = "5df8cc7bf5f26a1b3cde37df2023cd6590352ad2"
BASELINE_TREE_SHA256 = "7572ec911f852c9d420c6728f74fedbfbc88f2652e2df4b9b4963a363fda9ac6"
PARTITION_SALT = "eaai-phase2-techqa-v1"

EXPECTED_TOTAL_ROWS = 910
EXPECTED_ELIGIBLE_ROWS = 608
EXPECTED_DOCUMENTS = 496
DEVELOPMENT_SIZE = 200
HELDOUT_SIZE = 200
RESERVE_SIZE = 208

CHUNKERS = ("fixed_128", "fixed_254", "recursive_254", "sentence_254")
CONDITIONS = ("hybrid", "reranked")
PRIMARY_GENERATOR = "qwen"
SECONDARY_GENERATOR = "mistral"

NUMERIC_FEATURES = (
    "query_token_count",
    "dense_bm25_jaccard_at_20",
    "dense_bm25_jaccard_at_4",
    "fused_top1_score",
    "fused_top1_top2_margin",
    "fused_top4_top5_margin",
    "fused_score_entropy",
    "fused_top4_mean_dense_rank",
    "fused_top4_mean_bm25_rank",
    "fused_top4_mean_query_overlap",
    "fused_top4_max_query_overlap",
    "fused_top4_mean_chunk_tokens",
    "fused_top4_sd_chunk_tokens",
)

PRIMARY_BOOTSTRAP_DRAWS = 20_000
PRIMARY_BOOTSTRAP_SEED = 20_260_809
PRIMARY_RANDOMIZATION_DRAWS = 100_000
PRIMARY_RANDOMIZATION_SEED = 20_260_810
