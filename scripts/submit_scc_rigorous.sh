#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${1:-$REPO_ROOT/configs/scc_rigorous_qwen.json}"
OUTPUT_DIR="${2:-$REPO_ROOT/outputs/scc_rigorous_qwen}"

mkdir -p "$REPO_ROOT/logs"

PROJECT="${SCC_PROJECT:-cs505am}"
QUEUE="${SCC_QUEUE:-academic-gpu}"
GPU_COUNT="${SCC_GPU_COUNT:-1}"
GPU_TYPE="${SCC_GPU_TYPE:-A100}"
GPU_MEMORY="${SCC_GPU_MEMORY:-80G}"
GPU_CAPABILITY="${SCC_GPU_CAPABILITY:-7.0}"
THREADS="${SCC_THREADS:-8}"
MEMORY="${SCC_MEMORY:-64G}"
WALLTIME="${SCC_WALLTIME:-24:00:00}"
JOB_NAME="${SCC_JOB_NAME:-chunkrag-rigorous}"

qsub \
  -P "$PROJECT" \
  -q "$QUEUE" \
  -N "$JOB_NAME" \
  -o "$REPO_ROOT/logs" \
  -V \
  -pe omp "$THREADS" \
  -l "h_rt=$WALLTIME,mem_total=$MEMORY,gpus=$GPU_COUNT,gpu_type=$GPU_TYPE,gpu_memory=$GPU_MEMORY,gpu_c=$GPU_CAPABILITY" \
  "$REPO_ROOT/scripts/chunkrag_scc.qsub" \
  "$CONFIG_PATH" \
  "$OUTPUT_DIR"
