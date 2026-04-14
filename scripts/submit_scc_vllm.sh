#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${1:-$REPO_ROOT/outputs/openwebui_vllm}"

mkdir -p "$REPO_ROOT/logs"

PROJECT="${SCC_PROJECT:-cs505am}"
QUEUE="${SCC_QUEUE:-academic-gpu}"
GPU_COUNT="${SCC_GPU_COUNT:-1}"
GPU_TYPE="${SCC_GPU_TYPE:-A40}"
GPU_MEMORY="${SCC_GPU_MEMORY:-48G}"
GPU_CAPABILITY="${SCC_GPU_CAPABILITY:-7.0}"
THREADS="${SCC_THREADS:-8}"
MEMORY="${SCC_MEMORY:-48G}"
WALLTIME="${SCC_WALLTIME:-12:00:00}"
JOB_NAME="${SCC_JOB_NAME:-chunkrag-vllm}"

qsub \
  -P "$PROJECT" \
  -q "$QUEUE" \
  -N "$JOB_NAME" \
  -o "$REPO_ROOT/logs" \
  -V \
  -pe omp "$THREADS" \
  -l "h_rt=$WALLTIME,mem_total=$MEMORY,gpus=$GPU_COUNT,gpu_type=$GPU_TYPE,gpu_memory=$GPU_MEMORY,gpu_c=$GPU_CAPABILITY" \
  "$REPO_ROOT/scripts/vllm_scc.qsub" \
  "$OUTPUT_DIR"
