#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Launcher for the Starscream SPX-vs-CPX attention benchmark.
#
# Usage:
#   ./run_starscream_bench.sh [spx|cpx]        # mode defaults to spx
#
# The HARDWARE compute-partition mode must already match the requested mode:
#   SPX -> `rocm-smi --setcomputepartition SPX`  (8 physical GPUs)
#   CPX -> `rocm-smi --setcomputepartition CPX`  (64 vGPUs / XCDs)
# This script does NOT switch the partition mode for you.
#
# Environment overrides (optional):
#   VLLM_SRC     path to the mounted vllm source   (default: /workspace/vllm)
#   SEQ_LENS     comma list of seq lengths          (default: 256,8192,131072)
#   BATCH_SIZES  comma list of batch sizes          (default: 1,8,32,64,128,256,512,1024)
#   Q_HEADS      total query heads                   (default: 128)
#   KV_HEADS     total kv heads                      (default: 128)
#   HEAD_SIZE    head dimension                      (default: 128)
#   CPX_SIZE     XCDs per physical GPU (cpx mode)    (default: 8)
#   WARMUP       warmup iters                        (default: 10)
#   ITERS        timed iters                         (default: 50)
#   CSV          output csv path                     (default: /workspace/bench_<mode>.csv)

set -euo pipefail

MODE="${1:-spx}"
MODE="$(echo "$MODE" | tr '[:upper:]' '[:lower:]')"

if [[ "$MODE" != "spx" && "$MODE" != "cpx" ]]; then
    echo "ERROR: mode must be 'spx' or 'cpx' (got '$MODE')" >&2
    exit 1
fi

VLLM_SRC="${VLLM_SRC:-/workspace/vllm}"
SEQ_LENS="${SEQ_LENS:-256,8192,131072}"
BATCH_SIZES="${BATCH_SIZES:-1,8,32,64,128,256,512,1024}"
Q_HEADS="${Q_HEADS:-128}"
KV_HEADS="${KV_HEADS:-128}"
HEAD_SIZE="${HEAD_SIZE:-128}"
CPX_SIZE="${CPX_SIZE:-8}"
WARMUP="${WARMUP:-10}"
ITERS="${ITERS:-50}"
CSV="${CSV:-/workspace/bench_${MODE}.csv}"

BENCH="${VLLM_SRC}/tests/kernels/attention/bench_starscream_attention.py"

COMMON_ARGS=(
    --total-q-heads "$Q_HEADS"
    --total-kv-heads "$KV_HEADS"
    --head-size "$HEAD_SIZE"
    --seq-lens "$SEQ_LENS"
    --batch-sizes "$BATCH_SIZES"
    --warmup "$WARMUP"
    --iters "$ITERS"
    --csv "$CSV"
)

echo "=================================================================="
echo " Starscream attention benchmark"
echo "   mode        : $MODE"
echo "   vllm src    : $VLLM_SRC"
echo "   seq_lens    : $SEQ_LENS"
echo "   batch_sizes : $BATCH_SIZES"
echo "   q/kv heads  : $Q_HEADS / $KV_HEADS   head_size: $HEAD_SIZE"
echo "   warmup/iters: $WARMUP / $ITERS"
echo "   csv         : $CSV"
echo "=================================================================="
echo " Reminder: hardware must be in $MODE partition mode already."
echo "   rocm-smi --setcomputepartition ${MODE^^}"
echo "=================================================================="

if [[ "$MODE" == "spx" ]]; then
    PYTHONPATH="$VLLM_SRC" \
    torchrun --nnodes=1 --nproc-per-node=8 \
        "$BENCH" --mode spx "${COMMON_ARGS[@]}"
else
    PYTHONPATH="$VLLM_SRC" \
    TORCH_SYMM_MEM_DISABLE_MULTICAST=1 \
    torchrun --nnodes=1 --nproc-per-node=64 \
        "$BENCH" --mode cpx --cpx-size "$CPX_SIZE" "${COMMON_ARGS[@]}"
fi
