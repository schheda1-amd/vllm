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
# Output-token offsets added to each base seq-len (KV grows as tokens generate).
# Default 0 = single snapshot. E.g. TOKEN_OFFSETS=0,64,128,192,256 samples the
# first 257 output tokens and reports per-output-token attention latency.
TOKEN_OFFSETS="${TOKEN_OFFSETS:-0}"
BATCH_SIZES="${BATCH_SIZES:-1,8,32,64,128,256,512,1024}"
# Llama3-70B under TP=8, single-GPU focus: per-GPU shard is 8 q-heads,
# 1 kv-head (GQA 64/8 q, 8/8 kv), head_size 128.
Q_HEADS="${Q_HEADS:-8}"
KV_HEADS="${KV_HEADS:-1}"
HEAD_SIZE="${HEAD_SIZE:-128}"
CPX_SIZE="${CPX_SIZE:-8}"
WARMUP="${WARMUP:-25}"
ITERS="${ITERS:-200}"
CSV="${CSV:-/workspace/bench_${MODE}.csv}"
# On-device signal-pad barrier (1) vs host dist.barrier (0). Only affects CPX.
# NOTE: CUDA-graph timing needs SIGNAL_PAD=1 for step1/step2 to be capturable
# (a host dist.barrier is not graph-capturable; those variants fall back to
# eager otherwise). rccl is unaffected.
SIGNAL_PAD="${SIGNAL_PAD:-0}"
# CUDA-graph timing: 1 = graph replay (default, removes launch overhead),
# 0 = eager (includes launch overhead).
CUDA_GRAPH="${CUDA_GRAPH:-1}"
# Which physical GPU's devices to pin. SPX: one GPU index (default 0).
# CPX: that GPU's 8 XCD indices (default 0-7). Override for a different GPU.
if [[ "$MODE" == "spx" ]]; then
    DEVICES="${DEVICES:-0}"
    NPROC=1
else
    DEVICES="${DEVICES:-0,1,2,3,4,5,6,7}"
    NPROC="$CPX_SIZE"
fi

BENCH="${VLLM_SRC}/tests/kernels/attention/bench_starscream_attention.py"

COMMON_ARGS=(
    --total-q-heads "$Q_HEADS"
    --total-kv-heads "$KV_HEADS"
    --head-size "$HEAD_SIZE"
    --seq-lens "$SEQ_LENS"
    --token-offsets "$TOKEN_OFFSETS"
    --batch-sizes "$BATCH_SIZES"
    --warmup "$WARMUP"
    --iters "$ITERS"
    --csv "$CSV"
)
if [[ "$CUDA_GRAPH" == "1" ]]; then
    COMMON_ARGS+=(--cuda-graph)
else
    COMMON_ARGS+=(--no-cuda-graph)
fi

echo "=================================================================="
echo " Starscream attention benchmark (single physical GPU)"
echo "   mode         : $MODE"
echo "   vllm src     : $VLLM_SRC"
echo "   devices      : $DEVICES   (nproc=$NPROC)"
echo "   seq_lens     : $SEQ_LENS"
echo "   token_offsets: $TOKEN_OFFSETS"
echo "   batch_sizes  : $BATCH_SIZES"
echo "   q/kv heads   : $Q_HEADS / $KV_HEADS   head_size: $HEAD_SIZE"
echo "   warmup/iters : $WARMUP / $ITERS"
echo "   cuda_graph   : $CUDA_GRAPH   (1=graph replay, 0=eager)"
echo "   signal_pad   : $SIGNAL_PAD   (CPX only)"
echo "   csv          : $CSV"
echo "=================================================================="
echo " Reminder: hardware must be in $MODE partition mode already."
echo "   rocm-smi --setcomputepartition ${MODE^^}"
echo "=================================================================="

if [[ "$MODE" == "spx" ]]; then
    CUDA_VISIBLE_DEVICES="$DEVICES" \
    PYTHONPATH="$VLLM_SRC" \
    torchrun --nnodes=1 --nproc-per-node="$NPROC" \
        "$BENCH" --mode spx "${COMMON_ARGS[@]}"
else
    CUDA_VISIBLE_DEVICES="$DEVICES" \
    PYTHONPATH="$VLLM_SRC" \
    TORCH_SYMM_MEM_DISABLE_MULTICAST=1 \
    VLLM_STARSCREAM_SIGNAL_PAD_BARRIER="$SIGNAL_PAD" \
    torchrun --nnodes=1 --nproc-per-node="$NPROC" \
        "$BENCH" --mode cpx --cpx-size "$CPX_SIZE" "${COMMON_ARGS[@]}"
fi
