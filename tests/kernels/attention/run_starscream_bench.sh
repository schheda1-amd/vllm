#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Launcher for the Starscream SPX-vs-CPX attention benchmark.
#
# Usage:
#   ./run_starscream_bench.sh [spx|cpx|cpx-baseline] [model]  # mode defaults spx
#     model (optional): llama3-70b (default) | llama3-405b | gpt-oss-120b
#       Sets the per-GPU TP=8 attention shard:
#         llama3-70b   -> 8 q  / 1 kv / head_size 128
#         llama3-405b  -> 16 q / 1 kv / head_size 128
#         gpt-oss-120b -> 8 q  / 1 kv / head_size 64
#       An explicit Q_HEADS/KV_HEADS/HEAD_SIZE env still overrides everything.
#
# The HARDWARE compute-partition mode must already match the requested mode:
#   SPX -> `rocm-smi --setcomputepartition SPX`  (single physical GPU, 1 rank)
#   CPX -> `rocm-smi --setcomputepartition CPX`  (that GPU's 8 XCDs)
# This script does NOT switch the partition mode for you.
#
# Environment overrides (optional):
#   VLLM_SRC     path to the mounted vllm source   (default: /workspace/vllm)
#   MODEL        llama3-70b | llama3-405b           (default: llama3-70b; or arg 2)
#   SEQ_LENS     comma list of seq lengths          (default: 256,8192,131072)
#   TOKEN_OFFSETS output-token offsets per base S    (default: 0)
#   BATCH_SIZES  comma list of batch sizes          (default: 1,8,32,64,128,256,512,1024)
#   Q_HEADS/KV_HEADS/HEAD_SIZE  override the model preset (default: unset)
#   CPX_SIZE     XCDs per physical GPU (cpx mode)    (default: 8)
#   SIGNAL_PAD   1 = on-device signal-pad barrier    (default: 0)
#   WARMUP       warmup iters                        (default: 25)
#   ITERS        timed iters                         (default: 200)
#   CSV          output csv path                     (default: /workspace/bench_<mode>.csv)

set -euo pipefail

MODE="${1:-spx}"
MODE="$(echo "$MODE" | tr '[:upper:]' '[:lower:]')"

# Three modes:
#   spx          : hardware in SPX; 1 rank owns the whole physical GPU. Baseline.
#   cpx          : hardware in CPX; 8 XCDs run the Starscream path (rccl/step1/
#                  step2), context split + merge. The proposed design.
#   cpx-baseline : hardware in CPX; 8 XCDs run the ORIGINAL vLLM path with
#                  starscream DISABLED -- i.e. what you get by simply switching a
#                  node to CPX and running unchanged code. Each XCD independently
#                  does full-context attention over a replicated KV head. This is
#                  the "CPX without Starscream" reference. It maps to the
#                  benchmark's --mode spx (enable_starscream=False) but launched
#                  on all 8 XCDs, so the per-XCD shard is 1 q-head / 1 kv-head.
if [[ "$MODE" != "spx" && "$MODE" != "cpx" && "$MODE" != "cpx-baseline" ]]; then
    echo "ERROR: mode must be 'spx', 'cpx', or 'cpx-baseline' (got '$MODE')" >&2
    exit 1
fi

# Model preset: arg 2 (if given) else $MODEL else llama3-70b.
MODEL="${2:-${MODEL:-llama3-70b}}"
MODEL="$(echo "$MODEL" | tr '[:upper:]' '[:lower:]')"
if [[ "$MODEL" != "llama3-70b" && "$MODEL" != "llama3-405b" \
      && "$MODEL" != "gpt-oss-120b" ]]; then
    echo "ERROR: model must be 'llama3-70b', 'llama3-405b', or 'gpt-oss-120b'"\
         "(got '$MODEL')" >&2
    exit 1
fi

VLLM_SRC="${VLLM_SRC:-/workspace/vllm}"
SEQ_LENS="${SEQ_LENS:-256,8192,131072}"
# Output-token offsets added to each base seq-len (KV grows as tokens generate).
# Default 0 = single snapshot. E.g. TOKEN_OFFSETS=0,64,128,192,256 samples the
# first 257 output tokens and reports per-output-token attention latency.
TOKEN_OFFSETS="${TOKEN_OFFSETS:-0}"
BATCH_SIZES="${BATCH_SIZES:-1,8,32,64,128,256,512,1024}"
# Head shapes come from the model preset (--model), applied by the benchmark.
# Q_HEADS / KV_HEADS / HEAD_SIZE are UNSET by default so the preset wins; set
# any of them to override the preset for that dimension.
CPX_SIZE="${CPX_SIZE:-8}"
WARMUP="${WARMUP:-25}"
ITERS="${ITERS:-200}"
CSV="${CSV:-/workspace/vllm/bench_${MODEL}_${MODE}.csv}"
# On-device signal-pad barrier (1) vs host dist.barrier (0). Only affects CPX.
# In eager mode both are valid; SIGNAL_PAD=1 keeps RCCL off the merge path
# (the point of step1/step2). Set SIGNAL_PAD=1 for the intended symm-mem runs.
SIGNAL_PAD="${SIGNAL_PAD:-0}"
# CUDA-graph timing: 1 = graph replay (default), 0 = eager.
CUDA_GRAPH="${CUDA_GRAPH:-1}"
# Map launcher mode -> devices, rank count, and the benchmark's --mode.
#   spx          : 1 rank, whole GPU, benchmark --mode spx (starscream off)
#   cpx          : 8 ranks (XCDs), benchmark --mode cpx  (starscream on)
#   cpx-baseline : 8 ranks (XCDs), benchmark --mode spx  (starscream OFF) --
#                  original code on CPX hardware, no context split/merge.
if [[ "$MODE" == "spx" ]]; then
    DEVICES="${DEVICES:-0}"
    NPROC=1
    BENCH_MODE="spx"
elif [[ "$MODE" == "cpx" ]]; then
    DEVICES="${DEVICES:-0,1,2,3,4,5,6,7}"
    NPROC="$CPX_SIZE"
    BENCH_MODE="cpx"
else  # cpx-baseline
    DEVICES="${DEVICES:-0,1,2,3,4,5,6,7}"
    NPROC="$CPX_SIZE"
    BENCH_MODE="spx"
fi

BENCH="${VLLM_SRC}/tests/kernels/attention/bench_starscream_attention.py"

COMMON_ARGS=(
    --model "$MODEL"
    --seq-lens "$SEQ_LENS"
    --token-offsets "$TOKEN_OFFSETS"
    --batch-sizes "$BATCH_SIZES"
    --warmup "$WARMUP"
    --iters "$ITERS"
    --csv "$CSV"
)
# Optional per-dimension overrides: only passed if the env var is set, so they
# take precedence over the --model preset (the benchmark applies --model first).
[[ -n "${Q_HEADS:-}" ]]   && COMMON_ARGS+=(--total-q-heads "$Q_HEADS")
[[ -n "${KV_HEADS:-}" ]]  && COMMON_ARGS+=(--total-kv-heads "$KV_HEADS")
[[ -n "${HEAD_SIZE:-}" ]] && COMMON_ARGS+=(--head-size "$HEAD_SIZE")
if [[ "$CUDA_GRAPH" == "1" ]]; then
    COMMON_ARGS+=(--cuda-graph)
else
    COMMON_ARGS+=(--no-cuda-graph)
fi

echo "=================================================================="
echo " Starscream attention benchmark (single physical GPU)"
echo "   mode         : $MODE"
echo "   model        : $MODEL"
echo "   vllm src     : $VLLM_SRC"
echo "   devices      : $DEVICES   (nproc=$NPROC)"
echo "   seq_lens     : $SEQ_LENS"
echo "   token_offsets: $TOKEN_OFFSETS"
echo "   batch_sizes  : $BATCH_SIZES"
echo "   head override: q=${Q_HEADS:-preset} kv=${KV_HEADS:-preset} head_size=${HEAD_SIZE:-preset}"
echo "   warmup/iters : $WARMUP / $ITERS"
echo "   cuda_graph   : $CUDA_GRAPH   (1=graph replay, 0=eager)"
echo "   signal_pad   : $SIGNAL_PAD   (CPX only)"
echo "   csv          : $CSV"
# Required hardware partition per launcher mode: spx -> SPX, cpx & cpx-baseline
# -> CPX (both need the 8 XCDs visible).
if [[ "$MODE" == "spx" ]]; then HW_PART="SPX"; else HW_PART="CPX"; fi
echo "=================================================================="
echo " Reminder: hardware must be in $HW_PART partition mode already."
echo "   rocm-smi --setcomputepartition $HW_PART"
echo "=================================================================="

if [[ "$BENCH_MODE" == "cpx" ]]; then
    # Starscream path: symm-mem env + cpx-size.
    CUDA_VISIBLE_DEVICES="$DEVICES" \
    PYTHONPATH="$VLLM_SRC" \
    TORCH_SYMM_MEM_DISABLE_MULTICAST=1 \
    VLLM_STARSCREAM_SIGNAL_PAD_BARRIER="$SIGNAL_PAD" \
    torchrun --nnodes=1 --nproc-per-node="$NPROC" \
        "$BENCH" --mode cpx --cpx-size "$CPX_SIZE" "${COMMON_ARGS[@]}"
else
    # SPX code path (starscream off). Used by both `spx` (1 rank) and
    # `cpx-baseline` (8 XCDs). No symm-mem env, no cpx-size.
    CUDA_VISIBLE_DEVICES="$DEVICES" \
    PYTHONPATH="$VLLM_SRC" \
    torchrun --nnodes=1 --nproc-per-node="$NPROC" \
        "$BENCH" --mode spx "${COMMON_ARGS[@]}"
fi
