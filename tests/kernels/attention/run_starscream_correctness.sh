#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Correctness test for the Starscream symm-mem allgather + fused reduce path.
# Wraps tests/kernels/attention/test_starscream_symm_reduce.py, which compares
# step1 (symm-mem allgather -> reduce_segments) and step2 (fused kernel) against
# the RCCL baseline (get_dcp_group().all_gather -> reduce_segments).
#
# Runs on ONE physical GPU's XCDs (default 8), matching the CPX+NPS4 topology.
#
# Usage:
#   ./run_starscream_correctness.sh            # host dist.barrier path
#   SIGNAL_PAD=1 ./run_starscream_correctness.sh   # on-device signal-pad barrier
#
# Environment overrides (optional):
#   VLLM_SRC     path to mounted vllm source   (default: /workspace/vllm)
#   CPX_SIZE     XCDs per physical GPU          (default: 8)
#   DEVICES      CUDA_VISIBLE_DEVICES to pin    (default: 0,1,..,CPX_SIZE-1)
#   SIGNAL_PAD   1 = on-device signal-pad barrier, 0 = host dist.barrier (default 0)
#   MODE         step1 | step2 | all            (default: all)
#   NUM_TOKENS   decode batch (T)               (default: 32)
#   NUM_HEADS    query heads (div by CPX_SIZE)  (default: 64)
#   HEAD_SIZE    head dimension                 (default: 128)
#   SEQ_LEN      context length                 (default: 4096)
#   TRIALS       randomized trial count         (default: 8)
#   RTOL / ATOL  match tolerances              (default: 1e-3)

set -euo pipefail

VLLM_SRC="${VLLM_SRC:-/workspace/vllm}"
CPX_SIZE="${CPX_SIZE:-8}"
SIGNAL_PAD="${SIGNAL_PAD:-0}"
MODE="${MODE:-all}"
NUM_TOKENS="${NUM_TOKENS:-32}"
NUM_HEADS="${NUM_HEADS:-64}"
HEAD_SIZE="${HEAD_SIZE:-128}"
SEQ_LEN="${SEQ_LEN:-4096}"
TRIALS="${TRIALS:-8}"
RTOL="${RTOL:-1e-3}"
ATOL="${ATOL:-1e-3}"

# Default device list = 0..CPX_SIZE-1 (one physical GPU's XCDs).
if [[ -z "${DEVICES:-}" ]]; then
    DEVICES="$(seq -s, 0 $((CPX_SIZE - 1)))"
fi

TEST="${VLLM_SRC}/tests/kernels/attention/test_starscream_symm_reduce.py"

echo "=================================================================="
echo " Starscream correctness test (symm-mem allgather + fused reduce)"
echo "   vllm src   : $VLLM_SRC"
echo "   devices    : $DEVICES   (nproc=$CPX_SIZE)"
echo "   signal_pad : $SIGNAL_PAD   (0=dist.barrier, 1=on-device)"
echo "   mode       : $MODE"
echo "   shape      : tokens=$NUM_TOKENS heads=$NUM_HEADS head_size=$HEAD_SIZE seq_len=$SEQ_LEN"
echo "   trials     : $TRIALS   tol: rtol=$RTOL atol=$ATOL"
echo "=================================================================="

CUDA_VISIBLE_DEVICES="$DEVICES" \
PYTHONPATH="$VLLM_SRC" \
TORCH_SYMM_MEM_DISABLE_MULTICAST=1 \
VLLM_STARSCREAM_SIGNAL_PAD_BARRIER="$SIGNAL_PAD" \
torchrun --nnodes=1 --nproc-per-node="$CPX_SIZE" \
    "$TEST" \
    --mode "$MODE" \
    --cpx-size "$CPX_SIZE" \
    --num-tokens "$NUM_TOKENS" \
    --num-heads "$NUM_HEADS" \
    --head-size "$HEAD_SIZE" \
    --seq-len "$SEQ_LEN" \
    --trials "$TRIALS" \
    --rtol "$RTOL" \
    --atol "$ATOL"
