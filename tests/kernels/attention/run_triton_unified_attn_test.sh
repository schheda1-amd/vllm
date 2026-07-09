#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Correctness test for unified_attention in triton_unified_attention.py.
#
# This wraps the existing pytest suite
#   tests/kernels/attention/test_triton_unified_attention.py
# which compares unified_attention() against a reference paged-attention
# implementation across a parametrized sweep (num_heads, head_size, block_size,
# sliding_window, dtype, soft_cap, num_blocks, fp8). It is single-GPU and does
# NOT exercise the Starscream cross-XCD path -- it validates the base attention
# kernel. No torchrun / distributed setup needed.
#
# Usage:
#   ./run_triton_unified_attn_test.sh [extra pytest args...]
#
# Environment overrides (optional):
#   VLLM_SRC   path to the mounted vllm source   (default: /workspace/vllm)
#   GPU        CUDA device to run on             (default: 0)

set -euo pipefail

VLLM_SRC="${VLLM_SRC:-/workspace/vllm}"
GPU="${GPU:-0}"
TEST="${VLLM_SRC}/tests/kernels/attention/test_triton_unified_attention.py"

echo "=================================================================="
echo " unified_attention correctness test"
echo "   vllm src : $VLLM_SRC"
echo "   gpu      : $GPU"
echo "   test     : $TEST"
echo "=================================================================="

CUDA_VISIBLE_DEVICES="$GPU" \
PYTHONPATH="$VLLM_SRC" \
    python -m pytest -v "$TEST" "$@"
