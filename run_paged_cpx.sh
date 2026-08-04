#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# HIP paged-attention decode benchmark: SPX baseline vs CPX + Starscream.
#
# Usage:
#   ./run_paged_cpx.sh spx            # baseline, hardware in SPX
#   ./run_paged_cpx.sh cpx [--check]  # Starscream, hardware in CPX
#
# The hardware partition must already match (this script does NOT switch it):
#   spx -> rocm-smi --setcomputepartition SPX
#   cpx -> rocm-smi --setcomputepartition CPX
#
# Pass --num-blocks the SAME on both sides or the comparison is not
# apples-to-apples (different KV scatter -> different TLB/cache behaviour):
#   ./run_paged_cpx.sh spx --num-blocks 131072
#   ./run_paged_cpx.sh cpx --num-blocks 131072

set -euo pipefail

MODE="${1:-cpx}"; shift || true
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CPX_SIZE="${CPX_SIZE:-8}"

case "$MODE" in
  spx)
    echo "== SPX baseline (paged_attention_rocm, 1 rank owns the whole GPU) =="
    CUDA_VISIBLE_DEVICES="${DEVICES:-0}" PYTHONPATH="$SCRIPT_DIR" \
      python3 "$SCRIPT_DIR/test_harness_1_paged.py" "$@"
    ;;
  cpx)
    echo "== CPX + Starscream ($CPX_SIZE XCDs, cross-XCD softmax merge) =="
    echo "   Reminder: rocm-smi --setcomputepartition CPX"
    # ROCm symmetric memory: the multicast path is typically unavailable, so
    # force the peer-pointer path the fused merge kernel actually uses.
    export TORCH_SYMM_MEM_DISABLE_MULTICAST="${TORCH_SYMM_MEM_DISABLE_MULTICAST:-1}"
    CUDA_VISIBLE_DEVICES="${DEVICES:-0,1,2,3,4,5,6,7}" PYTHONPATH="$SCRIPT_DIR" \
      torchrun --nnodes=1 --nproc-per-node="$CPX_SIZE" \
        "$SCRIPT_DIR/test_harness_1_paged_cpx.py" "$@"
    ;;
  *)
    echo "ERROR: mode must be 'spx' or 'cpx' (got '$MODE')" >&2
    exit 1
    ;;
esac
