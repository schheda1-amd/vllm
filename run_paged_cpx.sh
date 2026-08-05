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
# --num-blocks is PER RANK on the cpx side, so divide by the world size to keep
# the KV footprint on the PHYSICAL GPU the same:
#   ./run_paged_cpx.sh spx --num-blocks 131072
#   ./run_paged_cpx.sh cpx --num-blocks 16384     # 16384 x 8 == 131072
#
# The device's KV capacity is fixed; CPX does not grant 8x the HBM. Running the
# cpx side at 131072 per rank is not a "control" for the locality -- it is an
# 8x larger machine, which is a thing Starscream can never be run on. It would
# be treating each logical GPU as an additional GPU rather than as a slice of
# the one in front of you.
#
# So the smaller per-XCD pool is not a confound to be subtracted out. It IS
# what Starscream buys: reads stay inside one XCD's memory partition instead of
# scattering across the whole device. What it costs is that the cross-XCD
# softmax merge -- which SPX gets implicitly from the hardware -- becomes
# explicit: two signal-pad barriers and a peer-pointer merge kernel on every
# decode step. Starscream wins where the locality outruns that cost and loses
# where it does not (small batch, short context). To see the two sides apart,
# read the per-kernel breakdown from run_paged_kerntime.sh, not a pool-size knob.

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
