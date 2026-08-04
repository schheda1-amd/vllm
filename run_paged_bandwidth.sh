#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Memory-bandwidth profiling for the C/HIP paged attention kernel (SPX decode
# path, single GPU). Mirrors tests/kernels/attention/run_starscream_bandwidth.sh
# but for test_harness_1_paged_profile.py -- NO torchrun / distributed (the
# paged kernel harness is single-GPU only), so it profiles the SPX partition on
# one device.
#
# For each (seq_len, batch) cell it runs the launcher under rocprofv3 twice --
# once per DERIVED HBM counter (FETCH_SIZE, then WRITE_SIZE); requesting both in
# one pass exceeds the hardware counter-slot budget. Then it calls
# parse_rocprof_bandwidth.py, which globs the cell's dbs, scopes bytes+time to
# the paged kernels (paged_attention_ll4mi*), and computes:
#     bytes = (FETCH_SIZE + WRITE_SIZE) * 1024        # KB -> bytes
#     BW    = bytes / kernel_busy_ns                  # GB/s
#
# Head shapes are FIXED to the benchmarked paged config (same as
# test_harness_1_paged.py): 128 q / 8 kv (GQA 16), head_size 128, block_size 16,
# bfloat16, kv-cache-dtype auto. `model` is only a CSV/output-dir label.
#
# Usage:
#   ./run_paged_bandwidth.sh [label]     (default label: paged)
#
# Env overrides: VLLM_SRC, SEQ_LENS, BATCH_SIZES, PROFILE_ITERS, DEVICE,
#   OUT_DIR, CSV, DTYPE, BLOCK_SIZE, KV_CACHE_DTYPE,
#   NUM_QUERY_HEADS/NUM_KV_HEADS/HEAD_SIZE (override the fixed shapes).
#
# rocprofv3 must be on PATH (it is in the ROCm container). Hardware must be in
# SPX partition mode.

set -euo pipefail

LABEL="${1:-${LABEL:-paged}}"

# Root-relative paths, like test_harness_1_paged.py: default to this script's
# own directory (the vLLM root where the harness is copied), overridable.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_SRC="${VLLM_SRC:-$SCRIPT_DIR}"
SEQ_LENS="${SEQ_LENS:-256,8192,131072}"
BATCH_SIZES="${BATCH_SIZES:-1,8,32,64,128,256,512,1024}"
PROFILE_ITERS="${PROFILE_ITERS:-30}"
DEVICE="${DEVICE:-0}"
DTYPE="${DTYPE:-bfloat16}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-$VLLM_SRC/bw_${LABEL}_${TS}}"
CSV="${CSV:-$VLLM_SRC/bandwidth_${LABEL}.csv}"

LAUNCHER="$VLLM_SRC/test_harness_1_paged_profile.py"
PARSER="$VLLM_SRC/tests/kernels/attention/parse_rocprof_bandwidth.py"

# The paged kernels (attention.cu) all share this prefix: QKV_mfma16,
# QKV_mfma4, and the reduce kernel. This REPLACES the parser's Triton-oriented
# default filter (kernel_unified_attention,...), which would match nothing here.
KERNEL_FILTER="${KERNEL_FILTER:-paged_attention_ll4mi}"

# Optional per-dimension head overrides (else the launcher's --model preset).
HEAD_ARGS=()
[[ -n "${NUM_QUERY_HEADS:-}" ]] && HEAD_ARGS+=(--num-query-heads "$NUM_QUERY_HEADS")
[[ -n "${NUM_KV_HEADS:-}" ]]    && HEAD_ARGS+=(--num-kv-heads "$NUM_KV_HEADS")
[[ -n "${HEAD_SIZE:-}" ]]       && HEAD_ARGS+=(--head-size "$HEAD_SIZE")

for f in "$LAUNCHER" "$PARSER"; do
    [[ -f "$f" ]] || { echo "ERROR: missing $f" >&2; exit 1; }
done

mkdir -p "$OUT_DIR"
rm -f "$CSV"

echo "=================================================================="
echo " Paged-attention memory-bandwidth profiling (rocprofv3, SPX 1-GPU)"
echo "   label        : $LABEL"
echo "   shapes       : q=${NUM_QUERY_HEADS:-128} kv=${NUM_KV_HEADS:-8} head_size=${HEAD_SIZE:-128} (GQA $(( ${NUM_QUERY_HEADS:-128} / ${NUM_KV_HEADS:-8} )))"
echo "   dtype/kv     : $DTYPE / $KV_CACHE_DTYPE   block_size=$BLOCK_SIZE"
echo "   device       : $DEVICE"
echo "   seq_lens     : $SEQ_LENS"
echo "   batch_sizes  : $BATCH_SIZES"
echo "   profile_iters: $PROFILE_ITERS"
echo "   kernel filter: $KERNEL_FILTER"
echo "   out dir      : $OUT_DIR"
echo "   csv          : $CSV"
echo "=================================================================="
echo " Reminder: hardware must be in SPX partition mode already."
echo "=================================================================="

IFS=',' read -ra SEQS <<< "$SEQ_LENS"
IFS=',' read -ra BATCHES <<< "$BATCH_SIZES"

for S in "${SEQS[@]}"; do
    for B in "${BATCHES[@]}"; do
        CELL_DIR="$OUT_DIR/S${S}_B${B}"
        mkdir -p "$CELL_DIR"

        COMMON=(
            --model "$LABEL" "${HEAD_ARGS[@]}"
            --seq-len "$S" --batch "$B"
            --profile-iters "$PROFILE_ITERS"
            --block-size "$BLOCK_SIZE" --dtype "$DTYPE"
            --kv-cache-dtype "$KV_CACHE_DTYPE"
        )

        echo ""
        echo "--- profiling S=$S B=$B ---"

        # Two SEPARATE --pmc passes (FETCH_SIZE, WRITE_SIZE): each derived
        # counter expands to several base TCC counters, and both together
        # exceed the hardware counter-slot budget (rocprof "error 38").
        # Single GPU, single process -> no torchrun, one db per pass.
        pass_failed=0
        for CTR in FETCH_SIZE WRITE_SIZE; do
            env CUDA_VISIBLE_DEVICES="$DEVICE" PYTHONPATH="$VLLM_SRC" \
                rocprofv3 --pmc "$CTR" \
                          -d "$CELL_DIR/$CTR" \
                          -- python3 "$LAUNCHER" "${COMMON[@]}" \
                2>&1 | tee "$CELL_DIR/rocprof_$CTR.log" || {
                    echo "  rocprofv3 $CTR pass failed for S=$S B=$B"; \
                    pass_failed=1; break; }
        done
        [[ "$pass_failed" -eq 1 ]] && continue

        # SPX single-GPU -> 2 dbs (one per pass).
        NDB="$(find "$CELL_DIR" -name '*.db' 2>/dev/null | wc -l)"
        if [[ "$NDB" -eq 0 ]]; then
            echo "  no *.db found under $CELL_DIR; skipping parse"
            continue
        fi
        echo "  found $NDB db file(s) for this cell"

        PYTHONPATH="$VLLM_SRC" python3 "$PARSER" \
            --db-glob "$CELL_DIR/**/*.db" --mode spx \
            --seq-len "$S" --batch "$B" --out-csv "$CSV" \
            --kernel-filter "$KERNEL_FILTER"
    done
done

echo ""
echo "=================================================================="
echo " Paged bandwidth profiling complete. Summary CSV:"
echo "   $CSV"
echo "=================================================================="
[[ -f "$CSV" ]] && cat "$CSV"
