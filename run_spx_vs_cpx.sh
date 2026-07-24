#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# SPX-vs-CPX Triton attention benchmark launcher (single physical GPU, MI300X).
#
# Thin wrapper around tests/kernels/attention/bench_starscream_attention.py that
# hard-sets the target config and defaults the shapes to test_harness_1.py.
# Config is FIXED here (override via env if you must):
#     dtype       = bfloat16          (kernel-side; not a knob)
#     kv dtype    = bfloat16 / auto   (kernel-side; not a knob)
#     block size  = 16
#     head size   = 128
#     query heads = 128
#     kv heads    = 8   (GQA ratio = 16)
#     threads/blk = 256               (kernel constant, gfx942)
#     arch        = gfx942 / MI300X
#
# Usage:
#   ./run_spx_vs_cpx.sh [spx|cpx|cpx-baseline]      # mode defaults to spx
#
#   spx          : hardware in SPX; 1 rank owns the whole GPU. Baseline.
#   cpx          : hardware in CPX; 8 XCDs run the Starscream path
#                  (rccl/step1/step2 merge variants). The proposed design.
#   cpx-baseline : hardware in CPX; 8 XCDs run the ORIGINAL path, starscream
#                  OFF (CPX-without-Starscream reference).
#
# The HARDWARE partition must already match (this script does NOT switch it):
#   spx           -> rocm-smi --setcomputepartition SPX
#   cpx / baseline -> rocm-smi --setcomputepartition CPX
#
# Env overrides (optional): VLLM_SRC, SEQ_LENS, BATCH_SIZES, CPX_SIZE, WARMUP,
#   ITERS, CUDA_GRAPH, SIGNAL_PAD, CSV, and Q_HEADS/KV_HEADS/HEAD_SIZE/BLOCK_SIZE
#   (to override the fixed config).

set -euo pipefail

MODE="${1:-spx}"
MODE="$(echo "$MODE" | tr '[:upper:]' '[:lower:]')"
if [[ "$MODE" != "spx" && "$MODE" != "cpx" && "$MODE" != "cpx-baseline" ]]; then
    echo "ERROR: mode must be 'spx', 'cpx', or 'cpx-baseline' (got '$MODE')" >&2
    exit 1
fi

# Metric: latency (Python timing, default) | bandwidth (rocprofv3 HBM counters)
# | both. Latency writes bench_<mode>.csv; bandwidth writes bandwidth_<mode>.csv.
METRIC="${2:-latency}"
METRIC="$(echo "$METRIC" | tr '[:upper:]' '[:lower:]')"
if [[ "$METRIC" != "latency" && "$METRIC" != "bandwidth" && "$METRIC" != "both" ]]; then
    echo "ERROR: metric must be 'latency', 'bandwidth', or 'both' (got '$METRIC')" >&2
    exit 1
fi

# Root-relative default (this script sits in the vLLM root), overridable.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_SRC="${VLLM_SRC:-$SCRIPT_DIR}"

# --- FIXED config (override via env if truly needed) ---
Q_HEADS="${Q_HEADS:-128}"
KV_HEADS="${KV_HEADS:-8}"
HEAD_SIZE="${HEAD_SIZE:-128}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"

# --- Shapes: default to test_harness_1.py's sweep ---
# test_harness_1.py cells: seq {8192,131072} x batch {1,32,64,128,1024}.
# bench runs the full seq x batch cross-product (a superset of those cells).
SEQ_LENS="${SEQ_LENS:-8192,131072}"
BATCH_SIZES="${BATCH_SIZES:-1,32,64,128,1024}"

CPX_SIZE="${CPX_SIZE:-8}"
WARMUP="${WARMUP:-25}"
ITERS="${ITERS:-200}"
CUDA_GRAPH="${CUDA_GRAPH:-1}"     # 1 = CUDA-graph replay timing, 0 = eager
SIGNAL_PAD="${SIGNAL_PAD:-1}"     # 1 = on-device barrier (keeps RCCL off merge)
CSV="${CSV:-$VLLM_SRC/bench_${MODE}.csv}"

# Bandwidth (rocprofv3) settings.
PROFILE_ITERS="${PROFILE_ITERS:-30}"
BW_CSV="${BW_CSV:-$VLLM_SRC/bandwidth_${MODE}.csv}"
TS="$(date +%Y%m%d_%H%M%S)"
BW_OUT_DIR="${BW_OUT_DIR:-$VLLM_SRC/bw_${MODE}_${TS}}"

BENCH="$VLLM_SRC/tests/kernels/attention/bench_starscream_attention.py"
PARSER="$VLLM_SRC/tests/kernels/attention/parse_rocprof_bandwidth.py"
[[ -f "$BENCH" ]] || { echo "ERROR: missing $BENCH" >&2; exit 1; }

# Map launcher mode -> device set, rank count, benchmark --mode.
if [[ "$MODE" == "spx" ]]; then
    DEVICES="${DEVICES:-0}"; NPROC=1; BENCH_MODE="spx"; HW_PART="SPX"
elif [[ "$MODE" == "cpx" ]]; then
    DEVICES="${DEVICES:-0,1,2,3,4,5,6,7}"; NPROC="$CPX_SIZE"; BENCH_MODE="cpx"; HW_PART="CPX"
else  # cpx-baseline
    DEVICES="${DEVICES:-0,1,2,3,4,5,6,7}"; NPROC="$CPX_SIZE"; BENCH_MODE="spx"; HW_PART="CPX"
fi

COMMON_ARGS=(
    --total-q-heads "$Q_HEADS"
    --total-kv-heads "$KV_HEADS"
    --head-size "$HEAD_SIZE"
    --block-size "$BLOCK_SIZE"
    --seq-lens "$SEQ_LENS"
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

# Profiled variant per mode for the bandwidth path: spx->spx, cpx->step2
# (the fused merge, the proposed path), cpx-baseline->spx (starscream off).
if [[ "$MODE" == "cpx" ]]; then PROFILE_VARIANT="step2"; else PROFILE_VARIANT="spx"; fi

echo "=================================================================="
echo " SPX-vs-CPX Triton attention benchmark (single physical GPU)"
echo "   mode         : $MODE   (bench --mode $BENCH_MODE)"
echo "   metric       : $METRIC"
echo "   config       : q=$Q_HEADS kv=$KV_HEADS (GQA $((Q_HEADS / KV_HEADS))) head_size=$HEAD_SIZE block_size=$BLOCK_SIZE bf16"
echo "   devices      : $DEVICES   (nproc=$NPROC)"
echo "   seq_lens     : $SEQ_LENS"
echo "   batch_sizes  : $BATCH_SIZES"
echo "=================================================================="
echo " Reminder: hardware must be in $HW_PART partition mode already."
echo "   rocm-smi --setcomputepartition $HW_PART"
echo "=================================================================="

# --- Latency: Python timing (CUDA-graph), writes bench_<mode>.csv ---------
run_latency() {
    echo "[latency] warmup/iters=$WARMUP/$ITERS cuda_graph=$CUDA_GRAPH -> $CSV"
    if [[ "$BENCH_MODE" == "cpx" ]]; then
        CUDA_VISIBLE_DEVICES="$DEVICES" \
        PYTHONPATH="$VLLM_SRC" \
        TORCH_SYMM_MEM_DISABLE_MULTICAST=1 \
        VLLM_STARSCREAM_SIGNAL_PAD_BARRIER="$SIGNAL_PAD" \
        torchrun --nnodes=1 --nproc-per-node="$NPROC" \
            "$BENCH" --mode cpx --cpx-size "$CPX_SIZE" --variants step2 \
            "${COMMON_ARGS[@]}"
    else
        CUDA_VISIBLE_DEVICES="$DEVICES" \
        PYTHONPATH="$VLLM_SRC" \
        torchrun --nnodes=1 --nproc-per-node="$NPROC" \
            "$BENCH" --mode spx "${COMMON_ARGS[@]}"
    fi
}

# --- Bandwidth: rocprofv3 HBM counters, writes bandwidth_<mode>.csv --------
# Per (seq_len,batch) cell: TWO --pmc passes (FETCH_SIZE, WRITE_SIZE -- each is
# a derived counter; both in one pass exceed the HW counter budget), then the
# parser globs the cell's dbs and computes BW = (fetch+write)/kernel_busy_ns.
run_bandwidth() {
    [[ -f "$PARSER" ]] || { echo "ERROR: missing $PARSER" >&2; exit 1; }
    command -v rocprofv3 >/dev/null 2>&1 || {
        echo "ERROR: rocprofv3 not on PATH (needed for bandwidth)" >&2; exit 1; }
    mkdir -p "$BW_OUT_DIR"; rm -f "$BW_CSV"
    echo "[bandwidth] variant=$PROFILE_VARIANT iters=$PROFILE_ITERS -> $BW_CSV"

    local prof_common=(
        --total-q-heads "$Q_HEADS" --total-kv-heads "$KV_HEADS"
        --head-size "$HEAD_SIZE" --block-size "$BLOCK_SIZE"
        --profile --profile-variant "$PROFILE_VARIANT"
        --profile-iters "$PROFILE_ITERS" --skip-sanity
    )
    [[ "$BENCH_MODE" == "cpx" ]] && prof_common+=(--cpx-size "$CPX_SIZE")
    local bench_mode_flag="$BENCH_MODE"

    IFS=',' read -ra SEQS <<< "$SEQ_LENS"
    IFS=',' read -ra BATCHES <<< "$BATCH_SIZES"
    for S in "${SEQS[@]}"; do
        for B in "${BATCHES[@]}"; do
            local cell="$BW_OUT_DIR/S${S}_B${B}"; mkdir -p "$cell"
            echo "--- bandwidth S=$S B=$B (variant=$PROFILE_VARIANT) ---"
            local env_prefix=(CUDA_VISIBLE_DEVICES="$DEVICES" PYTHONPATH="$VLLM_SRC")
            [[ "$BENCH_MODE" == "cpx" ]] && env_prefix+=(
                TORCH_SYMM_MEM_DISABLE_MULTICAST=1
                VLLM_STARSCREAM_SIGNAL_PAD_BARRIER="$SIGNAL_PAD")

            local failed=0
            for CTR in FETCH_SIZE WRITE_SIZE; do
                env "${env_prefix[@]}" \
                    rocprofv3 --pmc "$CTR" -d "$cell/$CTR" \
                    -- torchrun --nnodes=1 --nproc-per-node="$NPROC" \
                       "$BENCH" --mode "$bench_mode_flag" \
                       --profile-seq-len "$S" --profile-batch "$B" \
                       "${prof_common[@]}" \
                    2>&1 | tee "$cell/rocprof_$CTR.log" || { failed=1; break; }
            done
            [[ "$failed" -eq 1 ]] && { echo "  pass failed; skipping"; continue; }

            [[ "$(find "$cell" -name '*.db' | wc -l)" -eq 0 ]] && {
                echo "  no db files; skipping parse"; continue; }
            PYTHONPATH="$VLLM_SRC" python3 "$PARSER" \
                --db-glob "$cell/**/*.db" --mode "$MODE" \
                --seq-len "$S" --batch "$B" --out-csv "$BW_CSV"
        done
    done
    echo "[bandwidth] done -> $BW_CSV"
    [[ -f "$BW_CSV" ]] && cat "$BW_CSV"
}

case "$METRIC" in
    latency)   run_latency ;;
    bandwidth) run_bandwidth ;;
    both)      run_latency; run_bandwidth ;;
esac
