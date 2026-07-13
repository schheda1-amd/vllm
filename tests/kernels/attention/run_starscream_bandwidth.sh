#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Memory-bandwidth profiling for the Starscream SPX-vs-CPX attention benchmark.
#
# Decode attention is memory-bound, so achieved HBM bandwidth is the metric of
# record: higher aggregate bandwidth correlates with better runtime. This script
# runs the benchmark in --profile mode under rocprofv3, collecting the
# memory-controller request counters (TCC_EA_RDREQ / TCC_EA_WRREQ) plus kernel
# durations, for each (seq_len, batch) cell -- ORIGINAL seq-len shapes only, no
# token offsets. It then computes:
#     bytes = (RDREQ + WRREQ) * 64                       # 64B EA line
#     BW    = bytes / kernel_busy_time                   # GB/s
#   SPX: single rank's BW.  CPX: SUM over the 8 XCDs (concurrent) = aggregate.
#
# Usage (same shape as run_starscream_bench.sh):
#   ./run_starscream_bandwidth.sh [spx|cpx|cpx-baseline] [model]
#
# The HARDWARE partition must already match (spx->SPX, cpx*->CPX). rocprofv3
# must be on PATH (it is in the container).
#
# Env overrides: VLLM_SRC, MODEL, SEQ_LENS, BATCH_SIZES, CPX_SIZE, SIGNAL_PAD,
#   PROFILE_ITERS, PROFILE_VARIANT, DEVICES, OUT_DIR, CSV, Q_HEADS/KV_HEADS/HEAD_SIZE.

set -euo pipefail

MODE="${1:-spx}"
MODE="$(echo "$MODE" | tr '[:upper:]' '[:lower:]')"
if [[ "$MODE" != "spx" && "$MODE" != "cpx" && "$MODE" != "cpx-baseline" ]]; then
    echo "ERROR: mode must be 'spx', 'cpx', or 'cpx-baseline' (got '$MODE')" >&2
    exit 1
fi

MODEL="${2:-${MODEL:-llama3-70b}}"
MODEL="$(echo "$MODEL" | tr '[:upper:]' '[:lower:]')"
if [[ "$MODEL" != "llama3-70b" && "$MODEL" != "llama3-405b" \
      && "$MODEL" != "gpt-oss-120b" ]]; then
    echo "ERROR: model must be llama3-70b | llama3-405b | gpt-oss-120b (got '$MODEL')" >&2
    exit 1
fi

VLLM_SRC="${VLLM_SRC:-/workspace/vllm}"
SEQ_LENS="${SEQ_LENS:-256,8192,131072}"
BATCH_SIZES="${BATCH_SIZES:-1,8,32,64,128,256,512,1024}"
CPX_SIZE="${CPX_SIZE:-8}"
SIGNAL_PAD="${SIGNAL_PAD:-1}"
PROFILE_ITERS="${PROFILE_ITERS:-30}"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-/workspace/vllm/bw_${MODEL}_${MODE}_${TS}}"
CSV="${CSV:-/workspace/vllm/bandwidth_${MODEL}_${MODE}.csv}"

BENCH="${VLLM_SRC}/tests/kernels/attention/bench_starscream_attention.py"
PARSER="${VLLM_SRC}/tests/kernels/attention/parse_rocprof_bandwidth.py"

# Mode -> devices / nproc / benchmark --mode / profiled variant.
if [[ "$MODE" == "spx" ]]; then
    DEVICES="${DEVICES:-0}"; NPROC=1; BENCH_MODE="spx"
    PROFILE_VARIANT="${PROFILE_VARIANT:-spx}"; HW_PART="SPX"
elif [[ "$MODE" == "cpx" ]]; then
    DEVICES="${DEVICES:-0,1,2,3,4,5,6,7}"; NPROC="$CPX_SIZE"; BENCH_MODE="cpx"
    PROFILE_VARIANT="${PROFILE_VARIANT:-step2}"; HW_PART="CPX"
else  # cpx-baseline
    DEVICES="${DEVICES:-0,1,2,3,4,5,6,7}"; NPROC="$CPX_SIZE"; BENCH_MODE="spx"
    PROFILE_VARIANT="${PROFILE_VARIANT:-spx}"; HW_PART="CPX"
fi

# Optional per-dimension head overrides (else --model preset wins).
HEAD_ARGS=()
[[ -n "${Q_HEADS:-}" ]]   && HEAD_ARGS+=(--total-q-heads "$Q_HEADS")
[[ -n "${KV_HEADS:-}" ]]  && HEAD_ARGS+=(--total-kv-heads "$KV_HEADS")
[[ -n "${HEAD_SIZE:-}" ]] && HEAD_ARGS+=(--head-size "$HEAD_SIZE")

mkdir -p "$OUT_DIR"
rm -f "$CSV"

echo "=================================================================="
echo " Starscream memory-bandwidth profiling (rocprofv3)"
echo "   mode / model : $MODE / $MODEL   (bench --mode $BENCH_MODE)"
echo "   variant      : $PROFILE_VARIANT"
echo "   devices      : $DEVICES   (nproc=$NPROC)"
echo "   seq_lens     : $SEQ_LENS"
echo "   batch_sizes  : $BATCH_SIZES"
echo "   profile_iters: $PROFILE_ITERS"
echo "   out dir      : $OUT_DIR"
echo "   csv          : $CSV"
echo "=================================================================="
echo " Reminder: hardware must be in $HW_PART partition mode already."
echo "=================================================================="

IFS=',' read -ra SEQS <<< "$SEQ_LENS"
IFS=',' read -ra BATCHES <<< "$BATCH_SIZES"

find_port() {
    local p=29500
    while ss -tuln 2>/dev/null | grep -q ":$p "; do
        p=$((p + 1)); [[ $p -gt 30500 ]] && p=29600
    done
    echo "$p"
}

for S in "${SEQS[@]}"; do
    for B in "${BATCHES[@]}"; do
        CELL_DIR="$OUT_DIR/S${S}_B${B}"
        mkdir -p "$CELL_DIR"
        PORT="$(find_port)"

        COMMON=(
            --mode "$BENCH_MODE" --model "$MODEL" "${HEAD_ARGS[@]}"
            --profile --profile-seq-len "$S" --profile-batch "$B"
            --profile-variant "$PROFILE_VARIANT" --profile-iters "$PROFILE_ITERS"
            --skip-sanity
        )
        [[ "$BENCH_MODE" == "cpx" ]] && COMMON+=(--cpx-size "$CPX_SIZE")

        echo ""
        echo "--- profiling S=$S B=$B (variant=$PROFILE_VARIANT) ---"

        # rocprofv3 PMC-ONLY pass. IMPORTANT: --pmc must NOT be combined with
        # --kernel-trace -- PMC collection needs a replay pass, and mixing the
        # two silently yields zero counter rows. A --pmc pass also populates
        # rocpd_kernel_dispatch (durations), so we get both from one run.
        # FETCH_SIZE / WRITE_SIZE are KB read/written at the HBM interface
        # (this GPU's exposed memory-traffic counters).
        env_prefix=(CUDA_VISIBLE_DEVICES="$DEVICES" PYTHONPATH="$VLLM_SRC")
        if [[ "$BENCH_MODE" == "cpx" ]]; then
            env_prefix+=(TORCH_SYMM_MEM_DISABLE_MULTICAST=1
                         VLLM_STARSCREAM_SIGNAL_PAD_BARRIER="$SIGNAL_PAD")
        fi

        env "${env_prefix[@]}" \
            rocprofv3 --pmc FETCH_SIZE WRITE_SIZE \
                      -d "$CELL_DIR" \
                      -- torchrun --nnodes=1 --nproc-per-node="$NPROC" \
                         --rdzv_endpoint="localhost:$PORT" \
                         "$BENCH" "${COMMON[@]}" \
            2>&1 | tee "$CELL_DIR/rocprof.log" || {
                echo "  rocprofv3 run failed for S=$S B=$B (see log)"; continue; }

        # rocprofv3 writes one <pid>_results.db PER RANK under -d (nested in a
        # per-node subdir). SPX -> 1 db; CPX -> 8 dbs (one per XCD). The parser
        # globs and merges all of them, so pass the whole cell dir.
        NDB="$(find "$CELL_DIR" -name '*.db' 2>/dev/null | wc -l)"
        if [[ "$NDB" -eq 0 ]]; then
            echo "  no *.db found under $CELL_DIR; skipping parse"
            continue
        fi
        echo "  found $NDB db file(s) for this cell"

        PYTHONPATH="$VLLM_SRC" python3 "$PARSER" \
            --db-glob "$CELL_DIR/**/*.db" --mode "$MODE" \
            --seq-len "$S" --batch "$B" --out-csv "$CSV"
    done
done

echo ""
echo "=================================================================="
echo " Bandwidth profiling complete. Summary CSV:"
echo "   $CSV"
echo "=================================================================="
[[ -f "$CSV" ]] && cat "$CSV"
