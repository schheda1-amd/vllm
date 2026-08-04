#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Per-kernel GPU-time profiling for the HIP paged-attention decode path,
# SPX baseline vs CPX + Starscream.
#
# Different question from run_paged_bandwidth.sh. That one asks "what bandwidth
# did the attention path reach" and collapses the filtered kernels into one
# number. This one asks "WHICH kernel is eating the CPX-vs-SPX gap" -- the QKV
# kernel, the HIP reduce kernel, or the cross-XCD merge -- so it uses
# --kernel-trace instead of --pmc:
#   * one pass, not two (no counter-slot budget to split FETCH/WRITE across)
#   * no PMC replay, so the durations are the real ones
#   * every kernel is kept, including the barrier and RCCL that a bandwidth
#     filter deliberately drops -- their cost is part of what is in question
#
# It profiles the SAME harnesses the benchmark numbers come from, pinned to one
# cell with --only, rather than a separate launcher that can drift out of sync.
#
# Usage (hardware partition must ALREADY match the mode; this does not switch):
#   rocm-smi --setcomputepartition SPX
#   ./run_paged_kerntime.sh spx
#   rocm-smi --setcomputepartition CPX
#   ./run_paged_kerntime.sh cpx
#   ./run_paged_kerntime.sh compare          # side-by-side from the shared CSV
#
# Both modes append to the same CSV ($CSV), so the two runs -- which cannot be
# in the same boot of the partition -- still land in one comparable table.
#
# Env: CELLS ("S,B" list), NUM_ITERS, CPX_SIZE, NUM_BLOCKS, DEVICES, OUT_DIR,
#      CSV, VLLM_SRC, TOP, RAW=1.

set -euo pipefail

MODE="${1:-cpx}"; shift || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_SRC="${VLLM_SRC:-$SCRIPT_DIR}"
PARSER="$VLLM_SRC/tests/kernels/attention/parse_rocprof_kernels.py"
CSV="${CSV:-$VLLM_SRC/kerntime_paged.csv}"

# 8192/1024 is the cell where CPX trails SPX; 131072/1024 is the cell where it
# wins. Profiling both in one go makes the difference between them readable
# instead of asserted.
CELLS="${CELLS:-8192,1024 131072,1024}"
NUM_ITERS="${NUM_ITERS:-20}"
CPX_SIZE="${CPX_SIZE:-8}"
TOP="${TOP:-15}"
RAW_ARG=()
[[ "${RAW:-0}" == "1" ]] && RAW_ARG=(--raw)

if [[ "$MODE" == "compare" ]]; then
    [[ -f "$CSV" ]] || { echo "ERROR: no $CSV -- run spx and cpx first" >&2; exit 1; }
    exec python3 "$PARSER" --compare --in-csv "$CSV"
fi

# Re-parse dbs that are already on disk -- the trace is the expensive part and
# it does not change when the parser does, so a tooling fix must not cost
# another partition flip and another profiling run.
#   ./run_paged_kerntime.sh reparse kern_spx_2026.../ spx
if [[ "$MODE" == "reparse" ]]; then
    DIR="${1:?usage: reparse <out_dir> <label>}"; LBL="${2:?usage: reparse <out_dir> <label>}"
    found=0
    for CELL_DIR in "$DIR"/S*_B*; do
        [[ -d "$CELL_DIR" ]] || continue
        CELL="$(basename "$CELL_DIR")"; S="${CELL#S}"; S="${S%%_B*}"; B="${CELL##*_B}"
        find "$CELL_DIR" -name '*.db' -print -quit | grep -q . || continue
        found=1
        python3 "$PARSER" --db-glob "$CELL_DIR/**/*.db" \
            --label "$LBL" --seq-len "$S" --batch "$B" \
            --top "$TOP" --out-csv "$CSV" "${RAW_ARG[@]}"
    done
    [[ "$found" -eq 1 ]] || { echo "ERROR: no S*_B*/**.db under $DIR" >&2; exit 1; }
    exit 0
fi

[[ -f "$PARSER" ]] || { echo "ERROR: missing $PARSER" >&2; exit 1; }
command -v rocprofv3 >/dev/null || { echo "ERROR: rocprofv3 not on PATH" >&2; exit 1; }

TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-$VLLM_SRC/kern_${MODE}_${TS}}"
mkdir -p "$OUT_DIR"

case "$MODE" in
  spx)
    HARNESS="$VLLM_SRC/test_harness_1_paged.py"
    # SPX: one rank owns the whole KV pool.
    NUM_BLOCKS="${NUM_BLOCKS:-131072}"
    DEVICES="${DEVICES:-0}"
    ;;
  cpx)
    HARNESS="$VLLM_SRC/test_harness_1_paged_cpx.py"
    # CPX: --num-blocks is PER RANK, so 131072/8 keeps the total pool -- and
    # therefore the KV scatter pattern -- the same as the SPX side. Getting this
    # wrong makes the two runs differ in TLB/cache behaviour, not just partition.
    NUM_BLOCKS="${NUM_BLOCKS:-16384}"
    DEVICES="${DEVICES:-0,1,2,3,4,5,6,7}"
    export TORCH_SYMM_MEM_DISABLE_MULTICAST="${TORCH_SYMM_MEM_DISABLE_MULTICAST:-1}"
    ;;
  *)
    echo "ERROR: mode must be spx | cpx | compare (got '$MODE')" >&2; exit 1 ;;
esac
[[ -f "$HARNESS" ]] || { echo "ERROR: missing $HARNESS" >&2; exit 1; }

echo "=================================================================="
echo " Paged-attention per-kernel time (rocprofv3 --kernel-trace)"
echo "   mode       : $MODE"
echo "   harness    : $(basename "$HARNESS")"
echo "   cells      : $CELLS"
echo "   num_iters  : $NUM_ITERS   num_blocks: $NUM_BLOCKS$([[ $MODE == cpx ]] && echo ' (per rank)')"
echo "   devices    : $DEVICES"
echo "   out dir    : $OUT_DIR"
echo "   csv        : $CSV"
echo "=================================================================="
echo " Reminder: hardware must ALREADY be in ${MODE^^} partition mode."
echo "=================================================================="

for CELL in $CELLS; do
    S="${CELL%%,*}"; B="${CELL##*,}"
    CELL_DIR="$OUT_DIR/S${S}_B${B}"
    mkdir -p "$CELL_DIR"

    echo ""
    echo "--- $MODE  S=$S B=$B ---"

    HARNESS_ARGS=(
        --only "$S,$B"
        --num-iters "$NUM_ITERS"
        --num-blocks "$NUM_BLOCKS"
        "$@"
    )

    # rocprofv3 wraps torchrun and follows its children, giving one db per rank
    # -- the pattern tests/kernels/attention/run_starscream_bandwidth.sh proved.
    if [[ "$MODE" == "cpx" ]]; then
        RUN=(torchrun --nnodes=1 --nproc-per-node="$CPX_SIZE"
             "$HARNESS" "${HARNESS_ARGS[@]}")
    else
        RUN=(python3 "$HARNESS" "${HARNESS_ARGS[@]}")
    fi

    env CUDA_VISIBLE_DEVICES="$DEVICES" PYTHONPATH="$VLLM_SRC" \
        rocprofv3 --kernel-trace --output-format rocpd \
                  -d "$CELL_DIR" \
                  -- "${RUN[@]}" \
        2>&1 | tee "$CELL_DIR/rocprof.log" || {
            echo "  rocprofv3 failed for S=$S B=$B"; continue; }

    NDB="$(find "$CELL_DIR" -name '*.db' 2>/dev/null | wc -l)"
    if [[ "$NDB" -eq 0 ]]; then
        echo "  no *.db under $CELL_DIR (rocprofv3 may have written csv);" \
             "skipping parse"
        continue
    fi
    echo "  found $NDB db file(s) ($([[ $MODE == cpx ]] && echo 'expect one per rank' || echo 'expect 1'))"

    PYTHONPATH="$VLLM_SRC" python3 "$PARSER" \
        --db-glob "$CELL_DIR/**/*.db" \
        --label "$MODE" --seq-len "$S" --batch "$B" \
        --top "$TOP" --out-csv "$CSV" "${RAW_ARG[@]}"
done

echo ""
echo "=================================================================="
echo " Done. Rows appended to $CSV"
echo " Run './run_paged_kerntime.sh compare' after BOTH modes have run."
echo "=================================================================="
