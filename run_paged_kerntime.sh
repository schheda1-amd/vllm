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
# The parser splits the two signal-pad barriers apart (they share one kernel
# symbol, dispatched either side of the merge) and reports each as floor + skew.
# That distinction is the whole point when deciding whether to attack them:
# barrier 1 is a cross-device RAW guard a put-based producer could delete,
# barrier 2 is a WAR guard on buffer reuse that stays, and skew is neither --
# it is load imbalance that no barrier change removes.
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
# Runs against whatever vllm your shell already imports. It does NOT touch
# PYTHONPATH and does NOT need a source checkout -- drop this script anywhere
# and give it explicit paths:
#
#   PARSER=/path/to/parse_rocprof_kernels.py \
#   CPX_HARNESS=/app/vllm/test_harness_1_paged_cpx.py \
#     ./run_paged_kerntime.sh cpx
#
# Env: CELLS ("S,B" list), NUM_ITERS, CPX_SIZE, NUM_BLOCKS, DEVICES, OUT_DIR,
#      CSV, TOP, RAW=1.
#      Paths: PARSER, SPX_HARNESS, CPX_HARNESS (each defaults to a location
#      under VLLM_SRC, which itself defaults to this script's directory).
#      VLLM_PYTHONPATH to prepend to PYTHONPATH; unset means don't touch it.

set -euo pipefail

MODE="${1:-cpx}"; shift || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_SRC="${VLLM_SRC:-$SCRIPT_DIR}"

# Every path below is an independent override, so this script can be dropped
# next to an ALREADY-INSTALLED vllm and point at the parser wherever it happens
# to live. Deriving them all from one root forced the layout of a source
# checkout onto a machine that only has the installed package.
PARSER="${PARSER:-$VLLM_SRC/tests/kernels/attention/parse_rocprof_kernels.py}"
SPX_HARNESS="${SPX_HARNESS:-$VLLM_SRC/test_harness_1_paged.py}"
CPX_HARNESS="${CPX_HARNESS:-$VLLM_SRC/test_harness_1_paged_cpx.py}"
CSV="${CSV:-$VLLM_SRC/kerntime_paged.csv}"

# PYTHONPATH is NOT set by default any more.
#
# It used to be forced to $VLLM_SRC, which is right for a source checkout with
# the extension built in place and actively wrong anywhere else: it puts an
# uncompiled source `vllm/` ahead of the installed package, and the only symptom
# is "failed to import vllm._rocm_C" from a build that is perfectly fine.
# Default is now to inherit the environment -- if `python3 -c "import vllm"`
# works in your shell, it works here.
#
# Set VLLM_PYTHONPATH explicitly to prepend something (e.g. a source checkout
# whose extension you did build in place).
PY_ENV=()
[[ -n "${VLLM_PYTHONPATH:-}" ]] && PY_ENV=(PYTHONPATH="$VLLM_PYTHONPATH${PYTHONPATH:+:$PYTHONPATH}")

# 8K first, across the batch range, because that is where CPX trails SPX and
# therefore where the per-kernel breakdown has something to explain. The three
# batches are not redundant:
#   B1    -- one sequence over 8 XCDs, ~1024 tokens each. Attention work is
#            near nothing, so whatever is left IS the fixed Starscream cost:
#            barriers, merge, dispatch. The cleanest read on the floor.
#   B64   -- the cell that wins at 16 query heads; shows the floor being
#            amortised rather than removed.
#   B1024 -- attention-dominated, so it bounds how small the fixed cost is
#            relative to real work.
# 131072,1024 stays as the contrast: same kernels, CPX ahead.
CELLS="${CELLS:-8192,1 8192,64 8192,1024 131072,1024}"
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
            --num-iters "$NUM_ITERS" \
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
    HARNESS="$SPX_HARNESS"
    # SPX: one rank owns the whole KV pool.
    NUM_BLOCKS="${NUM_BLOCKS:-131072}"
    DEVICES="${DEVICES:-0}"
    ;;
  cpx)
    HARNESS="$CPX_HARNESS"
    # CPX: --num-blocks is PER RANK, so 131072/8 keeps the KV footprint on the
    # PHYSICAL GPU the same as the SPX side. The device's capacity is fixed --
    # CPX does not grant 8x the HBM -- so 131072 per rank is not an alternative
    # worth profiling; it is a machine that does not exist, one that counts each
    # logical GPU as an extra GPU instead of a slice of the measured one.
    #
    # Each rank consequently scatters over an 8x smaller range and its reads
    # stay in its own memory partition. That locality is what Starscream buys by
    # using the topology explicitly, and it is paid for by making the cross-XCD
    # merge explicit -- barriers and a peer-pointer kernel where SPX has the
    # hardware do it. This script's per-kernel breakdown is how they are told
    # apart: QKV time carries the locality, merge and barrier carry the price.
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
echo "   harness    : $HARNESS"
echo "   parser     : $PARSER"
echo "   python     : $(python3 -c 'import vllm,os;print(os.path.dirname(vllm.__file__))' 2>/dev/null || echo 'import vllm FAILED')"
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

    env CUDA_VISIBLE_DEVICES="$DEVICES" "${PY_ENV[@]}" \
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

    # --num-iters is passed for the step-count cross-check only: the parser
    # derives the window from the trace, then says so loudly if the two
    # disagree. Without it a mis-anchored window is silent.
    env "${PY_ENV[@]}" python3 "$PARSER" \
        --db-glob "$CELL_DIR/**/*.db" \
        --label "$MODE" --seq-len "$S" --batch "$B" \
        --num-iters "$NUM_ITERS" \
        --top "$TOP" --out-csv "$CSV" "${RAW_ARG[@]}"
done

echo ""
echo "=================================================================="
echo " Done. Rows appended to $CSV"
echo " Run './run_paged_kerntime.sh compare' after BOTH modes have run."
echo "=================================================================="
