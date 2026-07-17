#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Kernel-time-breakdown profiler for the Starscream attention benchmark.
#
# Mirrors run_starscream_bench.sh's case set (spx | cpx | cpx-baseline, any
# model, full seq x batch sweep) but runs each cell under rocprofv3
# --kernel-trace and reports a PER-KERNEL TIME breakdown, so the collective
# (RCCL allgather), the attention kernel, and the merge (reduce_segments) are
# separated. Collective time here == collective + straggler wait, which is the
# real cost of the kernel on the critical path.
#
# For cpx mode we profile the rccl VARIANT (allgather + reduce_segments are
# distinct kernels -> cleanly separable). step1/step2 are intentionally NOT
# profiled here: step2 fuses the allgather into one kernel, so there is no
# separable collective to time.
#
# Usage (same shape as run_starscream_bench.sh):
#   rocm-smi --setcomputepartition SPX
#   ./profile_valley.sh spx  llama3-70b
#   rocm-smi --setcomputepartition CPX
#   ./profile_valley.sh cpx           llama3-70b   # rccl variant
#   ./profile_valley.sh cpx-baseline  llama3-70b
#
# Env overrides: VLLM_SRC, MODEL, SEQ_LENS, BATCH_SIZES, PROFILE_ITERS,
#   CPX_SIZE, SIGNAL_PAD, DEVICES, OUT_DIR, CSV, Q_HEADS/KV_HEADS/HEAD_SIZE,
#   NUM_SEGMENTS.

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
SEQ_LENS="${SEQ_LENS:-16384,32768,65536}"
BATCH_SIZES="${BATCH_SIZES:-1,8,32,64,128,256,512,1024}"
CPX_SIZE="${CPX_SIZE:-8}"
SIGNAL_PAD="${SIGNAL_PAD:-1}"          # only relevant to symm-mem variants
NUM_SEGMENTS="${NUM_SEGMENTS:-16}"
PROFILE_ITERS="${PROFILE_ITERS:-30}"
TS="$(date +%Y%m%d_%H%M%S)"
# OUT_DIR / CSV are finalized AFTER PROFILE_VARIANT is resolved (below), so the
# variant is in the filename and step2 never overwrites the rccl trace.

BENCH="${VLLM_SRC}/tests/kernels/attention/bench_starscream_attention.py"

# Mode -> devices / nproc / benchmark --mode / profiled variant.
#   spx          : 1 rank, whole GPU, bench --mode spx, variant spx
#   cpx          : 8 XCDs, bench --mode cpx, variant rccl (separable collective)
#   cpx-baseline : 8 XCDs, bench --mode spx (starscream OFF), variant spx
if [[ "$MODE" == "spx" ]]; then
    DEVICES="${DEVICES:-0}"; NPROC=1; BENCH_MODE="spx"
    PROFILE_VARIANT="${PROFILE_VARIANT:-spx}"; HW_PART="SPX"
elif [[ "$MODE" == "cpx" ]]; then
    DEVICES="${DEVICES:-0,1,2,3,4,5,6,7}"; NPROC="$CPX_SIZE"; BENCH_MODE="cpx"
    # Default rccl (separable AG + reduce_segments). Override with
    # PROFILE_VARIANT=step2 to trace the fused path.
    PROFILE_VARIANT="${PROFILE_VARIANT:-rccl}"; HW_PART="CPX"
else  # cpx-baseline
    DEVICES="${DEVICES:-0,1,2,3,4,5,6,7}"; NPROC="$CPX_SIZE"; BENCH_MODE="spx"
    PROFILE_VARIANT="${PROFILE_VARIANT:-spx}"; HW_PART="CPX"
fi

# ---------------------------------------------------------------------------
# NCCL kernel bucket -- what the `collective` bucket contains per variant:
#   rccl : query-AG (grp.all_gather(q_local) in _attention_call, ALWAYS rccl)
#          + meta-AG (get_dcp_group().all_gather of the reduce meta tensor).
#          => 2 nccl dispatches per iter.
#   step2: the meta-AG is FUSED into the Triton _fused_allgather_reduce_kernel
#          (peer reads, no nccl), so the ONLY nccl kernel left is the query-AG.
#          => 1 nccl dispatch per iter -- collective bucket == query-AG cost.
# Therefore:  meta-AG cost = rccl.collective - step2.collective  (query-AG is
# common to both). CRITICAL: step2 MUST run with SIGNAL_PAD=1, else the merge
# barrier is a host dist.barrier (an nccl collective) and pollutes the bucket.
if [[ "$PROFILE_VARIANT" == "step2" ]]; then
    if [[ "${SIGNAL_PAD}" != "1" ]]; then
        echo "  [note] forcing SIGNAL_PAD=1 for step2 (keeps nccl bucket == query-AG only)"
        SIGNAL_PAD=1
    fi
fi

# Filenames carry the variant so step2 traces never overwrite the rccl ones.
OUT_DIR="${OUT_DIR:-/workspace/vllm/ktime_${MODEL}_${MODE}_${PROFILE_VARIANT}_${TS}}"
CSV="${CSV:-/workspace/vllm/ktime_${MODEL}_${MODE}_${PROFILE_VARIANT}.csv}"

# Optional per-dimension head overrides (else --model preset wins).
HEAD_ARGS=()
[[ -n "${Q_HEADS:-}" ]]   && HEAD_ARGS+=(--total-q-heads "$Q_HEADS")
[[ -n "${KV_HEADS:-}" ]]  && HEAD_ARGS+=(--total-kv-heads "$KV_HEADS")
[[ -n "${HEAD_SIZE:-}" ]] && HEAD_ARGS+=(--head-size "$HEAD_SIZE")

mkdir -p "$OUT_DIR"
rm -f "$CSV"

echo "=================================================================="
echo " Starscream kernel-time breakdown (rocprofv3 --kernel-trace)"
echo "   mode / model : $MODE / $MODEL   (bench --mode $BENCH_MODE)"
echo "   variant      : $PROFILE_VARIANT"
echo "   devices      : $DEVICES   (nproc=$NPROC)"
echo "   seq_lens     : $SEQ_LENS"
echo "   batch_sizes  : $BATCH_SIZES"
echo "   num_segments : $NUM_SEGMENTS   profile_iters: $PROFILE_ITERS"
echo "   out dir      : $OUT_DIR"
echo "   csv          : $CSV"
echo "   Reminder: hardware must be in $HW_PART partition mode."
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
        CELL="$OUT_DIR/S${S}_B${B}"
        mkdir -p "$CELL"
        P="$(find_port)"

        COMMON=(
            --mode "$BENCH_MODE" --model "$MODEL" "${HEAD_ARGS[@]}"
            --profile --profile-seq-len "$S" --profile-batch "$B"
            --profile-variant "$PROFILE_VARIANT" --profile-iters "$PROFILE_ITERS"
            --skip-sanity
        )
        [[ "$BENCH_MODE" == "cpx" ]] && COMMON+=(--cpx-size "$CPX_SIZE")

        env_prefix=(CUDA_VISIBLE_DEVICES="$DEVICES" PYTHONPATH="$VLLM_SRC"
                    VLLM_STARSCREAM_NUM_SEGMENTS="$NUM_SEGMENTS")
        if [[ "$BENCH_MODE" == "cpx" ]]; then
            env_prefix+=(TORCH_SYMM_MEM_DISABLE_MULTICAST=1
                         VLLM_STARSCREAM_SIGNAL_PAD_BARRIER="$SIGNAL_PAD")
        fi

        echo ""
        echo "--- kernel-trace S=$S B=$B (variant=$PROFILE_VARIANT) ---"
        env "${env_prefix[@]}" \
            rocprofv3 --kernel-trace -d "$CELL" \
                -- torchrun --nnodes=1 --nproc-per-node="$NPROC" \
                   --rdzv_endpoint="localhost:$P" \
                   "$BENCH" "${COMMON[@]}" \
            2>&1 | tee "$CELL/rocprof.log" || {
                echo "  rocprofv3 failed for S=$S B=$B"; continue; }

        NDB="$(find "$CELL" -name '*.db' 2>/dev/null | wc -l)"
        if [[ "$NDB" -eq 0 ]]; then
            echo "  no *.db found; skipping parse"; continue
        fi

        # Parse per-kernel durations, bucket into attention / collective /
        # merge / barrier / other, take MAX across ranks per bucket (critical
        # path = slowest XCD), append one row to CSV.
        MODE="$MODE" SEQ="$S" BATCH="$B" VARIANT="$PROFILE_VARIANT" \
        CELL="$CELL" CSV="$CSV" ITERS="$PROFILE_ITERS" \
        python3 - <<'PYEOF'
import csv, glob, os, sqlite3

mode=os.environ["MODE"]; seq=int(os.environ["SEQ"]); batch=int(os.environ["BATCH"])
variant=os.environ["VARIANT"]; cell=os.environ["CELL"]; out=os.environ["CSV"]
iters=int(os.environ["ITERS"])

def resolve(con,*cands):
    have={r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view')")}
    for c in cands:
        if c in have: return c
    for c in cands:
        for n in have:
            if n.startswith(c+"_"): return n
    return None

def bucket(name):
    n=(name or "").lower()
    if "unified_attention" in n: return "attention"
    if "nccldevkernel" in n or "rccl" in n: return "collective"
    if "signal_pad_barrier" in n: return "barrier"
    if "reduce_segments" in n or "fused_allgather_reduce" in n: return "merge"
    if "fillbuffer" in n or "copybuffer" in n: return "setup"
    return "other"

# Aggregate per db (per rank): sum durations per bucket. Then across dbs take
# MAX per bucket (critical path = slowest XCD, incl. collective straggler wait).
dbs=sorted(glob.glob(os.path.join(cell,"**","*.db"),recursive=True))
per_db=[]
for db in dbs:
    con=sqlite3.connect(db)
    disp=resolve(con,"rocpd_kernel_dispatch")
    ksym=resolve(con,"kernel_symbols","rocpd_info_kernel_symbol")
    if not (disp and ksym): con.close(); continue
    cols=[r[1] for r in con.execute(f'PRAGMA table_info("{ksym}")')]
    ncol=next((c for c in ("kernel_name","formatted_kernel_name","display_name","name") if c in cols),None)
    icol=next((c for c in ("id","kernel_id") if c in cols),None)
    b={}; b["_nccl_disp"]=0
    q=(f'SELECT ks."{ncol}", SUM(kd."end"-kd.start), COUNT(*) '
       f'FROM {disp} kd JOIN {ksym} ks ON kd.kernel_id=ks."{icol}" '
       f'GROUP BY ks."{ncol}"')
    for name,tot,cnt in con.execute(q):
        bk=bucket(name)
        b[bk]=b.get(bk,0.0)+float(tot or 0.0)
        if bk=="collective":
            b["_nccl_disp"]+=int(cnt or 0)
    per_db.append(b); con.close()

buckets=["attention","collective","merge","barrier","setup","other"]
agg={k:0.0 for k in buckets}
for k in buckets:
    agg[k]=max((d.get(k,0.0) for d in per_db), default=0.0)  # max across ranks
# nccl dispatches PER ITER on the busiest rank (rccl->~2, step2->~1). Validates
# that the collective bucket is isolated to query-AG for step2.
max_nccl=max((d.get("_nccl_disp",0) for d in per_db), default=0)
nccl_per_iter = (max_nccl/iters) if iters else 0
# per-iter (ns) -> us
def us(x): return (x/iters)/1e3
total_us=sum(us(agg[k]) for k in buckets)

hdr=("mode,variant,seq_len,batch,n_dbs,"
     "attention_us,collective_us,merge_us,barrier_us,setup_us,other_us,total_us,"
     "nccl_disp_per_iter\n")
newf=not os.path.exists(out)
with open(out,"a") as f:
    if newf: f.write(hdr)
    f.write(f"{mode},{variant},{seq},{batch},{len(per_db)},"
            f"{us(agg['attention']):.3f},{us(agg['collective']):.3f},"
            f"{us(agg['merge']):.3f},{us(agg['barrier']):.3f},"
            f"{us(agg['setup']):.3f},{us(agg['other']):.3f},{total_us:.3f},"
            f"{nccl_per_iter:.2f}\n")
print(f"  [ktime] attn={us(agg['attention']):.1f} coll={us(agg['collective']):.1f} "
      f"merge={us(agg['merge']):.1f} barrier={us(agg['barrier']):.1f} us "
      f"nccl/iter={nccl_per_iter:.1f} (max over {len(per_db)} ranks)")
PYEOF
    done
done

echo ""
echo "=================================================================="
echo " Kernel-time breakdown complete. CSV: $CSV"
echo "=================================================================="
[[ -f "$CSV" ]] && cat "$CSV"
