#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Minimal probe: does forcing the 2D attention kernel help CPX-Starscream step2
# in the small/mid-batch regime? Runs ONLY a handful of cells per model
# (the underperformers + 1-2 performant anchors) and emits latency for
# FORCE_2D=1 alongside the existing FORCE_2D=0 (3D) numbers you already have.
#
# STEP 0 (do this first): correctness gate. Nothing below is trustworthy until
# 2D output == 3D output on the CPX setup:
#   TORCH_SYMM_MEM_DISABLE_MULTICAST=1 \
#   torchrun --nnodes=1 --nproc-per-node=8 \
#     tests/kernels/attention/test_force2d_parity.py --cpx-size 8 \
#     --seq-len 16384 --batches 32,64
#
# Usage (CPX hardware):
#   ./run_force2d_probe.sh [model]     # default llama3-70b
#
# The probe cells (seq,batch): valley = (16384,32) (16384,64) (32768,64);
# anchors = (65536,1) (131072,256). Override with CELLS="s:b,s:b,...".
#
# Env: VLLM_SRC, MODEL, CPX_SIZE, CELLS, WARMUP, ITERS, CUDA_GRAPH, CSV.

set -euo pipefail

MODEL="${1:-${MODEL:-llama3-70b}}"
MODEL="$(echo "$MODEL" | tr '[:upper:]' '[:lower:]')"
VLLM_SRC="${VLLM_SRC:-/workspace/vllm}"
CPX_SIZE="${CPX_SIZE:-8}"
WARMUP="${WARMUP:-25}"
ITERS="${ITERS:-200}"
CUDA_GRAPH="${CUDA_GRAPH:-1}"
# probe cells: "seq:batch" list. Underperformers + a couple anchors.
CELLS="${CELLS:-16384:32,16384:64,32768:64,65536:1,131072:256}"
CSV="${CSV:-/workspace/vllm/force2d_${MODEL}.csv}"

BENCH="${VLLM_SRC}/tests/kernels/attention/bench_starscream_attention.py"
DEVICES="${DEVICES:-0,1,2,3,4,5,6,7}"
rm -f "$CSV"

echo "=================================================================="
echo " FORCE_2D probe: $MODEL  (CPX step2 only)"
echo "   cells (seq:batch): $CELLS"
echo "   warmup/iters=$WARMUP/$ITERS  cuda_graph=$CUDA_GRAPH"
echo "   csv: $CSV"
echo "   Reminder: hardware must be in CPX; run the parity gate first."
echo "=================================================================="

find_port(){ local p=29500; while ss -tuln 2>/dev/null|grep -q ":$p "; do p=$((p+1)); [[ $p -gt 30500 ]]&&p=29600; done; echo "$p"; }

graph_arg="--cuda-graph"; [[ "$CUDA_GRAPH" == "0" ]] && graph_arg="--no-cuda-graph"

for f2d in 0 1; do
  for cell in ${CELLS//,/ }; do
    S="${cell%%:*}"; B="${cell##*:}"
    P="$(find_port)"
    tag="3D"; [[ "$f2d" == "1" ]] && tag="2D"
    echo ""
    echo "--- $MODEL  seq=$S batch=$B  kernel=$tag (FORCE_2D=$f2d) ---"
    # one CSV per (f2d); we tag rows via a distinct CSV then merge. Simpler:
    # write to a per-variant temp csv and append with a kernel column.
    tmp="${CSV}.tmp"
    CUDA_VISIBLE_DEVICES="$DEVICES" \
    PYTHONPATH="$VLLM_SRC" \
    TORCH_SYMM_MEM_DISABLE_MULTICAST=1 \
    VLLM_STARSCREAM_SIGNAL_PAD_BARRIER=1 \
    VLLM_STARSCREAM_FORCE_2D="$f2d" \
    torchrun --nnodes=1 --nproc-per-node="$CPX_SIZE" \
        --rdzv_endpoint="localhost:$P" \
        "$BENCH" --mode cpx --model "$MODEL" --cpx-size "$CPX_SIZE" \
        --seq-lens "$S" --batch-sizes "$B" --token-offsets 0 \
        --warmup "$WARMUP" --iters "$ITERS" $graph_arg \
        --skip-sanity --csv "$tmp"
    # append step2 rows with a kernel tag to the master csv (rank 0 wrote tmp)
    if [[ -f "$tmp" ]]; then
      python3 - "$tmp" "$CSV" "$tag" "$MODEL" <<'PYEOF'
import csv,sys,os
tmp,master,tag,model=sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4]
rows=[r for r in csv.DictReader(open(tmp))
      if r.get("variant")=="step2" and r.get("row_type")=="raw" and r.get("latency_us")]
newf=not os.path.exists(master)
with open(master,"a") as f:
    if newf: f.write("model,kernel,seq_len,batch,step2_us\n")
    for r in rows:
        f.write(f"{model},{tag},{r['base_seq_len']},{r['batch']},{r['latency_us']}\n")
os.remove(tmp)
PYEOF
    fi
  done
done

echo ""
echo "=================================================================="
echo " FORCE_2D probe complete: $CSV"
echo "=================================================================="
[[ -f "$CSV" ]] && cat "$CSV"
