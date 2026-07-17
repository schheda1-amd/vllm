#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Tiling sweep for the CPX-Starscream 3D decode-attention GEMMs (QKt / PV).
# Varies the split-KV tiling knobs to size the grid to the XCD's 38 CUs with
# MFMA-sweet-spot tiles, and measures CPX-step2 latency vs the defaults.
#
#   NUM_SEGMENTS : context (key) split. default 16. GEMM-2 split-K factor.
#   BLOCK_Q      : query tokens per threadblock. BLOCK_M = BLOCK_Q*num_q_per_kv
#                  = the M dim of BOTH gemms. default 2 (BLOCK_M=16). Raise to
#                  push M toward 64 (MFMA sweet spot).
#   TILE_SIZE    : key tile. GEMM-1 N / GEMM-2 K. default 16. Raise for fatter
#                  K/V loads.
#
# Only the 3D-kernel regime is meaningful (batch below the 2D switch, ~<86).
# batch=128 crosses to the 2D kernel where these knobs are ignored -- included
# only as a boundary marker.
#
# Usage (CPX hardware):
#   ./run_tiling_sweep.sh [model]      # default llama3-70b
#
# Env: VLLM_SRC, MODEL, SEQ_LENS, BATCH_SIZES, CONFIGS, CPX_SIZE, WARMUP, ITERS,
#      CUDA_GRAPH, CSV.
#   CONFIGS = semicolon list of "seg,blockq,tile". Include "16,2,16" as the
#   default baseline for direct comparison. Example:
#     CONFIGS="16,2,16;8,4,32;4,8,32;4,8,64;2,8,64"

set -euo pipefail

MODEL="${1:-${MODEL:-llama3-70b}}"
MODEL="$(echo "$MODEL" | tr '[:upper:]' '[:lower:]')"
VLLM_SRC="${VLLM_SRC:-/workspace/vllm}"
SEQ_LENS="${SEQ_LENS:-16384,32768,65536}"
BATCH_SIZES="${BATCH_SIZES:-1,8,32,64,128}"
CPX_SIZE="${CPX_SIZE:-8}"
WARMUP="${WARMUP:-25}"
ITERS="${ITERS:-200}"
CUDA_GRAPH="${CUDA_GRAPH:-1}"
# "seg,blockq,tile" triples. First is the default baseline.
CONFIGS="${CONFIGS:-16,2,16;8,4,16;8,4,32;4,8,32;4,8,64;2,8,64}"
CSV="${CSV:-/workspace/vllm/tiling_${MODEL}.csv}"

BENCH="${VLLM_SRC}/tests/kernels/attention/bench_starscream_attention.py"
DEVICES="${DEVICES:-0,1,2,3,4,5,6,7}"
rm -f "$CSV"
graph_arg="--cuda-graph"; [[ "$CUDA_GRAPH" == "0" ]] && graph_arg="--no-cuda-graph"

echo "=================================================================="
echo " Tiling sweep: $MODEL  (CPX step2)"
echo "   seq_lens=$SEQ_LENS  batch_sizes=$BATCH_SIZES"
echo "   configs (seg,blockq,tile): $CONFIGS"
echo "   warmup/iters=$WARMUP/$ITERS  cuda_graph=$CUDA_GRAPH"
echo "   csv: $CSV"
echo "   Reminder: CPX hardware; batch>=128 crosses to 2D (knobs ignored)."
echo "=================================================================="

find_port(){ local p=29500; while ss -tuln 2>/dev/null|grep -q ":$p "; do p=$((p+1)); [[ $p -gt 30500 ]]&&p=29600; done; echo "$p"; }

IFS=';' read -ra CFGS <<< "$CONFIGS"
for cfg in "${CFGS[@]}"; do
    IFS=',' read -r SEG BQ TS <<< "$cfg"
    P="$(find_port)"
    echo ""
    echo "--- config: NUM_SEGMENTS=$SEG BLOCK_Q=$BQ TILE_SIZE=$TS ---"
    tmp="${CSV}.tmp"
    CUDA_VISIBLE_DEVICES="$DEVICES" \
    PYTHONPATH="$VLLM_SRC" \
    TORCH_SYMM_MEM_DISABLE_MULTICAST=1 \
    VLLM_STARSCREAM_SIGNAL_PAD_BARRIER=1 \
    VLLM_STARSCREAM_NUM_SEGMENTS="$SEG" \
    VLLM_STARSCREAM_BLOCK_Q="$BQ" \
    VLLM_STARSCREAM_TILE_SIZE="$TS" \
    torchrun --nnodes=1 --nproc-per-node="$CPX_SIZE" \
        --rdzv_endpoint="localhost:$P" \
        "$BENCH" --mode cpx --model "$MODEL" --cpx-size "$CPX_SIZE" \
        --seq-lens "$SEQ_LENS" --batch-sizes "$BATCH_SIZES" --token-offsets 0 \
        --warmup "$WARMUP" --iters "$ITERS" $graph_arg \
        --skip-sanity --csv "$tmp" || { echo "  run failed for $cfg"; continue; }

    if [[ -f "$tmp" ]]; then
      python3 - "$tmp" "$CSV" "$MODEL" "$SEG" "$BQ" "$TS" <<'PYEOF'
import csv,sys,os
tmp,master,model,seg,bq,ts=sys.argv[1:7]
rows=[r for r in csv.DictReader(open(tmp))
      if r.get("variant")=="step2" and r.get("row_type")=="raw" and r.get("latency_us")]
newf=not os.path.exists(master)
with open(master,"a") as f:
    if newf: f.write("model,num_segments,block_q,tile_size,seq_len,batch,step2_us\n")
    for r in rows:
        f.write(f"{model},{seg},{bq},{ts},{r['base_seq_len']},{r['batch']},{r['latency_us']}\n")
os.remove(tmp)
PYEOF
    fi
done

echo ""
echo "=================================================================="
echo " Tiling sweep complete: $CSV"
echo "=================================================================="
[[ -f "$CSV" ]] && cat "$CSV"
