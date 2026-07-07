# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
End-to-end attention benchmark: SPX vs CPX+NPS4 for Starscream (DeepSeek-R1).

Measures wall-clock of the *decode attention step* (start to end of attention)
under two hardware compute-partition modes:

  * SPX : 8 physical GPUs, each GPU does attention over the FULL context.
          Standard path (enable_starscream=False, no cross-XCD merge).

  * CPX : 8 physical GPUs in CPX+NPS4 = 64 vGPUs (XCDs). Each XCD does
          attention over 1/cpx of the context, then merges across the XCDs of
          its physical GPU. The merge is benchmarked in 3 variants:
            - rccl  : RCCL/NCCL allgather -> reduce_segments   (stock)
            - step1 : symm-mem allgather   -> reduce_segments   (collective swap)
            - step2 : fused symm-mem allgather + reduce         (single kernel)

The thesis: CPX partitioning cuts per-XCD attention compute ~cpx x, but makes
the cross-XCD communication explicit. RCCL is a poor fit for intra-GPU peer
traffic and can bottleneck; symmetric memory (step1/step2) gives CPX a
performant merge so it stays competitive with (or beats) SPX.

IMPORTANT
---------
* SPX and CPX are HARDWARE partition modes set outside this script
  (`rocm-smi --setcomputepartition SPX|CPX`). Run the benchmark ONCE PER MODE,
  each in the matching hardware config, and compare the CSVs offline.
* --mode must match the hardware: the script checks the visible device count
  (SPX => nproc == physical GPU count; CPX => nproc == 8 * physical GPUs) and
  refuses to run on a mismatch.
* Wall-clock is reduced with MAX across ranks: the slowest XCD is the critical
  path (its attention + merge gates the physical GPU).
* On OOM for a (batch, seq_len) cell, ALL ranks skip that cell together (via an
  all-reduce of the alloc-success flag) to avoid a collective hang; the cell is
  recorded as SKIP(OOM). No extrapolation.

Run -- SPX mode (8 GPUs, hardware in SPX):
  rocm-smi --setcomputepartition SPX
  PYTHONPATH=/workspace/vllm \
  torchrun --nnodes=1 --nproc-per-node=8 \
    tests/kernels/attention/bench_starscream_attention.py \
    --mode spx --total-q-heads 128 --total-kv-heads 128 --head-size 128 \
    --seq-lens 256,8192,131072 --batch-sizes 1,8,32,64,128,256,512,1024 \
    --csv /workspace/bench_spx.csv

Run -- CPX mode (64 vGPUs, hardware in CPX+NPS4):
  rocm-smi --setcomputepartition CPX
  PYTHONPATH=/workspace/vllm \
  TORCH_SYMM_MEM_DISABLE_MULTICAST=1 \
  torchrun --nnodes=1 --nproc-per-node=64 \
    tests/kernels/attention/bench_starscream_attention.py \
    --mode cpx --cpx-size 8 --total-q-heads 128 --total-kv-heads 128 --head-size 128 \
    --seq-lens 256,8192,131072 --batch-sizes 1,8,32,64,128,256,512,1024 \
    --csv /workspace/bench_cpx.csv
"""

import argparse
import math
import os

import torch
import torch.distributed as dist
import triton

# ---------------------------------------------------------------------------
# Distributed bootstrap
# ---------------------------------------------------------------------------
_rank       = int(os.environ.get("RANK", 0))
_local_rank = int(os.environ.get("LOCAL_RANK", 0))
_world_size = int(os.environ.get("WORLD_SIZE", 1))

torch.cuda.set_device(_local_rank)

from vllm.distributed.parallel_state import (  # noqa: E402
    init_distributed_environment,
    initialize_model_parallel,
    get_dcp_group,
    destroy_model_parallel,
)

init_distributed_environment(
    world_size=_world_size,
    rank=_rank,
    local_rank=_local_rank,
    distributed_init_method="env://",
    backend="nccl",
)

try:
    import torch.distributed._symmetric_memory as _symm_mem
    _symm_mem.enable_symm_mem_for_group(dist.group.WORLD.group_name)
except Exception:
    pass

from vllm.attention.ops.triton_unified_attention import unified_attention  # noqa: E402


# Informational scenario labels (from the DeepSeek-R1 workload table), keyed by
# seq_len. Purely for annotating output rows.
_SCENARIO_BY_SEQLEN = {
    256:     "ShortChat/QA",
    8192:    "Summ/Reason",
    131072:  "Writing/LargeCoding",
    2621440: "AdvCoding",
}


def _log(msg: str) -> None:
    if _rank == 0:
        print(msg, flush=True)


def _all_reduce_max(val: float) -> float:
    t = torch.tensor([val], dtype=torch.float64, device=f"cuda:{_local_rank}")
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.item()


def _all_reduce_min_int(val: int) -> int:
    t = torch.tensor([val], dtype=torch.int64, device=f"cuda:{_local_rank}")
    dist.all_reduce(t, op=dist.ReduceOp.MIN)
    return int(t.item())


def _set_variant_env(variant: str) -> None:
    """Toggle the merge implementation via env flags (read fresh each call)."""
    if variant == "rccl":
        os.environ["VLLM_STARSCREAM_USE_SYMM_MEM"] = "0"
        os.environ["VLLM_STARSCREAM_FUSE_REDUCE"] = "0"
    elif variant == "step1":
        os.environ["VLLM_STARSCREAM_USE_SYMM_MEM"] = "1"
        os.environ["VLLM_STARSCREAM_FUSE_REDUCE"] = "0"
    elif variant == "step2":
        os.environ["VLLM_STARSCREAM_USE_SYMM_MEM"] = "0"
        os.environ["VLLM_STARSCREAM_FUSE_REDUCE"] = "1"
    else:
        raise ValueError(variant)


def _try_alloc(mode, B, S, qh_dev, kvh, head_size, block_size, cpx, device):
    """Allocate q/kv-cache/block-table for one (B, S) cell.

    Returns (ok: bool, tensors: dict|None). ok=False on OOM.
    In CPX, qh_dev is the GATHERED query-head count; local heads = qh_dev//cpx
    and the per-XCD context is ceil(S/cpx).
    """
    try:
        dtype = torch.bfloat16
        local_ctx = S if mode == "spx" else math.ceil(S / cpx)
        npb = math.ceil(local_ctx / block_size)          # blocks per sequence
        total_blocks = B * npb

        k_cache = torch.randn(total_blocks, block_size, kvh, head_size,
                              dtype=dtype, device=device)
        v_cache = torch.randn(total_blocks, block_size, kvh, head_size,
                              dtype=dtype, device=device)

        block_table = torch.arange(total_blocks, dtype=torch.int32,
                                   device=device).reshape(B, npb)

        # Decode: 1 query token per sequence.
        cu_seqlens_q = torch.arange(B + 1, dtype=torch.int32, device=device)
        seqused_k = torch.full((B,), S, dtype=torch.int32, device=device)

        if mode == "spx":
            q = torch.randn(B, qh_dev, head_size, dtype=dtype, device=device)
            out = torch.empty(B, qh_dev, head_size, dtype=dtype, device=device)
            q_local = None
        else:
            qh_local = qh_dev // cpx
            q_local = torch.randn(B, qh_local, head_size, dtype=dtype,
                                  device=device)
            out = torch.empty(B, qh_local, head_size, dtype=dtype, device=device)
            q = None  # built each iter via all_gather (faithful to backend)

        descale_shape = (B, kvh)
        k_descale = torch.ones(descale_shape, dtype=torch.float32, device=device)
        v_descale = torch.ones(descale_shape, dtype=torch.float32, device=device)

        torch.cuda.synchronize()
        return True, dict(
            k_cache=k_cache, v_cache=v_cache, block_table=block_table,
            cu_seqlens_q=cu_seqlens_q, seqused_k=seqused_k, q=q, q_local=q_local,
            out=out, k_descale=k_descale, v_descale=v_descale,
            qh_dev=qh_dev, kvh=kvh, head_size=head_size, S=S,
        )
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" in str(e).lower() or isinstance(
                e, torch.cuda.OutOfMemoryError):
            torch.cuda.empty_cache()
            return False, None
        raise


def _attention_call(mode, tn, grp, cpx, head_size):
    """One decode attention step (start to end of attention)."""
    if mode == "cpx":
        # Faithful to triton_attn.py forward: gather query heads across the XCDs
        # (RCCL, constant across merge variants), then run the starscream path.
        q = grp.all_gather(tn["q_local"].contiguous(), dim=-2)
        unified_attention(
            q=q, k=tn["k_cache"], v=tn["v_cache"], out=tn["out"],
            slice_idx=0, starscream_rank=grp.rank_in_group,
            cu_seqlens_q=tn["cu_seqlens_q"], max_seqlen_q=1,
            seqused_k=tn["seqused_k"], max_seqlen_k=tn["S"],
            softmax_scale=head_size ** -0.5, causal=True,
            window_size=(-1, -1), block_table=tn["block_table"],
            softcap=0, q_descale=None,
            k_descale=tn["k_descale"], v_descale=tn["v_descale"],
            cpx_size=cpx, enable_starscream=True,
        )
    else:
        unified_attention(
            q=tn["q"], k=tn["k_cache"], v=tn["v_cache"], out=tn["out"],
            slice_idx=0, starscream_rank=0,
            cu_seqlens_q=tn["cu_seqlens_q"], max_seqlen_q=1,
            seqused_k=tn["seqused_k"], max_seqlen_k=tn["S"],
            softmax_scale=head_size ** -0.5, causal=True,
            window_size=(-1, -1), block_table=tn["block_table"],
            softcap=0, q_descale=None,
            k_descale=tn["k_descale"], v_descale=tn["v_descale"],
            cpx_size=1, enable_starscream=False,
        )


def _bench_variant(mode, tn, grp, cpx, head_size, warmup, iters):
    """Median per-iter latency (us) for this rank, then MAX across ranks."""
    for _ in range(warmup):
        _attention_call(mode, tn, grp, cpx, head_size)
    torch.cuda.synchronize()
    dist.barrier()

    times_ms = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        dist.barrier()
        start.record()
        _attention_call(mode, tn, grp, cpx, head_size)
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))

    times_ms.sort()
    median_us = times_ms[len(times_ms) // 2] * 1000.0
    return _all_reduce_max(median_us)  # slowest XCD = critical path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["spx", "cpx"], required=True)
    p.add_argument("--cpx-size", type=int, default=8,
                   help="XCDs per physical GPU (CPX mode)")
    p.add_argument("--total-q-heads", type=int, default=128,
                   help="Total query heads across the model (DeepSeek-R1: 128)")
    p.add_argument("--total-kv-heads", type=int, default=128,
                   help="Total KV heads across the model")
    p.add_argument("--head-size", type=int, default=128)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--seq-lens", type=str, default="256,8192,131072")
    p.add_argument("--batch-sizes", type=str,
                   default="1,8,32,64,128,256,512,1024")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--csv", type=str, default="",
                   help="Optional path to write CSV results (rank 0)")
    p.add_argument("--skip-sanity", action="store_true",
                   help="Skip the pre-sweep sanity check")
    p.add_argument("--sanity-seq-len", type=int, default=8192,
                   help="seq_len for the pre-sweep sanity check")
    p.add_argument("--sanity-batch", type=int, default=1,
                   help="batch size for the pre-sweep sanity check")
    args = p.parse_args()

    # --- Mode / hardware consistency check ---
    if args.mode == "cpx":
        assert _world_size % args.cpx_size == 0, (
            f"CPX: WORLD_SIZE ({_world_size}) must be divisible by cpx-size "
            f"({args.cpx_size}). Expected 64 vGPUs for 8 physical GPUs.")
        num_physical = _world_size // args.cpx_size
        cpx = args.cpx_size
    else:
        num_physical = _world_size
        cpx = 1

    _log("")
    _log("=" * 100)
    _log(f"Starscream attention benchmark  --  MODE = {args.mode.upper()}")
    _log(f"  world_size (devices)   : {_world_size}"
         + ("  (vGPUs/XCDs)" if args.mode == "cpx" else "  (physical GPUs)"))
    _log(f"  physical GPUs          : {num_physical}")
    if args.mode == "cpx":
        _log(f"  cpx_size (XCDs/GPU)    : {cpx}   -> {num_physical} DCP groups")
    _log(f"  total q-heads / kv-heads: {args.total_q_heads} / {args.total_kv_heads}")
    _log(f"  head_size / block_size : {args.head_size} / {args.block_size}")
    _log(f"  warmup / iters         : {args.warmup} / {args.iters}  (median, MAX over ranks)")
    _log("=" * 100)

    # Per-device head counts. In CPX these are the GATHERED per-XCD heads
    # (== one physical GPU's share); local heads = qh_dev // cpx.
    qh_dev = args.total_q_heads // num_physical
    kvh = max(1, args.total_kv_heads // num_physical)
    assert qh_dev % cpx == 0, (
        f"gathered q-heads ({qh_dev}) must be divisible by cpx ({cpx})")
    assert qh_dev % kvh == 0, (
        f"q-heads ({qh_dev}) must be divisible by kv-heads ({kvh})")
    _log(f"  per-device: q-heads(gathered)={qh_dev}  kv-heads={kvh}"
         + (f"  local-q-heads={qh_dev // cpx}" if args.mode == "cpx" else ""))
    _log("")

    # TP == world_size; DCP == cpx (1 group per physical GPU in CPX; trivial in SPX).
    initialize_model_parallel(
        tensor_model_parallel_size=_world_size,
        pipeline_model_parallel_size=1,
        decode_context_model_parallel_size=cpx,
    )
    grp = get_dcp_group()
    device = torch.device(f"cuda:{_local_rank}")

    seq_lens = [int(x) for x in args.seq_lens.split(",")]
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    variants = ["rccl", "step1", "step2"] if args.mode == "cpx" else ["spx"]

    # --- Pre-sweep sanity check: run one small cell end-to-end so a broken
    # harness (bad shapes, symm-mem unavailable, collective hang) fails fast
    # before the full sweep allocates large KV caches. ---
    if not args.skip_sanity:
        sb, ss = args.sanity_batch, args.sanity_seq_len
        _log(f"[sanity] running one cell: batch={sb} seq_len={ss} "
             f"variants={variants} ...")
        ok, tn = _try_alloc(args.mode, sb, ss, qh_dev, kvh, args.head_size,
                            args.block_size, cpx, device)
        ok_all = _all_reduce_min_int(1 if ok else 0)
        if not ok_all:
            if ok:
                del tn
                torch.cuda.empty_cache()
            _log(f"[sanity] FAILED: OOM allocating batch={sb} seq_len={ss}. "
                 "Lower --sanity-seq-len/--sanity-batch or fix memory, "
                 "or pass --skip-sanity.")
            destroy_model_parallel()
            dist.destroy_process_group()
            return
        for v in variants:
            if args.mode == "cpx":
                _set_variant_env(v)
            us = _bench_variant(args.mode, tn, grp, cpx, args.head_size,
                                max(1, args.warmup // 2), max(2, args.iters // 5))
            _log(f"[sanity]   {v:>6}: {us:.2f} us")
        del tn
        torch.cuda.empty_cache()
        _log("[sanity] OK -- harness works; starting full sweep.\n")

    # Table header
    var_cols = "".join(f"{v + '_us':>14}" for v in variants)
    _log(f"{'seq_len':>9} {'batch':>7}{var_cols}   scenario")
    _log("-" * 100)

    csv_rows = []  # (mode, seq_len, batch, variant, us)  or  us=NaN for SKIP

    for S in seq_lens:
        scen = _SCENARIO_BY_SEQLEN.get(S, "")
        for B in batch_sizes:
            ok, tn = _try_alloc(args.mode, B, S, qh_dev, kvh, args.head_size,
                                args.block_size, cpx, device)
            ok_all = _all_reduce_min_int(1 if ok else 0)
            if not ok_all:
                if ok:  # this rank allocated but a peer OOM'd -- free it
                    del tn
                    torch.cuda.empty_cache()
                skip_cols = "".join(f"{'SKIP(OOM)':>14}" for _ in variants)
                _log(f"{S:>9} {B:>7}{skip_cols}   {scen}")
                for v in variants:
                    csv_rows.append((args.mode, S, B, v, float("nan")))
                continue

            results = {}
            for v in variants:
                if args.mode == "cpx":
                    _set_variant_env(v)
                us = _bench_variant(args.mode, tn, grp, cpx, args.head_size,
                                    args.warmup, args.iters)
                results[v] = us
                csv_rows.append((args.mode, S, B, v, us))

            val_cols = "".join(f"{results[v]:>14.2f}" for v in variants)
            _log(f"{S:>9} {B:>7}{val_cols}   {scen}")

            del tn
            torch.cuda.empty_cache()

    _log("=" * 100)

    # CSV output (rank 0)
    if args.csv and _rank == 0:
        with open(args.csv, "w") as f:
            f.write("mode,seq_len,batch,variant,latency_us\n")
            for mode, S, B, v, us in csv_rows:
                us_str = "" if us != us else f"{us:.4f}"  # NaN -> empty
                f.write(f"{mode},{S},{B},{v},{us_str}\n")
        _log(f"CSV written to {args.csv}")
    _log("")

    destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
