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


def _trapezoidal_per_token(offsets, lats):
    """Per-output-token attention latency via trapezoidal integration.

    `offsets` are output-token offsets (sorted ascending), `lats` the measured
    decode-step latency (us) at each. Decode attention cost is ~linear in
    context, so we integrate latency over the offset span and divide by that
    span -- an average that is unbiased for non-uniform offset spacing (unlike a
    plain mean, which over-weights whichever region is sampled more densely).

    Single point -> that point (nothing to integrate). Any NaN (a SKIP(OOM)
    among the offsets) -> NaN, since the curve is incomplete.
    """
    pts = [(o, l) for o, l in zip(offsets, lats)]
    if any(l != l for _, l in pts):  # NaN present
        return float("nan")
    if len(pts) == 1:
        return pts[0][1]
    span = pts[-1][0] - pts[0][0]
    if span == 0:
        return sum(l for _, l in pts) / len(pts)
    area = 0.0
    for (o0, l0), (o1, l1) in zip(pts[:-1], pts[1:]):
        area += 0.5 * (l0 + l1) * (o1 - o0)
    return area / span


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


def _bench_variant_eager(mode, tn, grp, cpx, head_size, warmup, iters):
    """Eager-mode timing. Median per-iter latency (us), then MAX across ranks.

    NOTE: each timed iteration includes host-side kernel-launch overhead. For
    microsecond-scale decode ops this can dominate and distort the cross-variant
    comparison; prefer the CUDA-graph path when capturable.
    """
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


def _bench_variant_graph(mode, tn, grp, cpx, head_size, warmup, iters):
    """CUDA-graph timing. Captures one attention call, times `iters` replays.

    Removes host launch overhead from the measurement so the numbers reflect
    real GPU/comm cost — the axis the SPX-vs-CPX / rccl-vs-step1-vs-step2
    comparison rests on. Raises if the region is not capturable (e.g. a variant
    that still uses a host dist.barrier); the caller falls back to eager.

    Returns median per-replay latency (us), MAX over ranks.

    CRITICAL: each replay is timed INDIVIDUALLY with a cross-rank barrier before
    it and a full synchronize after it. This is required for correctness in CPX
    mode, where the graph contains cross-XCD collectives (query all-gather +
    merge). If we instead timed a block of back-to-back replays with no per-iter
    sync, the 8 XCDs would pipeline/overlap across iterations and the collective
    latency would hide under adjacent iterations' compute -- making CPX look
    artificially fast. Per-replay sync forces every decode step to fully
    complete on all ranks before the next starts, so each sample is one true
    synchronized decode step. Graph capture still removes host launch overhead
    from within the step; only the (real) per-step comm latency remains.
    """
    call = lambda: _attention_call(mode, tn, grp, cpx, head_size)

    # Warmup on a side stream (required by torch before graph capture) — this
    # also settles Triton JIT and NCCL/symm-mem setup so capture is clean.
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            call()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    # Capture. If the region contains an uncapturable op (host dist.barrier),
    # this raises and the caller falls back to eager.
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        call()

    # Warmup replays (excluded from timing).
    for _ in range(warmup):
        g.replay()
    torch.cuda.synchronize()
    dist.barrier()

    # Time each replay individually; barrier-align ranks before each so no
    # cross-iteration overlap can hide collective latency.
    times_ms = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        dist.barrier()
        start.record()
        g.replay()
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))

    times_ms.sort()
    median_us = times_ms[len(times_ms) // 2] * 1000.0
    return _all_reduce_max(median_us)  # slowest XCD = critical path


def _bench_variant(mode, tn, grp, cpx, head_size, warmup, iters, use_graph):
    """Dispatch to graph or eager timing, with graph->eager fallback."""
    if use_graph:
        try:
            return _bench_variant_graph(
                mode, tn, grp, cpx, head_size, warmup, iters)
        except Exception as e:  # noqa: BLE001 - capture may fail on ROCm/barrier
            _log(f"    [graph capture failed, falling back to eager: "
                 f"{type(e).__name__}: {e}]")
    return _bench_variant_eager(mode, tn, grp, cpx, head_size, warmup, iters)


# Per-GPU attention shard shapes under TP=8 (single physical GPU focus).
# Full-model heads / TP (all GQA with 8 KV heads):
#   Llama3-70B  : 64 q-heads,  8 kv-heads, head_dim 128 -> /8 -> 8 q,  1 kv, 128
#   Llama3-405B : 128 q-heads, 8 kv-heads, head_dim 128 -> /8 -> 16 q, 1 kv, 128
#   GPT-OSS-120B: 64 q-heads,  8 kv-heads, head_dim  64 -> /8 -> 8 q,  1 kv, 64
_MODEL_CONFIGS = {
    "llama3-70b":  {"q_heads": 8,  "kv_heads": 1, "head_size": 128},
    "llama3-405b": {"q_heads": 16, "kv_heads": 1, "head_size": 128},
    "gpt-oss-120b": {"q_heads": 8, "kv_heads": 1, "head_size": 64},
}


def _run_profile_mode(args, grp, cpx, qh_dev, kvh, device):
    """Execute ONE shape / ONE variant N times for an external profiler.

    No timing, no CSV: the caller wraps this process in rocprofv3, which
    attributes memory counters to the attention kernels launched here. We just
    allocate the single (seq_len, batch) cell, select the variant via the same
    env flags the sweep uses, warm up (JIT + first-touch, excluded because the
    profiler counts all dispatches -- warmup is fine, it inflates counts
    uniformly and we report bandwidth = bytes/time which is warmup-invariant),
    then run --profile-iters attention calls.
    """
    S = args.profile_seq_len
    B = args.profile_batch
    variant = args.profile_variant or ("spx" if args.mode == "spx" else "step2")

    _log(f"[profile] mode={args.mode} variant={variant} seq_len={S} batch={B} "
         f"iters={args.profile_iters}")

    ok, tn = _try_alloc(args.mode, B, S, qh_dev, kvh, args.head_size,
                        args.block_size, cpx, device)
    ok_all = _all_reduce_min_int(1 if ok else 0)
    if not ok_all:
        if ok:
            del tn
            torch.cuda.empty_cache()
        _log(f"[profile] SKIP(OOM) seq_len={S} batch={B}")
        return

    if args.mode == "cpx":
        _set_variant_env(variant)

    # Warmup (JIT / first-touch / rendezvous) BEFORE the profiled region would
    # ideally be excluded, but rocprofv3 counts the whole process. Since we
    # report bytes/time (an intensity ratio, not an absolute), a few warmup
    # dispatches do not bias it. Keep warmup small.
    for _ in range(3):
        _attention_call(args.mode, tn, grp, cpx, args.head_size)
    torch.cuda.synchronize()
    dist.barrier()

    for _ in range(args.profile_iters):
        _attention_call(args.mode, tn, grp, cpx, args.head_size)
    torch.cuda.synchronize()
    dist.barrier()
    _log(f"[profile] done: {args.profile_iters} calls of {variant}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["spx", "cpx"], required=True)
    p.add_argument("--model", choices=sorted(_MODEL_CONFIGS.keys()),
                   default=None,
                   help="If set, overrides --total-q-heads/--total-kv-heads/"
                        "--head-size with that model's per-GPU TP=8 attention "
                        "shard. Llama3-70b -> 8q/1kv/128; "
                        "Llama3-405b -> 16q/1kv/128.")
    p.add_argument("--cpx-size", type=int, default=8,
                   help="XCDs per physical GPU (CPX mode)")
    # These default to None so we can tell whether the user set them explicitly.
    # Resolution order: model preset -> explicit arg (if given) -> llama3-70b
    # fallback when neither model nor arg is provided.
    p.add_argument("--total-q-heads", type=int, default=None,
                   help="Query heads for this device set (per-GPU TP=8 shard). "
                        "Overrides the --model preset if given. Falls back to "
                        "llama3-70b (8) if neither --model nor this is set.")
    p.add_argument("--total-kv-heads", type=int, default=None,
                   help="KV heads for this device set. Overrides --model if "
                        "given. Falls back to llama3-70b (1).")
    p.add_argument("--head-size", type=int, default=None,
                   help="Head dimension. Overrides --model if given. Falls "
                        "back to llama3-70b (128).")
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--seq-lens", type=str, default="256,8192,131072",
                   help="Base context lengths (KV cache size at the START of "
                        "generation, ~= input tokens).")
    p.add_argument("--token-offsets", type=str, default="0",
                   help="Output-token offsets applied to each base seq-len. "
                        "For base S and offset o, we measure a decode step at "
                        "context S+o, i.e. the cost of generating output token "
                        "o+1. E.g. '0,64,128,192,256' samples the first 257 "
                        "output tokens. The per-output-token attention latency "
                        "is the trapezoidal integral over these offsets divided "
                        "by the span, emitted as `pertoken` rows. Default '0' "
                        "reproduces the single-snapshot behavior.")
    p.add_argument("--batch-sizes", type=str,
                   default="1,8,32,64,128,256,512,1024")
    p.add_argument("--warmup", type=int, default=25)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--cuda-graph", dest="cuda_graph", action="store_true",
                   default=True,
                   help="Time via CUDA graph replay (default). Removes host "
                        "launch overhead.")
    p.add_argument("--no-cuda-graph", dest="cuda_graph", action="store_false",
                   help="Force eager-mode timing.")
    p.add_argument("--csv", type=str, default="",
                   help="Optional path to write CSV results (rank 0)")
    p.add_argument("--variants", type=str, default=None,
                   help="Comma list of CPX merge variants to time "
                        "(subset of rccl,step1,step2). Default: all three. "
                        "Ignored in spx mode.")
    p.add_argument("--skip-sanity", action="store_true",
                   help="Skip the pre-sweep sanity check")
    p.add_argument("--sanity-seq-len", type=int, default=8192,
                   help="seq_len for the pre-sweep sanity check")
    p.add_argument("--sanity-batch", type=int, default=1,
                   help="batch size for the pre-sweep sanity check")
    # --- Profiling mode: run ONE shape / ONE variant under an external
    # profiler (rocprofv3). No timing, no CSV, no sweep -- just execute the
    # attention call `--profile-iters` times so the profiler attributes memory
    # counters to a single well-defined workload. Everything else is skipped. ---
    p.add_argument("--profile", action="store_true",
                   help="Profiling mode: execute one shape/variant N times for "
                        "an external profiler (rocprofv3), then exit.")
    p.add_argument("--profile-seq-len", type=int, default=8192)
    p.add_argument("--profile-batch", type=int, default=1)
    p.add_argument("--profile-variant", choices=["spx", "rccl", "step1",
                                                 "step2"], default=None,
                   help="Which path to profile. Defaults to 'spx' in spx mode "
                        "and 'step2' in cpx mode.")
    p.add_argument("--profile-iters", type=int, default=30,
                   help="Number of attention calls to execute under the "
                        "profiler (counters accumulate over these).")
    args = p.parse_args()

    # --- Resolve head shapes: model preset (or llama3-70b fallback) as the
    # base, then any explicitly-passed head arg overrides that dimension. ---
    cfg = _MODEL_CONFIGS[args.model or "llama3-70b"]
    if args.total_q_heads is None:
        args.total_q_heads = cfg["q_heads"]
    if args.total_kv_heads is None:
        args.total_kv_heads = cfg["kv_heads"]
    if args.head_size is None:
        args.head_size = cfg["head_size"]

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
    _log(f"  model preset           : {args.model or '(explicit heads)'}")
    _log(f"  total q-heads / kv-heads: {args.total_q_heads} / {args.total_kv_heads}")
    _log(f"  head_size / block_size : {args.head_size} / {args.block_size}")
    _log(f"  warmup / iters         : {args.warmup} / {args.iters}  (MAX over ranks)")
    _log(f"  timing mode            : {'CUDA graph replay' if args.cuda_graph else 'eager'}")
    _log(f"  token offsets          : {args.token_offsets}")
    _log(f"  reported latency       : per decode step; raw = one output token "
         f"at context base+offset")
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

    # --- Profiling mode: single shape / single variant, no timing. ---
    if args.profile:
        _run_profile_mode(args, grp, cpx, qh_dev, kvh, device)
        destroy_model_parallel()
        dist.destroy_process_group()
        return

    seq_lens = [int(x) for x in args.seq_lens.split(",")]
    token_offsets = sorted(int(x) for x in args.token_offsets.split(","))
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    if args.mode == "cpx":
        variants = ["rccl", "step1", "step2"]
        if args.variants:
            want = [v.strip() for v in args.variants.split(",") if v.strip()]
            bad = [v for v in want if v not in variants]
            assert not bad, f"unknown --variants {bad}; pick from {variants}"
            variants = want
    else:
        variants = ["spx"]

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
                                max(1, args.warmup // 2), max(2, args.iters // 5),
                                args.cuda_graph)
            _log(f"[sanity]   {v:>6}: {us:.2f} us")
        del tn
        torch.cuda.empty_cache()
        _log("[sanity] OK -- harness works; starting full sweep.\n")

    multi_offset = token_offsets != [0]

    # Table header. With multiple offsets we print the actual context (base+off)
    # and the output-token index (offset+1) so each row is self-describing.
    var_cols = "".join(f"{v + '_us':>14}" for v in variants)
    if multi_offset:
        _log(f"{'baseS':>8} {'ctx':>9} {'tok#':>6} {'batch':>7}{var_cols}   scenario")
    else:
        _log(f"{'seq_len':>9} {'batch':>7}{var_cols}   scenario")
    _log("-" * 110)

    # raw rows:      (mode, base_seq_len, offset, context, batch, variant, us)
    # pertoken rows: (mode, base_seq_len, batch, variant, per_token_us)
    raw_rows = []
    # points[(base_S, B, v)] = list of (offset, us) for trapezoidal aggregation
    points: dict = {}

    for base_S in seq_lens:
        scen = _SCENARIO_BY_SEQLEN.get(base_S, "")
        for off in token_offsets:
            S = base_S + off               # context at this output token
            tok_idx = off + 1              # 1-based output-token index
            for B in batch_sizes:
                ok, tn = _try_alloc(args.mode, B, S, qh_dev, kvh,
                                    args.head_size, args.block_size, cpx, device)
                ok_all = _all_reduce_min_int(1 if ok else 0)
                if not ok_all:
                    if ok:  # this rank allocated but a peer OOM'd -- free it
                        del tn
                        torch.cuda.empty_cache()
                    skip_cols = "".join(f"{'SKIP(OOM)':>14}" for _ in variants)
                    if multi_offset:
                        _log(f"{base_S:>8} {S:>9} {tok_idx:>6} {B:>7}"
                             f"{skip_cols}   {scen}")
                    else:
                        _log(f"{S:>9} {B:>7}{skip_cols}   {scen}")
                    for v in variants:
                        raw_rows.append(
                            (args.mode, base_S, off, S, B, v, float("nan")))
                        points.setdefault((base_S, B, v), []).append(
                            (off, float("nan")))
                    continue

                results = {}
                for v in variants:
                    if args.mode == "cpx":
                        _set_variant_env(v)
                    us = _bench_variant(args.mode, tn, grp, cpx, args.head_size,
                                        args.warmup, args.iters, args.cuda_graph)
                    results[v] = us
                    raw_rows.append((args.mode, base_S, off, S, B, v, us))
                    points.setdefault((base_S, B, v), []).append((off, us))

                val_cols = "".join(f"{results[v]:>14.2f}" for v in variants)
                if multi_offset:
                    _log(f"{base_S:>8} {S:>9} {tok_idx:>6} {B:>7}"
                         f"{val_cols}   {scen}")
                else:
                    _log(f"{S:>9} {B:>7}{val_cols}   {scen}")

                del tn
                torch.cuda.empty_cache()

    # --- Per-output-token attention latency (trapezoidal over offsets) ---
    pertoken_rows = []
    if multi_offset:
        _log("-" * 110)
        _log("Per-output-token attention latency (est., trapezoidal over "
             f"offsets {token_offsets}); ATTENTION ONLY, not full TPOT:")
        ptk_cols = "".join(f"{v + '_us':>14}" for v in variants)
        _log(f"{'baseS':>8} {'batch':>7}{ptk_cols}   scenario")
        for base_S in seq_lens:
            scen = _SCENARIO_BY_SEQLEN.get(base_S, "")
            for B in batch_sizes:
                cells = []
                for v in variants:
                    pts = sorted(points.get((base_S, B, v), []))
                    offs = [o for o, _ in pts]
                    lats = [l for _, l in pts]
                    ptk = _trapezoidal_per_token(offs, lats)
                    pertoken_rows.append((args.mode, base_S, B, v, ptk))
                    cells.append(ptk)
                cell_str = "".join(
                    ("SKIP(OOM)".rjust(14) if c != c else f"{c:>14.2f}")
                    for c in cells)
                _log(f"{base_S:>8} {B:>7}{cell_str}   {scen}")

    _log("=" * 110)

    # CSV output (rank 0). Long format with a row_type column so raw curve and
    # derived per-token scalar live in one file, ready for plotting.
    if args.csv and _rank == 0:
        with open(args.csv, "w") as f:
            f.write("mode,row_type,base_seq_len,offset,context,batch,"
                    "variant,latency_us\n")
            for mode, base_S, off, S, B, v, us in raw_rows:
                us_str = "" if us != us else f"{us:.4f}"  # NaN -> empty
                f.write(f"{mode},raw,{base_S},{off},{S},{B},{v},{us_str}\n")
            for mode, base_S, B, v, us in pertoken_rows:
                us_str = "" if us != us else f"{us:.4f}"
                # per-token rows aggregate over offsets -> offset/context blank
                f.write(f"{mode},pertoken,{base_S},,,{B},{v},{us_str}\n")
        _log(f"CSV written to {args.csv}")
    _log("")

    destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
