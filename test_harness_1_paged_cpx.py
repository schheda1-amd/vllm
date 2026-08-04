#!/usr/bin/env python3
"""Benchmark the HIP paged-attention decode kernel under CPX + Starscream.

Companion to test_harness_1_paged.py (the SPX baseline). Here the physical
MI300X is in CPX mode, so one rank runs per XCD. Each rank runs the ordinary
`paged_attention_rocm` kernel over a contiguous, block-aligned slice of the KV
context, emits a partial-softmax record instead of a final output, and the
ranks merge those records with a single small Triton kernel.

Run under torchrun, one process per XCD:

    rocm-smi --setcomputepartition CPX
    torchrun --nnodes=1 --nproc-per-node=8 test_harness_1_paged_cpx.py

Add --check to validate the merged result against a single-rank full-context
run of the same problem before benchmarking.

Numbers are reported by rank 0. Latency is the MAX across ranks (the merge is
a barrier, so the slowest XCD sets the step time); bandwidth is computed
against the FULL context, i.e. the bytes one physical GPU moved.
"""

import argparse
import os
import random
import time
from datetime import timedelta

import torch
import torch.distributed as dist

from vllm import _custom_ops as ops
from vllm.attention.ops.starscream_paged_merge import (
    make_meta_buffer,
    starscream_merge,
)
from vllm.attention.ops.starscream_paged_symm_merge import (
    fused_paged_merge,
    get_symm_meta,
)
from vllm.utils.torch_utils import (
    STR_DTYPE_TO_TORCH_DTYPE,
    create_kv_caches_with_random,
    set_random_seed,
)

# Smaller than the SPX harness's 128*1024. Every rank allocates its own full
# copy of the pool, so the total is world_size x this: at 128*1024 blocks
# (8.4 GiB each) 8 ranks would want 67 GiB, and under a memory partition that
# splits HBM per XCD it will not fit at all.
#
# NOTE for SPX-vs-CPX comparisons: the block IDs are drawn uniformly from
# [0, NUM_BLOCKS), so a different pool size means a different KV scatter
# pattern and therefore different TLB/cache behaviour. If the machine has the
# headroom (NPS1, where all XCDs see the full 192 GiB), pass
# --num-blocks 131072 to match test_harness_1_paged.py exactly.
DEFAULT_NUM_BLOCKS = 32 * 1024
PARTITION_SIZE_ROCM = 256

CONFIGS = [
    (8192, 1),
    (8192, 32),
    (8192, 64),
    (8192, 128),
    (8192, 1024),
    (131072, 1),
    (131072, 128),
    (131072, 1024),
]

# --check uses its own small shapes. The reference is a FULL-context run, whose
# tmp_out is world_size x bigger than the local one: at (131072, 1024) that is
# a 16 GiB allocation on an XCD that only owns ~24 GiB, and an OOM on one rank
# hangs every other rank in the next collective. These cover the branches that
# actually differ without going near that:
#   (4096, 2)     local 512 tok  -> num_partitions 2, below JCHUNK
#   (16384, 2)    local 2048 tok -> num_partitions 8
#   (131072, 1)   local 16k tok  -> num_partitions 64 == MAX_NPAR boundary;
#                                   reference npar 512 -> npar_loops 8 (the cap)
#   (8200, 1)     not a multiple of block_size*world -> ragged tail slice
#   (96, 1)       6 blocks over 8 ranks -> ranks 6,7 own zero tokens, which is
#                                   the num_partitions == 0 guard
CHECK_CONFIGS = [
    (4096, 2),
    (16384, 2),
    (131072, 1),
    (8200, 1),
    (96, 1),
]


def _select_configs(only: list[str] | None):
    """Resolve --only into a config list, or fall back to the full sweep.

    Every rank parses the same argv, so the resulting list is identical
    everywhere -- which it must be, since the ranks walk configs in lockstep
    and the symm-mem buffer cache is keyed on the per-config numel.
    """
    if not only:
        return CONFIGS
    picked = []
    for spec in only:
        try:
            s, b = (int(x) for x in spec.split(","))
        except ValueError:
            raise SystemExit(f"--only expects SEQLEN,BATCH (got {spec!r})")
        picked.append((s, b))
    return picked


class ShimGroup:
    """Minimal stand-in for vLLM's DCP ``GroupCoordinator``.

    The symm-mem manager only reads these five attributes. The harness runs
    under bare torchrun without vllm.distributed.parallel_state initialised,
    so rather than stand up a whole vLLM parallel state we hand it the default
    process group wearing the right shape.
    """

    def __init__(self, device: torch.device):
        self.device_group = dist.group.WORLD
        self.world_size = dist.get_world_size()
        self.rank_in_group = dist.get_rank()
        self.device = device
        self.unique_name = getattr(
            dist.group.WORLD, "group_name", "starscream_paged"
        )


def block_slice(total_blocks: int, seq_len: int, block_size: int, rank: int, world: int):
    """Block-aligned contiguous split of the context across ranks.

    Returns (block_start, num_blocks, num_tokens) for `rank`. Block alignment
    means a rank's slice starts at local token 0 of its own block-table row,
    which is exactly what the kernel assumes -- no kernel change needed.
    """
    base = total_blocks // world
    extra = total_blocks % world
    if rank < extra:
        start = rank * (base + 1)
        nblk = base + 1
    else:
        start = extra * (base + 1) + (rank - extra) * base
        nblk = base
    tok_start = start * block_size
    tok_end = min(seq_len, (start + nblk) * block_size)
    return start, nblk, max(0, tok_end - tok_start)


@torch.inference_mode()
def benchmark_config(
    num_seqs: int,
    seq_len: int,
    num_query_heads: int,
    num_kv_heads: int,
    head_size: int,
    block_size: int,
    dtype: torch.dtype,
    seed: int,
    rank: int,
    world: int,
    device: str = "cuda",
    kv_cache_dtype: str | None = None,
    num_iters: int = 100,
    check: bool = False,
    num_blocks: int = DEFAULT_NUM_BLOCKS,
    merge: str = "fused",
    scatter_heads: bool = False,
    group=None,
) -> dict:
    # Same seed on every rank: the query and the global block table must match,
    # since after the query all-gather every rank holds the full query.
    set_random_seed(seed)

    scale = float(1.0 / (head_size**0.5))
    query = torch.empty(
        num_seqs, num_query_heads, head_size, dtype=dtype, device=device
    )
    query.uniform_(-scale, scale)

    total_blocks = (seq_len + block_size - 1) // block_size
    global_block_tables = torch.tensor(
        [
            [random.randint(0, num_blocks - 1) for _ in range(total_blocks)]
            for _ in range(num_seqs)
        ],
        dtype=torch.int,
        device=device,
    )

    blk_start, nblk, local_len = block_slice(
        total_blocks, seq_len, block_size, rank, world
    )
    # A rank with an empty slice still has to take part in the all-gather, so
    # give it a one-block table and a zero seq_len; the kernel's num_partitions
    # == 0 early-out emits an identity record. blk_start is clamped because an
    # empty rank's start can sit past the end of the table, and Python slicing
    # would silently hand back a 0-wide tensor (block_tables.size(1) == 0).
    tbl_start = min(blk_start, total_blocks - 1)
    block_tables = global_block_tables[
        :, tbl_start : tbl_start + max(nblk, 1)
    ].contiguous()
    seq_lens_tensor = torch.full(
        (num_seqs,), local_len, dtype=torch.int, device=device
    )
    max_local_len = max(local_len, 1)

    key_caches, value_caches = create_kv_caches_with_random(
        num_blocks,
        block_size,
        1,
        num_kv_heads,
        head_size,
        kv_cache_dtype,
        dtype,
        device=device,
    )
    key_cache, value_cache = key_caches[0], value_caches[0]

    # `out` argument of paged_attention_rocm. Full width on purpose: the
    # binding derives num_heads from its shape. On the Starscream path the HIP
    # reduce kernel early-returns into starscream_meta_out and never writes it,
    # so it is pure scratch -- the kind of temporary redundancy that costs
    # nothing because no traffic touches it.
    output = torch.empty_like(query)

    # The tensor this rank actually OWNS. Under head-sharded aggregation that
    # is num_query_heads // world heads at local indices [0, chunk); across the
    # 8 XCDs the physical device holds all num_query_heads exactly once, which
    # is what SPX produces.
    # When not sharding it aliases `output` -- same shape, same dtype, and the
    # HIP kernel never writes it -- so the replicated path allocates exactly
    # what it used to (32 MiB matters on an XCD that owns ~24 GiB).
    heads_out = num_query_heads // world if scatter_heads else num_query_heads
    head_offset = rank * heads_out if scatter_heads else 0
    merged = (
        torch.empty((num_seqs, heads_out, head_size), dtype=dtype, device=device)
        if scatter_heads
        else output
    )

    num_partitions = (max_local_len + PARTITION_SIZE_ROCM - 1) // PARTITION_SIZE_ROCM
    tmp_output = torch.empty(
        (num_seqs, num_query_heads, num_partitions, head_size),
        dtype=output.dtype,
        device=device,
    )
    exp_sums = torch.empty(
        (num_seqs, num_query_heads, num_partitions),
        dtype=torch.float32,
        device=device,
    )
    max_logits = torch.empty_like(exp_sums)

    # Allocation checkpoint. Everything that can OOM has now been allocated,
    # and nothing past this point is a new large allocation. It must sit HERE,
    # ahead of the symm-mem rendezvous below, because both of the collectives
    # that follow can hang unboundedly on an asymmetric failure: rendezvous has
    # no timeout of its own, and the signal-pad barrier inside the fused merge
    # spins on a device flag forever. This RCCL barrier carries the process
    # group's 10-minute timeout, so a rank OOMing above turns into a bounded
    # failure instead of 8 wedged XCDs. Outside the timed region, once per
    # config, and it also covers the --check path (which returns before warmup).
    dist.barrier()

    # Fused path: get the symm-mem buffer FIRST and let the HIP reduce kernel
    # write its record straight into it -- no staging copy, no RCCL, no
    # gathered scratch tensor.
    meta = symm_buf = gathered = None
    if merge == "fused" and world > 1:
        meta, symm_buf = get_symm_meta(
            num_seqs, num_query_heads, head_size, group
        )
    if meta is None:
        merge = "rccl"
        meta = make_meta_buffer(num_seqs, num_query_heads, head_size, device)
        gathered = torch.empty(
            (world * num_seqs, num_query_heads, head_size + 2),
            dtype=torch.float32,
            device=device,
        )

    k_scale = v_scale = torch.tensor(1.0, dtype=torch.float32, device=device)

    def run_local(meta_out):
        ops.paged_attention_rocm(
            output,
            exp_sums,
            max_logits,
            tmp_output,
            query,
            key_cache,
            value_cache,
            num_kv_heads,
            scale,
            block_tables,
            seq_lens_tensor,
            None,
            block_size,
            max_local_len,
            None,
            kv_cache_dtype,
            k_scale,
            v_scale,
            starscream_meta_out=meta_out,
        )

    if merge == "fused":

        def run_step():
            run_local(meta)
            fused_paged_merge(
                merged,
                symm_buf,
                group,
                num_seqs,
                num_query_heads,  # global: addresses the peer records
                head_size,
                scatter_heads=scatter_heads,
            )

    else:

        def run_step():
            run_local(meta)
            starscream_merge(meta, merged, world, gathered=gathered)

    if check:
        return _correctness_check(
            run_step,
            merged,
            head_offset,
            query,
            key_cache,
            value_cache,
            global_block_tables,
            seq_len,
            num_seqs,
            num_query_heads,
            num_kv_heads,
            head_size,
            block_size,
            scale,
            kv_cache_dtype,
            k_scale,
            v_scale,
            local_len,
        )

    for _ in range(3):
        run_step()
    torch.cuda.synchronize()
    dist.barrier()

    start_time = time.perf_counter()
    for _ in range(num_iters):
        run_step()
    torch.cuda.synchronize()
    latency = (time.perf_counter() - start_time) / num_iters

    # NOTE: no collective here. The max-across-ranks reduction happens in
    # main(), after every rank has agreed that this config succeeded --
    # otherwise one rank raising (OOM, say) leaves the others hanging here.

    if kv_cache_dtype is not None and kv_cache_dtype != "auto":
        kv_bytes_per_elem = 1
    else:
        kv_bytes_per_elem = torch.finfo(dtype).bits // 8
    kv_bytes = num_seqs * seq_len * num_kv_heads * head_size * 2 * kv_bytes_per_elem
    q_dtype_bytes = torch.finfo(dtype).bits // 8
    qo_bytes = num_seqs * num_query_heads * head_size * q_dtype_bytes * 2
    total_bytes = kv_bytes + qo_bytes
    bandwidth_gbs = (total_bytes / latency) / (1024**3)

    return {
        "seq_len": seq_len,
        "batch_size": num_seqs,
        "latency_us": latency * 1e6,
        "bandwidth_gbs": bandwidth_gbs,
        "kv_bytes": kv_bytes,
        "total_bytes": total_bytes,
        "merge": merge,
    }


def _correctness_check(
    run_step,
    merged,
    head_offset,
    query,
    key_cache,
    value_cache,
    global_block_tables,
    seq_len,
    num_seqs,
    num_query_heads,
    num_kv_heads,
    head_size,
    block_size,
    scale,
    kv_cache_dtype,
    k_scale,
    v_scale,
    local_len,
):
    """Compare the merged CPX result against a full-context single-rank run.

    Every rank computes the same full-context reference (all ranks hold
    identical KV caches, since they share a seed). Under head-sharded
    aggregation each rank then checks a DIFFERENT slice of it, so the verdicts
    are no longer identical by construction and main() must all-reduce them
    before anyone is allowed to exit.
    """
    run_step()
    got = merged.clone()

    ref = torch.empty_like(query)
    npar = (seq_len + PARTITION_SIZE_ROCM - 1) // PARTITION_SIZE_ROCM
    tmp = torch.empty(
        (num_seqs, num_query_heads, npar, head_size),
        dtype=ref.dtype,
        device=ref.device,
    )
    es = torch.empty(
        (num_seqs, num_query_heads, npar), dtype=torch.float32, device=ref.device
    )
    ml = torch.empty_like(es)
    ops.paged_attention_rocm(
        ref,
        es,
        ml,
        tmp,
        query,
        key_cache,
        value_cache,
        num_kv_heads,
        scale,
        global_block_tables,
        torch.full((num_seqs,), seq_len, dtype=torch.int, device=ref.device),
        None,
        block_size,
        seq_len,
        None,
        kv_cache_dtype,
        k_scale,
        v_scale,
    )
    torch.cuda.synchronize()

    # Line up the reference with what this rank owns. Under head-sharded
    # aggregation `got` is [B, num_heads // world, D] holding global heads
    # [head_offset, head_offset + chunk) at local indices; otherwise it is the
    # full tensor and this slice is a no-op.
    ref = ref[:, head_offset : head_offset + got.shape[1]]

    # Tolerance-based: the producer and the reducer each carry their own 1e-6
    # epsilon and the reduction order differs, so bit-exactness is impossible.
    # Scale the error by the magnitude of the whole tensor rather than
    # per-element: attention outputs over thousands of zero-mean values average
    # towards 0, and a per-element denominator turns that into false failures.
    diff = (got.float() - ref.float()).abs()
    max_abs = diff.max().item()
    ref_scale = max(ref.float().abs().max().item(), 1e-6)
    max_rel = max_abs / ref_scale
    return {
        "seq_len": seq_len,
        "batch_size": num_seqs,
        "local_len": local_len,
        "max_abs": max_abs,
        "max_rel": max_rel,
        "ok": bool(max_rel < 2e-2) and not torch.isnan(got).any().item(),
    }


def main():
    parser = argparse.ArgumentParser(
        description="HIP paged attention under CPX + Starscream on MI300X"
    )
    parser.add_argument("--num-query-heads", type=int, default=128)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-size", type=int, choices=[64, 128], default=128)
    parser.add_argument("--block-size", type=int, choices=[16, 32], default=16)
    parser.add_argument(
        "--dtype", type=str, choices=["half", "bfloat16", "float"], default="bfloat16"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-iters", type=int, default=100)
    parser.add_argument(
        "--kv-cache-dtype",
        type=str,
        choices=["auto", "fp8", "fp8_e5m2", "fp8_e4m3"],
        default="auto",
    )
    parser.add_argument(
        "--num-blocks",
        type=int,
        default=DEFAULT_NUM_BLOCKS,
        help="KV cache pool size in blocks, PER RANK. Use 131072 to match "
             "test_harness_1_paged.py's scatter pattern if HBM allows.",
    )
    parser.add_argument(
        "--merge",
        choices=["fused", "rccl"],
        default="fused",
        help="fused: symmetric-memory peer-pointer merge, no RCCL on the "
             "critical path (the HIP kernel writes its record straight into "
             "symm-mem). rccl: all_gather_into_tensor + separate merge kernel.",
    )
    parser.add_argument(
        "--scatter-heads",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="head-sharded aggregation: rank r merges only heads "
             "[r*chunk, (r+1)*chunk), so each head is combined ONCE across the "
             "device -- the same output ownership SPX has, and what a "
             "TP-sharded o_proj consumes. Cuts cross-XCD merge traffic by the "
             "world size. --no-scatter-heads reverts to every rank merging "
             "every head (world x redundant); fused merge only.",
    )
    parser.add_argument(
        "--only",
        action="append",
        metavar="SEQLEN,BATCH",
        help="Run only this (seq_len, batch) cell instead of the full sweep. "
             "Repeatable. Used by the rocprof drivers so the profiled path is "
             "the benchmarked path, not a separate launcher that can drift.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate the merged output against a full-context single-rank run",
    )
    args = parser.parse_args()
    configs = _select_configs(args.only)

    if args.num_query_heads % args.num_kv_heads != 0:
        raise ValueError("num_query_heads must be divisible by num_kv_heads")

    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    # Bounded timeout: if the ranks ever desynchronise, die in 10 minutes
    # instead of pinning 8 XCDs indefinitely.
    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=10))

    dtype = STR_DTYPE_TO_TORCH_DTYPE[args.dtype]

    # Resolve the merge path once, loudly. Silently degrading to RCCL would
    # make the headline number mean something different than the flag says.
    group = ShimGroup(torch.device(f"cuda:{local_rank}"))
    merge_mode = args.merge
    if merge_mode == "fused" and world > 1:
        from vllm.distributed.device_communicators.symm_mem_allgather import (
            get_symm_mem_allgather_manager,
        )

        # The decision must be UNANIMOUS. It is taken from local state
        # (symm_mem_available + world size), so a split verdict is unlikely --
        # but if it ever happened the ranks that chose "fused" would block in
        # torch_symm_mem.rendezvous (a collective) while the ranks that chose
        # "rccl" went on to all_gather_into_tensor, and the job would deadlock
        # with no timeout. A MIN all-reduce makes any single "no" carry.
        ok = torch.tensor(
            [1 if get_symm_mem_allgather_manager(group) is not None else 0],
            dtype=torch.int32,
            device=f"cuda:{local_rank}",
        )
        dist.all_reduce(ok, op=dist.ReduceOp.MIN)
        if ok.item() == 0:
            merge_mode = "rccl"
            if rank == 0:
                print(
                    "  WARNING: symmetric memory unavailable on at least one "
                    "rank -- falling back to the RCCL merge.",
                    flush=True,
                )

    # Head sharding lives in the fused merge kernel's grid. The RCCL path does
    # not get it: its redundancy is in all_gather_into_tensor, which hands every
    # rank all world records regardless of which heads it needs, so narrowing
    # the kernel there would save arithmetic but not one byte of link traffic.
    # Reshaping that collective is a separate change; don't imply it happened.
    scatter_heads = args.scatter_heads and merge_mode == "fused" and world > 1
    if scatter_heads and args.num_query_heads % world != 0:
        scatter_heads = False
        if rank == 0:
            print(
                f"  WARNING: num_query_heads={args.num_query_heads} is not "
                f"divisible by world={world} -- head sharding disabled.",
                flush=True,
            )
    elif args.scatter_heads and not scatter_heads and world > 1:
        if rank == 0:
            print(
                "  WARNING: --scatter-heads applies to the fused merge only "
                "-- ignored on the RCCL path.",
                flush=True,
            )

    common = dict(
        num_query_heads=args.num_query_heads,
        num_kv_heads=args.num_kv_heads,
        head_size=args.head_size,
        block_size=args.block_size,
        dtype=dtype,
        seed=args.seed,
        rank=rank,
        world=world,
        kv_cache_dtype=args.kv_cache_dtype,
        num_blocks=args.num_blocks,
        merge=merge_mode,
        scatter_heads=scatter_heads,
        group=group,
    )

    if args.check:
        if rank == 0:
            print("Correctness: merged CPX output vs full-context single-rank run")
            print(
                f"{'SeqLen':>8} {'Batch':>6} {'LocalTok':>9} "
                f"{'max_abs':>11} {'max_rel':>11}  verdict"
            )
            print("-" * 62)
        all_ok = True
        for seq_len, batch_size in CHECK_CONFIGS:
            c = benchmark_config(
                num_seqs=batch_size,
                seq_len=seq_len,
                num_iters=0,
                check=True,
                **common,
            )
            # Under head sharding each rank checked a different head slice, so
            # rank 0's verdict is no longer the whole story -- and a rank
            # exiting alone on its own FAIL would strand the rest at the
            # barrier below. Reduce to the worst error seen anywhere.
            worst = torch.tensor(
                [c["max_abs"], c["max_rel"], 0.0 if c["ok"] else 1.0],
                dtype=torch.float64,
                device=f"cuda:{local_rank}",
            )
            dist.all_reduce(worst, op=dist.ReduceOp.MAX)
            c["max_abs"], c["max_rel"] = worst[0].item(), worst[1].item()
            c["ok"] = worst[2].item() == 0.0
            all_ok = all_ok and c["ok"]
            if rank == 0:
                print(
                    f"{c['seq_len']:>8} {c['batch_size']:>6} {c['local_len']:>9} "
                    f"{c['max_abs']:>11.3e} {c['max_rel']:>11.3e}  "
                    f"{'PASS' if c['ok'] else 'FAIL'}",
                    flush=True,
                )
            dist.barrier()
        if not all_ok:
            if rank == 0:
                print("\nCorrectness FAILED -- not benchmarking.", flush=True)
            dist.destroy_process_group()
            raise SystemExit(1)
        if rank == 0:
            print()

    if rank == 0:
        print("HIP Paged Attention (CPX + Starscream) Benchmark on MI300X")
        print(f"  cpx_size={world}")
        print(
            f"  num_query_heads={args.num_query_heads}, "
            f"num_kv_heads={args.num_kv_heads}"
        )
        print(f"  head_size={args.head_size}, block_size={args.block_size}")
        print(f"  dtype={args.dtype}, kv_cache_dtype={args.kv_cache_dtype}")
        print(f"  num_iters={args.num_iters}, num_blocks={args.num_blocks}/rank")
        print(f"  merge={merge_mode}, scatter_heads={scatter_heads}")
        print()
        print(f"{'SeqLen':>10} {'BatchSize':>10} {'Latency(us)':>12} {'BW(GB/s)':>10}")
        print("-" * 50)

    results = []
    for seq_len, batch_size in configs:
        err = None
        r = None
        try:
            r = benchmark_config(
                num_seqs=batch_size,
                seq_len=seq_len,
                num_iters=args.num_iters,
                check=False,
                **common,
            )
        except Exception as e:  # noqa: BLE001 - a bench should survive one cell
            err = str(e)
            torch.cuda.empty_cache()

        # Agree on success BEFORE any further collective: an asymmetric failure
        # (an OOM on one XCD) would otherwise strand the other ranks.
        st = torch.tensor(
            [0 if err else 1], dtype=torch.int32, device=f"cuda:{local_rank}"
        )
        dist.all_reduce(st, op=dist.ReduceOp.MIN)

        if st.item() == 0:
            if rank == 0:
                print(
                    f"  ERROR for seq_len={seq_len}, batch_size={batch_size}: "
                    f"{err or 'failed on another rank'}",
                    flush=True,
                )
            results.append(
                {
                    "seq_len": seq_len,
                    "batch_size": batch_size,
                    "latency_us": 0,
                    "bandwidth_gbs": 0,
                    "error": err or "failed on another rank",
                }
            )
            continue

        # The merge is a barrier, so the slowest XCD sets the step time.
        lat = torch.tensor(
            [r["latency_us"]], dtype=torch.float64, device=f"cuda:{local_rank}"
        )
        dist.all_reduce(lat, op=dist.ReduceOp.MAX)
        scale_up = lat.item() / r["latency_us"]
        r["latency_us"] = lat.item()
        r["bandwidth_gbs"] /= scale_up

        if rank == 0:
            print(
                f"{r['seq_len']:>10} {r['batch_size']:>10} "
                f"{r['latency_us']:>12.3f} {r['bandwidth_gbs']:>10.1f}",
                flush=True,
            )
        results.append(r)

    valid = [r for r in results if r.get("bandwidth_gbs", 0) > 0]
    primary_metric = (
        sum(r["bandwidth_gbs"] for r in valid) / len(valid) if valid else 0.0
    )

    if rank == 0:
        print()
        print(
            f"Average bandwidth across {len(valid)} configs: {primary_metric:.1f} GB/s"
        )
        print("===== AMDPILOT_METRIC v1 =====")
        print(f"metric_value: {primary_metric}")
        print("===== END AMDPILOT_METRIC =====")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
