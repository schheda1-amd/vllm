#!/usr/bin/env python3
"""Benchmark this repo's unified_attention SPX path on MI300X.

Same measurement METHOD as test_harness_1.py (analytical KV+Q+O bytes divided by
host wall-clock latency, reported as GiB/s), but drives this repository's
unified_attention kernel (enable_starscream=False, cpx_size=1) instead of the
stock paged_attention_rocm. Holding the method + shapes fixed lets the resulting
numbers sit on the same ruler as the reference paged-attention numbers, so the
delta is a clean "unified_attention SPX vs paged_attention_rocm" comparison.

SPX = the standard single-partition decode path (no cross-XCD merge).
"""

import argparse
import math
import time

import torch

from vllm.attention.ops.triton_unified_attention import unified_attention
from vllm.utils.torch_utils import (
    STR_DTYPE_TO_TORCH_DTYPE,
    set_random_seed,
)

# Configurations from the issue table
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
    device: str = "cuda",
    num_iters: int = 100,
) -> dict:
    set_random_seed(seed)

    softmax_scale = float(head_size ** -0.5)

    # unified_attention layout (matches bench_starscream_attention.py _try_alloc
    # SPX branch): paged KV cache as (total_blocks, block_size, kv_heads,
    # head_size), one query token per sequence (decode), contiguous block table.
    npb = math.ceil(seq_len / block_size)          # blocks per sequence
    total_blocks = num_seqs * npb

    key_cache = torch.randn(
        total_blocks, block_size, num_kv_heads, head_size, dtype=dtype, device=device
    )
    value_cache = torch.randn(
        total_blocks, block_size, num_kv_heads, head_size, dtype=dtype, device=device
    )
    block_tables = torch.arange(
        total_blocks, dtype=torch.int32, device=device
    ).reshape(num_seqs, npb)

    # Decode: 1 query token per sequence.
    cu_seqlens_q = torch.arange(num_seqs + 1, dtype=torch.int32, device=device)
    seqused_k = torch.full((num_seqs,), seq_len, dtype=torch.int32, device=device)

    query = torch.randn(num_seqs, num_query_heads, head_size, dtype=dtype, device=device)
    output = torch.empty_like(query)

    k_descale = torch.ones(num_seqs, num_kv_heads, dtype=torch.float32, device=device)
    v_descale = torch.ones(num_seqs, num_kv_heads, dtype=torch.float32, device=device)

    def run_kernel():
        unified_attention(
            q=query,
            k=key_cache,
            v=value_cache,
            out=output,
            slice_idx=0,
            starscream_rank=0,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=1,
            seqused_k=seqused_k,
            max_seqlen_k=seq_len,
            softmax_scale=softmax_scale,
            causal=True,
            window_size=(-1, -1),
            block_table=block_tables,
            softcap=0,
            q_descale=None,
            k_descale=k_descale,
            v_descale=v_descale,
            cpx_size=1,
            enable_starscream=False,
        )

    # Warmup
    for _ in range(3):
        run_kernel()
    torch.cuda.synchronize()

    # Benchmark
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    for _ in range(num_iters):
        run_kernel()
    torch.cuda.synchronize()
    end_time = time.perf_counter()

    latency = (end_time - start_time) / num_iters

    # Compute effective bandwidth
    # KV cache bytes read = num_seqs * seq_len * num_kv_heads * head_size * 2 (K+V) * dtype_bytes
    dtype_bytes = torch.finfo(dtype).bits // 8
    kv_bytes = (
        num_seqs * seq_len * num_kv_heads * head_size * 2 * dtype_bytes
    )
    # Also account for query read and output write (small relative to KV)
    qo_bytes = num_seqs * num_query_heads * head_size * dtype_bytes * 2  # read Q + write O
    total_bytes = kv_bytes + qo_bytes
    bandwidth_gbs = (total_bytes / latency) / (1024**3)

    return {
        "seq_len": seq_len,
        "batch_size": num_seqs,
        "latency_us": latency * 1e6,
        "bandwidth_gbs": bandwidth_gbs,
        "kv_bytes": kv_bytes,
        "total_bytes": total_bytes,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark this repo's unified_attention SPX path on MI300X"
    )
    parser.add_argument(
        "--num-query-heads", type=int, default=128,
        help="Number of query heads (GPT-120B-like)"
    )
    parser.add_argument(
        "--num-kv-heads", type=int, default=8,
        help="Number of KV heads (GQA)"
    )
    parser.add_argument(
        "--head-size", type=int, choices=[64, 128], default=128
    )
    parser.add_argument(
        "--block-size", type=int, choices=[16, 32], default=16
    )
    parser.add_argument(
        "--dtype", type=str, choices=["half", "bfloat16", "float"],
        default="bfloat16"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--num-iters", type=int, default=100,
        help="Number of benchmark iterations per config"
    )
    args = parser.parse_args()

    if args.num_query_heads % args.num_kv_heads != 0:
        raise ValueError("num_query_heads must be divisible by num_kv_heads")

    dtype = STR_DTYPE_TO_TORCH_DTYPE[args.dtype]

    print(f"unified_attention (SPX) Benchmark on MI300X")
    print(f"  num_query_heads={args.num_query_heads}, num_kv_heads={args.num_kv_heads}")
    print(f"  head_size={args.head_size}, block_size={args.block_size}")
    print(f"  dtype={args.dtype}")
    print(f"  num_iters={args.num_iters}")
    print()
    print(f"{'SeqLen':>10} {'BatchSize':>10} {'Latency(us)':>12} {'BW(GB/s)':>10}")
    print("-" * 50)

    results = []
    for seq_len, batch_size in CONFIGS:
        try:
            r = benchmark_config(
                num_seqs=batch_size,
                seq_len=seq_len,
                num_query_heads=args.num_query_heads,
                num_kv_heads=args.num_kv_heads,
                head_size=args.head_size,
                block_size=args.block_size,
                dtype=dtype,
                seed=args.seed,
                num_iters=args.num_iters,
            )
            print(
                f"{r['seq_len']:>10} {r['batch_size']:>10} "
                f"{r['latency_us']:>12.3f} {r['bandwidth_gbs']:>10.1f}"
            )
            results.append(r)
        except Exception as e:
            print(f"  ERROR for seq_len={seq_len}, batch_size={batch_size}: {e}")
            results.append({
                "seq_len": seq_len,
                "batch_size": batch_size,
                "latency_us": 0,
                "bandwidth_gbs": 0,
                "error": str(e),
            })

    # Compute primary metric: average bandwidth across all configs
    valid_results = [r for r in results if r.get("bandwidth_gbs", 0) > 0]
    if valid_results:
        primary_metric = sum(r["bandwidth_gbs"] for r in valid_results) / len(valid_results)
    else:
        primary_metric = 0.0

    print()
    print(f"Average bandwidth across {len(valid_results)} configs: {primary_metric:.1f} GB/s")

    # Canonical metric envelope
    print("===== AMDPILOT_METRIC v1 =====")
    print(f"metric_value: {primary_metric}")
    print("===== END AMDPILOT_METRIC =====")


if __name__ == "__main__":
    main()