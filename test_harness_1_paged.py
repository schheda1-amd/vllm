#!/usr/bin/env python3
"""Benchmark vLLM default paged attention (SPX) on MI300X.

Directly benchmarks the paged_attention_rocm kernel with synthetic data,
matching the approach of vLLM's benchmarks/kernels/benchmark_paged_attention.py.
Computes effective HBM bandwidth (GB/s) for each (seq_len, batch_size) configuration.

SPX = Single-Prefix eXecution (standard paged attention decode path).

This is the C/HIP-binding variant: paged_attention_rocm forwards to
torch.ops._rocm_C.paged_attention. For the pure-Triton counterpart see
test_harness_1_wired.py (unified_attention). Run either or both as desired.

Wiring notes (verified against the current vLLM tree):
  - set_random_seed, create_kv_caches_with_random, STR_DTYPE_TO_TORCH_DTYPE
    all live in vllm.utils.torch_utils (NOT vllm.utils).
  - paged_attention_rocm lives in vllm._custom_ops and forwards to
    torch.ops._rocm_C.paged_attention.
  - PARTITION_SIZE_ROCM = 256 is the correct partition size for the *custom*
    ROCm kernel (paged_attention_rocm). The 1024 value in the reference script
    applies only to the non-custom paged_attention_v2 path on non-Navi.
"""

import argparse
import random
import time

import torch

from vllm import _custom_ops as ops
from vllm.utils.torch_utils import (
    STR_DTYPE_TO_TORCH_DTYPE,
    create_kv_caches_with_random,
    set_random_seed,
)

DEFAULT_NUM_BLOCKS = 128 * 1024
PARTITION_SIZE_ROCM = 256

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
    kv_cache_dtype: str | None = None,
    num_iters: int = 100,
    num_blocks: int = DEFAULT_NUM_BLOCKS,
) -> dict:
    set_random_seed(seed)

    scale = float(1.0 / (head_size**0.5))
    query = torch.empty(
        num_seqs, num_query_heads, head_size, dtype=dtype, device=device
    )
    query.uniform_(-scale, scale)

    seq_lens = [seq_len for _ in range(num_seqs)]
    max_seq_len = max(seq_lens)
    seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int, device=device)

    max_num_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
    block_tables_lst: list[list[int]] = []
    for _ in range(num_seqs):
        block_table = [
            random.randint(0, num_blocks - 1) for _ in range(max_num_blocks_per_seq)
        ]
        block_tables_lst.append(block_table)

    block_tables = torch.tensor(block_tables_lst, dtype=torch.int, device=device)

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

    output = torch.empty_like(query)
    num_partitions = (max_seq_len + PARTITION_SIZE_ROCM - 1) // PARTITION_SIZE_ROCM
    tmp_output = torch.empty(
        size=(num_seqs, num_query_heads, num_partitions, head_size),
        dtype=output.dtype,
        device=output.device,
    )
    exp_sums = torch.empty(
        size=(num_seqs, num_query_heads, num_partitions),
        dtype=torch.float32,
        device=output.device,
    )
    max_logits = torch.empty_like(exp_sums)

    k_scale = v_scale = torch.tensor(1.0, dtype=torch.float32, device=device)

    def run_kernel():
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
            max_seq_len,
            None,
            kv_cache_dtype,
            k_scale,
            v_scale,
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

    # Compute effective bandwidth. Byte width keys off the *KV cache* dtype
    # (1 byte under fp8) so the number is not inflated when quantized.
    if kv_cache_dtype is not None and kv_cache_dtype != "auto":
        kv_bytes_per_elem = 1  # fp8 storage
    else:
        kv_bytes_per_elem = torch.finfo(dtype).bits // 8
    kv_bytes = num_seqs * seq_len * num_kv_heads * head_size * 2 * kv_bytes_per_elem
    # Query read and output write use the compute dtype (small relative to KV).
    q_dtype_bytes = torch.finfo(dtype).bits // 8
    qo_bytes = num_seqs * num_query_heads * head_size * q_dtype_bytes * 2  # read Q + write O
    total_bytes = kv_bytes + qo_bytes
    # NOTE: divisor is 1024**3 (GiB/s), reported under a "GB/s" column header.
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
        description="Benchmark vLLM default paged attention (SPX) on MI300X"
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
    parser.add_argument(
        "--kv-cache-dtype", type=str,
        choices=["auto", "fp8", "fp8_e5m2", "fp8_e4m3"],
        default="auto"
    )
    parser.add_argument(
        "--num-blocks", type=int, default=DEFAULT_NUM_BLOCKS,
        help="KV cache pool size in blocks. MUST match the value passed to "
             "test_harness_1_paged_cpx.py, or the two runs use different KV "
             "scatter patterns and the comparison is not apples-to-apples."
    )
    args = parser.parse_args()

    if args.num_query_heads % args.num_kv_heads != 0:
        raise ValueError("num_query_heads must be divisible by num_kv_heads")

    dtype = STR_DTYPE_TO_TORCH_DTYPE[args.dtype]
    kv_cache_dtype = args.kv_cache_dtype

    print(f"vLLM Paged Attention (SPX, rocm_C) Benchmark on MI300X")
    print(f"  num_query_heads={args.num_query_heads}, num_kv_heads={args.num_kv_heads}")
    print(f"  head_size={args.head_size}, block_size={args.block_size}")
    print(f"  dtype={args.dtype}, kv_cache_dtype={args.kv_cache_dtype}")
    print(f"  num_iters={args.num_iters}, num_blocks={args.num_blocks}")
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
                kv_cache_dtype=kv_cache_dtype,
                num_iters=args.num_iters,
                num_blocks=args.num_blocks,
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
