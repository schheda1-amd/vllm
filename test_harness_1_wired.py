#!/usr/bin/env python3
"""Benchmark vLLM Triton unified attention on MI300X.

Directly benchmarks the pure-Triton `unified_attention` decode path with
synthetic data. No C/HIP bindings are used: unified_attention lives in
vllm/v1/attention/ops/triton_unified_attention.py and dispatches to
Triton JIT kernels (kernel_unified_attention_2d / _3d), NOT to
torch.ops._rocm_C / torch.ops._C.

Computes effective HBM bandwidth (GB/s) for each (seq_len, batch_size) config.

Decode scenario: every request contributes a single query token
(query_len == 1) attending over a KV history of `seq_len` tokens
(kv_len == seq_len), with causal masking.

Wiring notes (verified against the current vLLM tree):
  - unified_attention        -> vllm.v1.attention.ops.triton_unified_attention
  - set_random_seed          -> vllm.utils.torch_utils
  - next_power_of_2          -> vllm.utils.math_utils
  - current_platform         -> vllm.platforms
  KV cache layout is flat (num_blocks, block_size, num_kv_heads, head_size);
  there is NO head_size//x split (that split is specific to the C paged
  attention kernel, so create_kv_caches_with_random is intentionally NOT used).
"""

import argparse
import time

import torch

from vllm.platforms import current_platform
from vllm.utils.math_utils import next_power_of_2
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.ops.triton_unified_attention import unified_attention

NUM_BLOCKS = 128 * 1024

# Configurations from the issue table: (seq_len == kv_len, batch_size == num_seqs)
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
    kv_cache_dtype: str = "auto",
    num_iters: int = 100,
    seq_threshold_3d: int | None = None,
    num_par_softmax_segments: int = 16,
) -> dict:
    set_random_seed(seed)

    scale = float(1.0 / (head_size**0.5))

    # Decode: one query token per sequence, attending over `seq_len` KV tokens.
    query_lens = [1 for _ in range(num_seqs)]
    kv_lens_lst = [seq_len for _ in range(num_seqs)]
    max_query_len = max(query_lens)
    max_kv_len = max(kv_lens_lst)

    query = torch.empty(
        sum(query_lens), num_query_heads, head_size, dtype=dtype, device=device
    )
    query.uniform_(-scale, scale)

    # Flat KV cache layout expected by unified_attention.
    key_cache = torch.empty(
        NUM_BLOCKS, block_size, num_kv_heads, head_size, dtype=dtype, device=device
    )
    key_cache.uniform_(-scale, scale)
    value_cache = torch.empty_like(key_cache)
    value_cache.uniform_(-scale, scale)

    cu_query_lens = torch.tensor(
        [0] + query_lens, dtype=torch.int32, device=device
    ).cumsum(dim=0, dtype=torch.int32)
    kv_lens = torch.tensor(kv_lens_lst, dtype=torch.int32, device=device)

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(
        0,
        NUM_BLOCKS,
        (num_seqs, max_num_blocks_per_seq),
        dtype=torch.int32,
        device=device,
    )

    output = torch.empty_like(query)

    # Optional fp8 KV quantization (no C bindings; just a dtype cast + descales).
    q_descale = None
    k_descale = None
    v_descale = None
    kv_bytes_per_elem = torch.finfo(dtype).bits // 8
    if kv_cache_dtype != "auto":
        fp8_dtype = (
            torch.float8_e4m3fnuz
            if current_platform.is_rocm()
            else torch.float8_e4m3fn
        )
        key_cache = key_cache.to(fp8_dtype)
        value_cache = value_cache.to(fp8_dtype)
        kv_bytes_per_elem = 1
        scale_shape = (num_seqs, num_kv_heads)
        k_descale = torch.ones(scale_shape, dtype=torch.float32, device=device)
        v_descale = torch.ones(scale_shape, dtype=torch.float32, device=device)

    # Optional 3D-kernel scratch buffers. When seq_threshold_3d is None the
    # 2D kernel path is used and no scratch is required.
    softmax_segm_output = None
    softmax_segm_max = None
    softmax_segm_expsum = None
    if seq_threshold_3d is not None:
        head_size_padded = next_power_of_2(head_size)
        softmax_segm_output = torch.empty(
            (num_seqs, num_query_heads, num_par_softmax_segments, head_size_padded),
            dtype=torch.float32,
            device=device,
        )
        softmax_segm_max = torch.empty(
            (num_seqs, num_query_heads, num_par_softmax_segments),
            dtype=torch.float32,
            device=device,
        )
        softmax_segm_expsum = torch.empty(
            (num_seqs, num_query_heads, num_par_softmax_segments),
            dtype=torch.float32,
            device=device,
        )

    def run_kernel():
        unified_attention(
            q=query,
            k=key_cache,
            v=value_cache,
            out=output,
            cu_seqlens_q=cu_query_lens,
            max_seqlen_q=max_query_len,
            seqused_k=kv_lens,
            max_seqlen_k=max_kv_len,
            softmax_scale=scale,
            causal=True,
            window_size=(-1, -1),
            block_table=block_tables,
            softcap=0,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            seq_threshold_3D=seq_threshold_3d,
            num_par_softmax_segments=num_par_softmax_segments,
            softmax_segm_output=softmax_segm_output,
            softmax_segm_max=softmax_segm_max,
            softmax_segm_expsum=softmax_segm_expsum,
        )

    # Warmup (also triggers Triton autotune/JIT compilation).
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

    # Effective bandwidth. KV bytes dominate; byte width keys off the *KV cache*
    # dtype (1 byte under fp8) so the number is not inflated when quantized.
    kv_bytes = num_seqs * seq_len * num_kv_heads * head_size * 2 * kv_bytes_per_elem
    q_dtype_bytes = torch.finfo(dtype).bits // 8
    qo_bytes = num_seqs * num_query_heads * head_size * q_dtype_bytes * 2  # Q read + O write
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
        description="Benchmark vLLM Triton unified attention on MI300X"
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
        "--head-size", type=int, choices=[64, 128, 256], default=128
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
        "--seq-threshold-3d", type=int, default=None,
        help="Enable the 3D decode kernel for batches with num_seqs <= this "
        "threshold. Default (None) always uses the 2D kernel."
    )
    args = parser.parse_args()

    if args.num_query_heads % args.num_kv_heads != 0:
        raise ValueError("num_query_heads must be divisible by num_kv_heads")

    dtype = {
        "half": torch.half,
        "bfloat16": torch.bfloat16,
        "float": torch.float,
    }[args.dtype]

    print(f"vLLM Triton Unified Attention Benchmark on MI300X")
    print(f"  num_query_heads={args.num_query_heads}, num_kv_heads={args.num_kv_heads}")
    print(f"  head_size={args.head_size}, block_size={args.block_size}")
    print(f"  dtype={args.dtype}, kv_cache_dtype={args.kv_cache_dtype}")
    print(f"  seq_threshold_3d={args.seq_threshold_3d}")
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
                kv_cache_dtype=args.kv_cache_dtype,
                num_iters=args.num_iters,
                seq_threshold_3d=args.seq_threshold_3d,
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
