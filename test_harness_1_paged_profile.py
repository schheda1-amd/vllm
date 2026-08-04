#!/usr/bin/env python3
"""Single-cell launcher for rocprofv3 bandwidth profiling of the C/HIP paged
attention kernel (SPX decode path).

Companion to test_harness_1_paged.py. That harness sweeps a CONFIGS list and
computes bandwidth from wall-clock time. This one instead runs EXACTLY ONE
(seq_len, batch) cell and just launches the kernel `--profile-iters` times, so
rocprofv3 can attribute HBM FETCH_SIZE / WRITE_SIZE counters and kernel-busy
time to it. No timing/bandwidth math here -- run_paged_bandwidth.sh drives this
under rocprofv3 and parse_rocprof_bandwidth.py computes the bandwidth.

Only the paged kernel dispatches (paged_attention_ll4mi_*) are the metric of
record; the one-time KV-cache fills are excluded downstream by the parser's
--kernel-filter, so they don't need to be avoided here.

Head shapes and dtypes are the SAME as test_harness_1_paged.py (the config that
was benchmarked): full-model 128 q-heads / 8 kv-heads (GQA 16), head_size 128,
block_size 16, bfloat16, kv-cache-dtype auto. This makes the rocprof
bandwidth directly comparable to that harness's wall-clock bandwidth (the
harness gives the locality-ideal lower bound; rocprof measures real HBM
traffic + busy time, expected slightly higher).
"""

import argparse
import random

import torch

from vllm import _custom_ops as ops
from vllm.utils.torch_utils import (
    STR_DTYPE_TO_TORCH_DTYPE,
    create_kv_caches_with_random,
    set_random_seed,
)

NUM_BLOCKS = 128 * 1024
PARTITION_SIZE_ROCM = 256


@torch.inference_mode()
def profile_cell(
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
    profile_iters: int = 30,
    warmup_iters: int = 3,
) -> None:
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
            random.randint(0, NUM_BLOCKS - 1) for _ in range(max_num_blocks_per_seq)
        ]
        block_tables_lst.append(block_table)
    block_tables = torch.tensor(block_tables_lst, dtype=torch.int, device=device)

    key_caches, value_caches = create_kv_caches_with_random(
        NUM_BLOCKS,
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

    for _ in range(warmup_iters):
        run_kernel()
    torch.cuda.synchronize()

    # Profiled region: only paged_attention_ll4mi_* dispatches matter; the
    # parser scopes bytes AND time to them via --kernel-filter.
    for _ in range(profile_iters):
        run_kernel()
    torch.cuda.synchronize()


def main():
    p = argparse.ArgumentParser(
        description="Single-cell rocprof launcher for paged attention (SPX)"
    )
    # Defaults MATCH test_harness_1_paged.py (the benchmarked config):
    # 128 q / 8 kv (GQA 16), head_size 128, block_size 16, bfloat16, kv auto.
    p.add_argument("--model", default=None,
                   help="Optional label only (does not change shapes).")
    p.add_argument("--seq-len", type=int, required=True)
    p.add_argument("--batch", type=int, required=True)
    p.add_argument("--num-query-heads", type=int, default=128,
                   help="Number of query heads (default 128)")
    p.add_argument("--num-kv-heads", type=int, default=8,
                   help="Number of KV heads / GQA (default 8)")
    p.add_argument("--head-size", type=int, choices=[64, 128], default=128)
    p.add_argument("--block-size", type=int, choices=[16, 32], default=16)
    p.add_argument("--dtype", type=str, choices=["half", "bfloat16", "float"],
                   default="bfloat16")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--profile-iters", type=int, default=30)
    p.add_argument("--kv-cache-dtype", type=str,
                   choices=["auto", "fp8", "fp8_e5m2", "fp8_e4m3"], default="auto")
    args = p.parse_args()

    if args.num_query_heads % args.num_kv_heads != 0:
        raise ValueError("num_query_heads must be divisible by num_kv_heads")

    dtype = STR_DTYPE_TO_TORCH_DTYPE[args.dtype]

    print(f"[paged-profile] model={args.model or '(default)'} "
          f"S={args.seq_len} B={args.batch} "
          f"q={args.num_query_heads} kv={args.num_kv_heads} "
          f"head_size={args.head_size} "
          f"block_size={args.block_size} dtype={args.dtype} "
          f"kv_cache_dtype={args.kv_cache_dtype} iters={args.profile_iters}")

    profile_cell(
        num_seqs=args.batch,
        seq_len=args.seq_len,
        num_query_heads=args.num_query_heads,
        num_kv_heads=args.num_kv_heads,
        head_size=args.head_size,
        block_size=args.block_size,
        dtype=dtype,
        seed=args.seed,
        kv_cache_dtype=args.kv_cache_dtype,
        profile_iters=args.profile_iters,
    )
    print("[paged-profile] done")


if __name__ == "__main__":
    main()
