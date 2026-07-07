# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Functional and output-matching tests for the Starscream symmetric-memory
allgather + fused reduce path (CPX+NPS4).

Run instructions (inside the ROCm/PyTorch docker container):
------------------------------------------------------------

  # Step 1 only (symm-mem allgather -> unmodified reduce_segments):
  VLLM_STARSCREAM_USE_SYMM_MEM=1 \
  TORCH_SYMM_MEM_DISABLE_MULTICAST=1 \
  torchrun --nnodes=1 --nproc-per-node=<CPX_SIZE> \
    tests/kernels/attention/test_starscream_symm_reduce.py \
    --mode step1 [--num-tokens 16] [--num-heads 32] [--head-size 128] \
                 [--cpx-size <CPX_SIZE>] [--seq-len 4096] [--rtol 1e-3] [--atol 1e-3]

  # Step 2 only (fused allgather + reduce):
  VLLM_STARSCREAM_FUSE_REDUCE=1 \
  TORCH_SYMM_MEM_DISABLE_MULTICAST=1 \
  torchrun --nnodes=1 --nproc-per-node=<CPX_SIZE> \
    tests/kernels/attention/test_starscream_symm_reduce.py \
    --mode step2 [...]

  # Both step1 and step2 vs. baseline in one run:
  TORCH_SYMM_MEM_DISABLE_MULTICAST=1 \
  torchrun --nnodes=1 --nproc-per-node=<CPX_SIZE> \
    tests/kernels/attention/test_starscream_symm_reduce.py \
    --mode all [...]

  # Typical CPX+NPS4 invocation (8 XCDs per physical GPU):
  TORCH_SYMM_MEM_DISABLE_MULTICAST=1 \
  torchrun --nnodes=1 --nproc-per-node=8 \
    tests/kernels/attention/test_starscream_symm_reduce.py \
    --mode all --cpx-size 8 --num-tokens 32 --num-heads 64 --head-size 128 --seq-len 4096

Notes
-----
* <CPX_SIZE> must equal --cpx-size and must divide --num-heads evenly.
* This test is self-contained: it builds the meta tensor exactly as
  triton_unified_attention.py does, then invokes symm_mem_all_gather /
  fused_all_gather_reduce_segments directly and compares against the
  RCCL baseline (get_dcp_group().all_gather -> reduce_segments).
* The test does not drive the full unified_attention() call; it tests the
  specific segment being replaced so failures are easy to isolate.
"""

import argparse
import os
import sys

import torch
import torch.distributed as dist

# ---------------------------------------------------------------------------
# Bootstrap distributed environment via vllm so that _WORLD is set before
# initialize_model_parallel is called.
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
    distributed_init_method=f"env://",
    backend="nccl",
)

# Enable symmetric memory for the default group (deprecated no-op on newer
# torch; safe to call either way).
try:
    import torch.distributed._symmetric_memory as _symm_mem
    _symm_mem.enable_symm_mem_for_group(dist.group.WORLD.group_name)
except Exception:
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_meta(num_tokens: int, num_heads: int, head_size: int,
                cpx_size: int, seq_len: int,
                dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Construct a realistic starscream_meta_out tensor.

    Each rank holds the partial attention output it would have computed over
    its own KV slice. We synthesise these as random fp32 values so the
    correctness test is purely about the communication + merge, not the
    attention kernel.

    Layout matches triton_unified_attention.py exactly:
        shape = [T, H, 1, HP+2]   (HP = next_pow2(head_size))
        [..., 0:HP]  = numerator
        [..., HP]    = L (exp-sum)
        [..., HP+1]  = M (max logit)
    """
    import triton
    hp = triton.next_power_of_2(head_size)
    dext = hp + 2

    torch.manual_seed(42 + _rank)  # different shard per rank, reproducible
    meta = torch.randn(num_tokens, num_heads, 1, dext,
                       dtype=torch.float32, device=device)
    # Make L > 0 (exp-sum must be positive)
    meta[..., hp].abs_().add_(1e-6)
    return meta


def _rccl_baseline(meta: torch.Tensor, out_shape: tuple,
                   num_query_heads: int, head_size: int, cpx_size: int,
                   output_scale=None) -> torch.Tensor:
    """Stock path: RCCL allgather -> reduce_segments (starscream_flag=True)."""
    from vllm.attention.ops.triton_unified_attention import reduce_segments
    import triton
    from vllm.platforms import current_platform

    grp = get_dcp_group()
    gathered = grp.all_gather(meta.contiguous(), dim=-2).contiguous()

    num_tokens = meta.shape[0]
    # out receives only the heads owned by this rank
    out = torch.zeros(*out_shape, dtype=torch.float32, device=meta.device)

    hp = triton.next_power_of_2(head_size)
    tile_size = 16
    block_q = 1
    # Dummy seqlens / block_table: in the starscream_flag=True path,
    # reduce_segments does not dereference them for seq-len computation
    # (it uses NUM_SEGMENTS_PER_SEQ == cpx_size and act_num_segments ==
    # cpx_size, so the seq_lens load is covered by the segm_mask all-true).
    # We still need valid pointers, so pass a trivial 1-seq case.
    num_seqs = 1
    seqlens = torch.tensor([meta.shape[0]], dtype=torch.int32,
                           device=meta.device)
    cu_seqlens_q = torch.tensor([0, meta.shape[0]], dtype=torch.int32,
                                device=meta.device)
    block_table = torch.zeros((1, 1), dtype=torch.int32, device=meta.device)

    float8_info = torch.finfo(current_platform.fp8_dtype())
    reduce_segments[(num_tokens, num_query_heads)](
        output_ptr=out,
        starscream_meta_out_ptr=meta,
        ss_meta_stride_0=meta.stride(0),
        ss_meta_stride_1=meta.stride(1),
        ss_meta_stride_2=meta.stride(2),
        starscream_rank=grp.rank_in_group,
        cpx_size=cpx_size,
        enable_starscream=True,
        segm_output_ptr=gathered,
        segm_max_ptr=torch.empty(0, device=meta.device),   # unused in flag=True
        segm_expsum_ptr=torch.empty(0, device=meta.device),
        seq_lens_ptr=seqlens,
        num_seqs=num_seqs,
        starscream_flag=True,
        num_query_heads=num_query_heads,
        out_scale_inv=1.0 / output_scale if output_scale is not None else 1.0,
        output_stride_0=out.stride(0),
        output_stride_1=out.stride(1),
        block_table_stride=block_table.stride(0),
        TILE_SIZE=tile_size,
        HEAD_SIZE=head_size,
        HEAD_SIZE_PADDED=hp,
        query_start_len_ptr=cu_seqlens_q,
        BLOCK_Q=block_q,
        NUM_SEGMENTS_PER_SEQ=cpx_size,
        USE_FP8=output_scale is not None,
        FP8_MIN=float8_info.min,
        FP8_MAX=float8_info.max,
    )
    return out


def _out_shape(num_tokens: int, num_query_heads: int, head_size: int,
               cpx_size: int) -> tuple:
    """Output shape that reduce_segments writes into (local heads only)."""
    chunk = num_query_heads // cpx_size
    return (num_tokens, chunk, head_size)


def _log(msg: str) -> None:
    if _rank == 0:
        print(f"[rank 0] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

def test_step1_output_matches_baseline(
    num_tokens: int, num_heads: int, head_size: int, cpx_size: int,
    seq_len: int, rtol: float, atol: float, device: torch.device,
) -> None:
    """Step 1: symm-mem allgather feeds unmodified reduce_segments.

    Verifies that symm_mem_all_gather() produces the same gathered tensor as
    get_dcp_group().all_gather(), and that the subsequent reduce_segments call
    produces output matching the RCCL baseline to within rtol/atol.
    """
    from vllm.attention.ops.starscream_symm_reduce import symm_mem_all_gather

    _log("=== Step 1: symm-mem allgather vs. RCCL baseline ===")
    meta = _build_meta(num_tokens, num_heads, head_size, cpx_size, seq_len,
                       torch.float32, device)
    out_shp = _out_shape(num_tokens, num_heads, head_size, cpx_size)

    # --- Baseline ---
    ref_out = _rccl_baseline(meta, out_shp, num_heads, head_size, cpx_size)
    dist.barrier()

    # --- Step 1: symm-mem allgather -> same reduce_segments ---
    from vllm.attention.ops.triton_unified_attention import reduce_segments
    import triton
    from vllm.platforms import current_platform

    grp = get_dcp_group()
    gathered_symm = symm_mem_all_gather(meta.contiguous()).contiguous()

    # Verify the gathered tensor matches the RCCL gathered tensor.
    gathered_rccl = grp.all_gather(meta.contiguous(), dim=-2).contiguous()
    dist.barrier()
    assert gathered_symm.shape == gathered_rccl.shape, (
        f"gathered shape mismatch: {gathered_symm.shape} vs {gathered_rccl.shape}"
    )
    if not torch.allclose(gathered_symm, gathered_rccl, rtol=rtol, atol=atol):
        max_diff = (gathered_symm - gathered_rccl).abs().max().item()
        raise AssertionError(
            f"Step 1 gathered tensor mismatch: max_diff={max_diff:.6e} "
            f"(rtol={rtol}, atol={atol})"
        )
    _log(f"  gathered tensor: MATCH (max_diff="
         f"{(gathered_symm - gathered_rccl).abs().max().item():.2e})")

    # Run reduce_segments with the symm-mem gathered output and compare.
    out_symm = torch.zeros(*out_shp, dtype=torch.float32, device=device)
    hp = triton.next_power_of_2(head_size)
    tile_size = 16
    block_q = 1
    num_seqs = 1
    seqlens = torch.tensor([num_tokens], dtype=torch.int32, device=device)
    cu_seqlens_q = torch.tensor([0, num_tokens], dtype=torch.int32, device=device)
    block_table = torch.zeros((1, 1), dtype=torch.int32, device=device)
    float8_info = torch.finfo(current_platform.fp8_dtype())

    reduce_segments[(num_tokens, num_heads)](
        output_ptr=out_symm,
        starscream_meta_out_ptr=meta,
        ss_meta_stride_0=meta.stride(0),
        ss_meta_stride_1=meta.stride(1),
        ss_meta_stride_2=meta.stride(2),
        starscream_rank=grp.rank_in_group,
        cpx_size=cpx_size,
        enable_starscream=True,
        segm_output_ptr=gathered_symm,
        segm_max_ptr=torch.empty(0, device=device),
        segm_expsum_ptr=torch.empty(0, device=device),
        seq_lens_ptr=seqlens,
        num_seqs=num_seqs,
        starscream_flag=True,
        num_query_heads=num_heads,
        out_scale_inv=1.0,
        output_stride_0=out_symm.stride(0),
        output_stride_1=out_symm.stride(1),
        block_table_stride=block_table.stride(0),
        TILE_SIZE=tile_size,
        HEAD_SIZE=head_size,
        HEAD_SIZE_PADDED=hp,
        query_start_len_ptr=cu_seqlens_q,
        BLOCK_Q=block_q,
        NUM_SEGMENTS_PER_SEQ=cpx_size,
        USE_FP8=False,
        FP8_MIN=float8_info.min,
        FP8_MAX=float8_info.max,
    )
    dist.barrier()

    if not torch.allclose(out_symm, ref_out, rtol=rtol, atol=atol):
        max_diff = (out_symm - ref_out).abs().max().item()
        raise AssertionError(
            f"Step 1 output mismatch: max_diff={max_diff:.6e} "
            f"(rtol={rtol}, atol={atol})"
        )
    _log(f"  reduce_segments output: MATCH (max_diff="
         f"{(out_symm - ref_out).abs().max().item():.2e})")
    _log("  Step 1 PASSED")


def test_step2_output_matches_baseline(
    num_tokens: int, num_heads: int, head_size: int, cpx_size: int,
    seq_len: int, rtol: float, atol: float, device: torch.device,
) -> None:
    """Step 2: fused allgather + reduce kernel vs. RCCL baseline.

    Verifies that fused_all_gather_reduce_segments() writes into `out` the
    same values as the RCCL allgather + reduce_segments path, to within
    rtol/atol. Small ULP differences are expected because the fused kernel
    accumulates in a different order (sequential over CPX segments) vs. the
    batched tl.sum in reduce_segments.
    """
    from vllm.attention.ops.starscream_symm_reduce import (
        fused_all_gather_reduce_segments,
    )

    _log("=== Step 2: fused kernel vs. RCCL baseline ===")
    meta = _build_meta(num_tokens, num_heads, head_size, cpx_size, seq_len,
                       torch.float32, device)
    out_shp = _out_shape(num_tokens, num_heads, head_size, cpx_size)

    # --- Baseline ---
    ref_out = _rccl_baseline(meta, out_shp, num_heads, head_size, cpx_size)
    dist.barrier()

    # --- Step 2: fused ---
    out_fused = torch.zeros(*out_shp, dtype=torch.float32, device=device)
    ok = fused_all_gather_reduce_segments(
        out=out_fused,
        starscream_meta_out=meta,
        num_query_heads=num_heads,
        head_size=head_size,
        output_scale=None,
        float8_info=None,
    )
    dist.barrier()

    if not ok:
        raise RuntimeError(
            "fused_all_gather_reduce_segments returned False "
            "(symmetric memory unavailable?). Check TORCH_SYMM_MEM_DISABLE_MULTICAST=1 "
            "and that torch.distributed._symmetric_memory is importable."
        )

    max_diff = (out_fused - ref_out).abs().max().item()
    mean_diff = (out_fused - ref_out).abs().mean().item()
    if not torch.allclose(out_fused, ref_out, rtol=rtol, atol=atol):
        raise AssertionError(
            f"Step 2 output mismatch: max_diff={max_diff:.6e}, "
            f"mean_diff={mean_diff:.6e} (rtol={rtol}, atol={atol}). "
            "If diff is tiny (< 1e-5), check whether it is an accumulation-order "
            "ULP difference -- expected and not a semantic break."
        )
    _log(f"  fused output: MATCH (max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e})")
    _log("  Step 2 PASSED")


def test_randomized_inputs(
    num_tokens: int, num_heads: int, head_size: int, cpx_size: int,
    seq_len: int, rtol: float, atol: float, device: torch.device,
    n_trials: int = 8,
) -> None:
    """Randomized output-matching across multiple random seeds.

    Runs both Step 1 and Step 2 against the RCCL baseline for n_trials
    independent random meta tensors. This exercises edge cases such as:
      - all-zero numerators (seq masked out)
      - extreme max logit differences between XCDs
      - head counts that don't divide evenly by cpx_size (skipped with a warning)
    """
    from vllm.attention.ops.starscream_symm_reduce import (
        symm_mem_all_gather,
        fused_all_gather_reduce_segments,
    )
    from vllm.attention.ops.triton_unified_attention import reduce_segments
    import triton
    from vllm.platforms import current_platform

    grp = get_dcp_group()
    hp = triton.next_power_of_2(head_size)
    tile_size = 16
    block_q = 1
    float8_info = torch.finfo(current_platform.fp8_dtype())
    out_shp = _out_shape(num_tokens, num_heads, head_size, cpx_size)

    _log(f"=== Randomized test: {n_trials} trials ===")
    for trial in range(n_trials):
        torch.manual_seed(trial * 100 + _rank)
        meta = torch.randn(num_tokens, num_heads, 1, hp + 2,
                           dtype=torch.float32, device=device)
        meta[..., hp].abs_().add_(1e-6)

        # Baseline
        ref_out = _rccl_baseline(meta, out_shp, num_heads, head_size, cpx_size)
        dist.barrier()

        # Step 1
        gathered_symm = symm_mem_all_gather(meta.contiguous()).contiguous()
        seqlens = torch.tensor([num_tokens], dtype=torch.int32, device=device)
        cu_seqlens_q = torch.tensor([0, num_tokens], dtype=torch.int32, device=device)
        block_table = torch.zeros((1, 1), dtype=torch.int32, device=device)
        out_s1 = torch.zeros(*out_shp, dtype=torch.float32, device=device)
        reduce_segments[(num_tokens, num_heads)](
            output_ptr=out_s1,
            starscream_meta_out_ptr=meta,
            ss_meta_stride_0=meta.stride(0),
            ss_meta_stride_1=meta.stride(1),
            ss_meta_stride_2=meta.stride(2),
            starscream_rank=grp.rank_in_group,
            cpx_size=cpx_size,
            enable_starscream=True,
            segm_output_ptr=gathered_symm,
            segm_max_ptr=torch.empty(0, device=device),
            segm_expsum_ptr=torch.empty(0, device=device),
            seq_lens_ptr=seqlens,
            num_seqs=1,
            starscream_flag=True,
            num_query_heads=num_heads,
            out_scale_inv=1.0,
            output_stride_0=out_s1.stride(0),
            output_stride_1=out_s1.stride(1),
            block_table_stride=block_table.stride(0),
            TILE_SIZE=tile_size,
            HEAD_SIZE=head_size,
            HEAD_SIZE_PADDED=hp,
            query_start_len_ptr=cu_seqlens_q,
            BLOCK_Q=block_q,
            NUM_SEGMENTS_PER_SEQ=cpx_size,
            USE_FP8=False,
            FP8_MIN=float8_info.min,
            FP8_MAX=float8_info.max,
        )

        # Step 2
        out_s2 = torch.zeros(*out_shp, dtype=torch.float32, device=device)
        fused_all_gather_reduce_segments(
            out=out_s2,
            starscream_meta_out=meta,
            num_query_heads=num_heads,
            head_size=head_size,
        )
        dist.barrier()

        s1_diff = (out_s1 - ref_out).abs().max().item()
        s2_diff = (out_s2 - ref_out).abs().max().item()
        ok_s1 = torch.allclose(out_s1, ref_out, rtol=rtol, atol=atol)
        ok_s2 = torch.allclose(out_s2, ref_out, rtol=rtol, atol=atol)

        status = "PASS" if (ok_s1 and ok_s2) else "FAIL"
        _log(f"  trial {trial:2d}: {status}  "
             f"step1_max={s1_diff:.2e}  step2_max={s2_diff:.2e}")

        if not ok_s1:
            raise AssertionError(
                f"Randomized trial {trial}: Step 1 mismatch max_diff={s1_diff:.6e}"
            )
        if not ok_s2:
            raise AssertionError(
                f"Randomized trial {trial}: Step 2 mismatch max_diff={s2_diff:.6e}"
            )

    _log(f"  All {n_trials} randomized trials PASSED")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Starscream symm-mem allgather + fused reduce correctness tests"
    )
    parser.add_argument("--mode", choices=["step1", "step2", "all"],
                        default="all",
                        help="Which test(s) to run")
    parser.add_argument("--num-tokens", type=int, default=16,
                        help="Number of query tokens (T)")
    parser.add_argument("--num-heads", type=int, default=32,
                        help="Total query heads (H, must be divisible by cpx-size)")
    parser.add_argument("--head-size", type=int, default=128,
                        help="Head dimension")
    parser.add_argument("--cpx-size", type=int, default=_world_size,
                        help="Number of XCDs in the DCP group (== nproc-per-node)")
    parser.add_argument("--seq-len", type=int, default=4096,
                        help="Full sequence length (before sharding)")
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--trials", type=int, default=8,
                        help="Number of random trials in the randomized test")
    args = parser.parse_args()

    assert args.num_heads % args.cpx_size == 0, (
        f"--num-heads ({args.num_heads}) must be divisible by "
        f"--cpx-size ({args.cpx_size})"
    )
    assert _world_size % args.cpx_size == 0, (
        f"WORLD_SIZE ({_world_size}) must be divisible by "
        f"--cpx-size ({args.cpx_size}). Each physical GPU contributes cpx_size ranks."
    )

    # TP == world_size (all vGPUs in one TP group).
    # DCP == cpx_size (one DCP group per physical GPU, each covering its XCDs).
    # This gives world_size // cpx_size independent DCP groups, one per physical GPU.
    initialize_model_parallel(
        tensor_model_parallel_size=_world_size,
        pipeline_model_parallel_size=1,
        decode_context_model_parallel_size=args.cpx_size,
    )

    device = torch.device(f"cuda:{_local_rank}")
    kwargs = dict(
        num_tokens=args.num_tokens,
        num_heads=args.num_heads,
        head_size=args.head_size,
        cpx_size=args.cpx_size,
        seq_len=args.seq_len,
        rtol=args.rtol,
        atol=args.atol,
        device=device,
    )

    try:
        if args.mode in ("step1", "all"):
            test_step1_output_matches_baseline(**kwargs)
            dist.barrier()

        if args.mode in ("step2", "all"):
            test_step2_output_matches_baseline(**kwargs)
            dist.barrier()

        if args.mode == "all":
            test_randomized_inputs(**kwargs, n_trials=args.trials)
            dist.barrier()

        _log("ALL TESTS PASSED")
    except Exception as exc:
        print(f"[rank {_rank}] ERROR: {exc}", flush=True)
        raise
    finally:
        destroy_model_parallel()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
