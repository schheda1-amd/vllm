# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Symmetric-memory allgather + fused reduce for Starscream (CPX+NPS4).

This module supplies two entry points used by ``unified_attention`` to replace
the RCCL/NCCL cross-XCD allgather that precedes ``reduce_segments`` in the
Starscream decode path:

1. :func:`symm_mem_all_gather` - a drop-in replacement for
   ``get_dcp_group().all_gather(meta, dim=-2)``. Same in/out shapes and same
   concatenation order (segment index == group rank). The gathered tensor is
   still consumed by the *unmodified* ``reduce_segments`` kernel.

2. :func:`fused_all_gather_reduce_segments` - fuses the allgather and the
   ``reduce_segments`` online-softmax merge into a single Triton kernel that
   reads peer XCD buffers directly via symmetric-memory peer pointers, performs
   the exact same max/rescale/sum/normalize math, and scatters each rank's
   owned heads into ``out``. The original ``reduce_segments`` is never touched.

Memory layout (must match the producer in triton_unified_attention.py)
----------------------------------------------------------------------
Each rank produces ``starscream_meta_out`` of shape ``[T, H, 1, Dext]`` (fp32),
with ``Dext = next_pow2(head_size) + 2`` and, per (token, head), the ``Dext``
contiguous slots laid out as::

    [0 : HP]     numerator  = sum_j exp(s_j - m_local) * V_j
    [HP]         L          = sum_j exp(s_j - m_local)        (exp-sum)
    [HP + 1]     M          = m_local                          (max logit)

where ``HP = next_pow2(head_size)``. The merge rescales every XCD's partials to
a common max before dividing - the online-softmax invariant.
"""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton


# ---------------------------------------------------------------------------
# On-device signal-pad barrier (RCCL-free replacement for dist.barrier)
# ---------------------------------------------------------------------------
# CUDA-graph-safe, sense-reversing (double-buffered) all-to-all barrier.
#
# Why not a host generation counter: under CUDA graph capture the host Python
# runs ONCE, so a `gen` passed as a kernel argument freezes at its capture value
# and the barrier becomes a no-op on every replay (each wait sees the previous
# replay's flags already satisfied). The barrier must carry NO host state. Here
# the phase lives in device memory and the KERNEL advances it, so every graph
# replay re-executes a correct, freshly-phased barrier. (Pattern adapted from
# the reference rs+gemm_symm.py sync_kernel, extended to double-buffering so it
# is also safe in a tight per-iteration replay loop with no work between
# barriers.)
#
# Protocol per call:
#   * phase = *phase_ptr;  bank = phase & 1;  base = bank * WORLD_SIZE
#   * arrive: atomic_add(+1) into slot `my_rank` of EVERY peer's pad in this
#     bank, release+sys -> publishes our prior data store and announces arrival.
#   * wait: spin until ALL WORLD_SIZE slots of MY OWN bank == WORLD_SIZE
#     (every rank, including self, added 1), acquire+sys -> peers' data visible.
#   * reset: zero MY OWN bank's slots for reuse two barriers later, then store
#     phase+1. Alternating banks guarantee a fast rank's next arrival lands in
#     the OTHER bank, so it can't clobber a slot a slow rank is still reading,
#     and this rank's reset of bank b only races with uses of bank b that are
#     >= 2 barriers away (impossible: a peer can't lap by two barriers).
#   * Single program, single warp -> no inter-CTA co-residency requirement,
#     deadlock-free.
@triton.jit
def _signal_pad_barrier_kernel(
    signal_peer_ptrs,   # [W] int64 base pointers of every rank's 2-bank pad
    phase_ptr,          # int32[1] device-side local phase counter
    my_rank,            # this rank's index in the group
    WORLD_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,  # next_pow2(WORLD_SIZE), lane block for the W peers
):
    peer = tl.arange(0, BLOCK)
    mask = peer < WORLD_SIZE

    # Device-side phase -> bank parity. Advances on every (re)play.
    phase = tl.load(phase_ptr)
    base = (phase % 2) * WORLD_SIZE

    # --- arrive: +1 into slot `my_rank` of every peer's pad, this bank ---
    peer_base = tl.load(signal_peer_ptrs + peer, mask=mask, other=0)
    flag_ptr = peer_base.to(tl.pointer_type(tl.int32)) + base + my_rank
    tl.atomic_add(flag_ptr, 1, mask=mask, sem="release", scope="sys")

    # --- wait: all WORLD_SIZE slots of my own bank == WORLD_SIZE ---
    my_base = tl.load(signal_peer_ptrs + my_rank).to(tl.pointer_type(tl.int32))
    slot_ptr = my_base + base + peer
    done = 0
    while done == 0:
        vals = tl.atomic_add(slot_ptr, 0, mask=mask, sem="acquire", scope="sys")
        # masked-out lanes forced to WORLD_SIZE so they never gate the min.
        minv = tl.min(tl.where(mask, vals, WORLD_SIZE))
        done = (minv >= WORLD_SIZE).to(tl.int32)

    # --- reset my bank for reuse, and advance phase for the next call ---
    tl.store(slot_ptr, 0, mask=mask)
    tl.store(phase_ptr, phase + 1)


def signal_pad_barrier(buf) -> None:
    """On-device, CUDA-graph-safe barrier over the group via a symm-mem pad.

    Replaces ``dist.barrier`` (an RCCL collective) with a single-warp Triton
    kernel that arrives/waits through symmetric-memory peer pointers, keeping
    RCCL out of the merge critical path. Sense-reversing + device-side phase, so
    it is correct under CUDA graph replay. ``buf`` is a SymmMemAllGatherBuffer.
    """
    _signal_pad_barrier_kernel[(1,)](
        signal_peer_ptrs=buf.signal_peer_ptrs,
        phase_ptr=buf.phase,
        my_rank=buf.rank_in_group,
        WORLD_SIZE=buf.world_size,
        BLOCK=triton.next_power_of_2(buf.world_size),
        num_warps=1,
    )


def _barrier(group, buf) -> None:
    """Bracket barrier for the symm-mem allgather.

    Dispatches to the on-device signal-pad barrier when
    VLLM_STARSCREAM_SIGNAL_PAD_BARRIER is set (keeps RCCL off the critical
    path), else the host dist.barrier. Both give the same happens-before
    guarantee around the peer copy/read.
    """
    import vllm.envs as envs

    from vllm.distributed.device_communicators.symm_mem_allgather import (
        symm_mem_barrier,
    )

    if envs.VLLM_STARSCREAM_SIGNAL_PAD_BARRIER:
        signal_pad_barrier(buf)
    else:
        symm_mem_barrier(group)


# ---------------------------------------------------------------------------
# Step 1: symmetric-memory allgather (drop-in for get_dcp_group().all_gather)
# ---------------------------------------------------------------------------
@triton.jit
def _symm_allgather_kernel(
    out_ptr,          # [T, H, CPX, Dext] contiguous, local output
    peer_ptrs,        # [CPX] int64 base pointers of each peer's staging buffer
    num_rows,         # T * H
    CPX: tl.constexpr,      # number of XCDs in the group (== world size)
    DEXT: tl.constexpr,     # extended head dim (next_pow2(head)+2)
    BLOCK_D: tl.constexpr,  # next_pow2(DEXT)
):
    row = tl.program_id(0)      # flattened (token, head) index
    peer = tl.program_id(1)     # source XCD / group rank

    if row >= num_rows:
        return

    d = tl.arange(0, BLOCK_D)
    d_mask = d < DEXT

    # Each peer's staging buffer holds its own [T, H, 1, Dext] meta, so the
    # per-(token,head) source offset is identical across peers.
    src_base = row * DEXT
    peer_base = tl.load(peer_ptrs + peer)
    peer_ptr = peer_base.to(tl.pointer_type(tl.float32))
    vals = tl.load(peer_ptr + src_base + d, mask=d_mask, other=0.0)

    # Destination: segment `peer` of the [T, H, CPX, Dext] output.
    dst_base = row * (CPX * DEXT) + peer * DEXT
    tl.store(out_ptr + dst_base + d, vals, mask=d_mask)


def symm_mem_all_gather(meta: torch.Tensor) -> torch.Tensor:
    """Allgather ``meta`` [T, H, 1, Dext] over the DCP group via symm-mem.

    Returns a contiguous [T, H, CPX, Dext] tensor identical in layout to
    ``get_dcp_group().all_gather(meta, dim=-2)``. Falls back to the RCCL
    collective if symmetric memory is unavailable.
    """
    from vllm.distributed.device_communicators.symm_mem_allgather import (
        get_symm_mem_allgather_manager,
    )
    from vllm.distributed.parallel_state import get_dcp_group

    group = get_dcp_group()
    mgr = get_symm_mem_allgather_manager(group)
    if mgr is None:
        # No symm-mem: preserve semantics with the stock collective.
        return group.all_gather(meta.contiguous(), dim=-2).contiguous()

    T, H, one, dext = meta.shape
    assert one == 1, f"expected meta dim -2 == 1, got {one}"
    cpx = group.world_size
    numel = T * H * 1 * dext

    buf = mgr.get_buffer(numel, meta.dtype)
    # Publish this rank's shard into its symm-mem staging buffer.
    buf.buffer[:numel].copy_(meta.reshape(-1))
    _barrier(group, buf)

    out = torch.empty((T, H, cpx, dext), dtype=meta.dtype, device=meta.device)
    num_rows = T * H
    block_d = triton.next_power_of_2(dext)
    _symm_allgather_kernel[(num_rows, cpx)](
        out_ptr=out,
        peer_ptrs=buf.peer_ptrs,
        num_rows=num_rows,
        CPX=cpx,
        DEXT=dext,
        BLOCK_D=block_d,
    )
    # Ensure no peer frees/overwrites its staging buffer before reads complete.
    _barrier(group, buf)
    return out


# ---------------------------------------------------------------------------
# Step 2: fused allgather + reduce_segments (single kernel)
# ---------------------------------------------------------------------------
@triton.jit
def _fused_allgather_reduce_kernel(
    out_ptr,            # [T, H_local, head_size] final attention output
    peer_ptrs,          # [CPX] int64 base pointers of peer staging buffers
    out_scale_inv,      # float32 (only used when USE_FP8)
    starscream_rank,    # this rank's index within the group
    num_query_heads,    # H (global, pre-scatter head count)
    output_stride_0: tl.int64,
    output_stride_1: tl.int64,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,  # HP
    DEXT: tl.constexpr,              # HP + 2
    CPX: tl.constexpr,               # segments == group world size
    USE_FP8: tl.constexpr,
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    query_token_idx = tl.program_id(0)
    query_head_idx = tl.program_id(1)

    d = tl.arange(0, HEAD_SIZE_PADDED)
    dim_mask = d < HEAD_SIZE

    # Per-(token, head) offset is the same inside every peer's staging buffer,
    # which holds that peer's [T, H, 1, Dext] partials.
    src_base = (query_token_idx.to(tl.int64) * (num_query_heads * DEXT)
                + query_head_idx * DEXT)

    # Pass 1: global max across the CPX per-XCD local maxima. Mirrors
    # `overall_max = tl.max(segm_max)` in reduce_segments(flag=True).
    overall_max = float("-inf")
    for p in tl.static_range(CPX):
        peer_ptr = tl.load(peer_ptrs + p).to(tl.pointer_type(tl.float32))
        m_p = tl.load(peer_ptr + src_base + HEAD_SIZE_PADDED + 1)
        overall_max = tl.maximum(overall_max, m_p)

    # Pass 2: rescale each XCD's numerator and exp-sum to the common max and
    # accumulate. Mirrors the batch formulation exactly (rescale then sum).
    acc = tl.zeros([HEAD_SIZE_PADDED], dtype=tl.float32)
    overall_expsum = tl.zeros([], dtype=tl.float32)
    for p in tl.static_range(CPX):
        peer_ptr = tl.load(peer_ptrs + p).to(tl.pointer_type(tl.float32))
        num_p = tl.load(peer_ptr + src_base + d, mask=dim_mask, other=0.0)
        l_p = tl.load(peer_ptr + src_base + HEAD_SIZE_PADDED)
        m_p = tl.load(peer_ptr + src_base + HEAD_SIZE_PADDED + 1)
        scale = tl.exp(m_p - overall_max)
        acc += num_p * scale
        overall_expsum += l_p * scale

    # Safe normalize (== reduce_segments: 0.0 when denom is 0).
    acc = tl.where(overall_expsum == 0.0, 0.0, acc / overall_expsum)

    # Head scatter: this rank owns heads [rank*chunk, (rank+1)*chunk).
    chunk_size = num_query_heads // CPX
    chunk_start = starscream_rank * chunk_size
    chunk_end = chunk_start + chunk_size
    in_rank = (query_head_idx >= chunk_start) & (query_head_idx < chunk_end)
    local_head_idx = query_head_idx - chunk_start
    output_mask = dim_mask & in_rank

    if USE_FP8:
        acc = acc * tl.load(out_scale_inv)
        acc = tl.clamp(acc, FP8_MIN, FP8_MAX)

    output_offset = (query_token_idx * output_stride_0
                     + local_head_idx * output_stride_1
                     + d)
    tl.store(out_ptr + output_offset, acc, mask=output_mask)


def fused_all_gather_reduce_segments(
    out: torch.Tensor,               # [T, H_local, head_size] destination
    starscream_meta_out: torch.Tensor,  # [T, H, 1, Dext] local partials
    num_query_heads: int,            # H (global)
    head_size: int,
    output_scale=None,               # optional fp8 out scale (float tensor)
    float8_info=None,                # torch.finfo for the fp8 dtype
) -> bool:
    """Fused symm-mem allgather + online-softmax merge + head scatter.

    Writes the normalized attention output for this rank's owned heads into
    ``out`` and returns True. Returns False (doing nothing) if symmetric memory
    is unavailable, so the caller can fall back to the allgather +
    ``reduce_segments`` path.
    """
    from vllm.distributed.device_communicators.symm_mem_allgather import (
        get_symm_mem_allgather_manager,
    )
    from vllm.distributed.parallel_state import get_dcp_group

    group = get_dcp_group()
    mgr = get_symm_mem_allgather_manager(group)
    if mgr is None:
        return False

    T, H, one, dext = starscream_meta_out.shape
    assert one == 1, f"expected meta dim -2 == 1, got {one}"
    assert H == num_query_heads, (
        f"meta head dim {H} != num_query_heads {num_query_heads}"
    )
    cpx = group.world_size
    hp = dext - 2
    numel = T * H * 1 * dext

    buf = mgr.get_buffer(numel, starscream_meta_out.dtype)
    buf.buffer[:numel].copy_(starscream_meta_out.reshape(-1))
    _barrier(group, buf)

    use_fp8 = output_scale is not None
    fp8_min = float8_info.min if float8_info is not None else 0.0
    fp8_max = float8_info.max if float8_info is not None else 0.0

    _fused_allgather_reduce_kernel[(T, num_query_heads)](
        out_ptr=out,
        peer_ptrs=buf.peer_ptrs,
        out_scale_inv=(1.0 / output_scale) if use_fp8 else 1.0,
        starscream_rank=group.rank_in_group,
        num_query_heads=num_query_heads,
        output_stride_0=out.stride(0),
        output_stride_1=out.stride(1),
        HEAD_SIZE=head_size,
        HEAD_SIZE_PADDED=hp,
        DEXT=dext,
        CPX=cpx,
        USE_FP8=use_fp8,
        FP8_MIN=fp8_min,
        FP8_MAX=fp8_max,
    )
    # Guard peer staging buffers until every rank has finished reading.
    _barrier(group, buf)
    return True
