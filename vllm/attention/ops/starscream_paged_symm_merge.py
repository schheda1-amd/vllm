# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused symmetric-memory cross-XCD merge for the HIP paged-attention kernel.

RCCL-free counterpart to :mod:`starscream_paged_merge`. That module does
``dist.all_gather_into_tensor`` into a `[world, num_seqs, num_heads, Dext]`
scratch buffer and then merges it; this one skips both the collective and the
scratch buffer, reading every peer XCD's record directly through
symmetric-memory peer pointers inside a single Triton kernel.

Relationship to the Triton path
-------------------------------
This is deliberately a near-duplicate of
``starscream_symm_reduce.fused_all_gather_reduce_segments``, which serves the
Triton ``unified_attention`` producer. The two producers disagree on the record
shape (`[T, H, 1, next_pow2(D) + 2]` there, `[num_seqs, num_heads, D + 2]` here)
and on who owns the staging allocation, so for POC purposes the kernels are kept
separate rather than generalised. The *math* is identical - same two-pass
rescale-to-common-max formulation, same ``== 0`` denominator guard - so the two
paths are semantically interchangeable.

Zero-copy production
--------------------
The Triton path stages its partials with
``buf.buffer[:numel].copy_(meta.reshape(-1))`` because ``unified_attention``
allocated the meta tensor before symm-mem was in the picture. Here the caller
gets the symm-mem buffer *first* (:func:`get_symm_meta`) and hands that view to
``paged_attention_rocm`` as ``starscream_meta_out``, so the HIP reduce kernel
writes its record straight into symmetric memory. At batch 1024 / 128 heads /
head_size 128 the record is 68 MiB per rank, so that is 136 MiB of copy traffic
removed from every decode step.

Head-sharded aggregation
------------------------
Every rank holds a partial record for *all* ``num_heads`` heads -- that is
forced by the input side, since a rank owns a non-overlapping KV slice and must
attend every query head against it. But the *combine* for head ``h`` is a fixed
piece of arithmetic over the 8 records for that head, and only one rank needs to
run it.

With ``SCATTER_HEADS=True`` the grid is ``(num_seqs, num_heads // CPX)`` and rank
``r`` merges only heads ``[r*chunk, (r+1)*chunk)``, writing them at local indices
``[0, chunk)``. The physical device still ends up owning all ``num_heads``
outputs, exactly once -- which is what SPX produces, and what a TP-sharded
``o_proj`` consumes. It is the inverse of the query all-gather, and it is free:
a grid bound, not a second collective.

With ``SCATTER_HEADS=False`` every rank merges every head, so the device
produces ``CPX`` identical copies and pulls ``CPX`` times the peer traffic for
the same answer. That mode exists so a rank's output can be diffed against a
single-rank full-context reference; it is not the shape you want to benchmark.

Record layout (must match ``ss_meta_out`` in csrc/rocm/attention.cu)
--------------------------------------------------------------------
Per (seq, head), ``HEAD_SIZE + 2`` contiguous fp32 slots::

    [0 : HEAD_SIZE]   numerator = sum_j exp(s_j - M_local) * V_j
    [HEAD_SIZE]       L         = sum_j exp(s_j - M_local)
    [HEAD_SIZE + 1]   M         = M_local   (SS_NEG_HUGE if the rank is empty)

Note the stride is ``HEAD_SIZE + 2``, which is never a power of two, so records
do not start on a 128B boundary. The Triton path has exactly the same property
(``Dext = HP + 2``); it costs a little coalescing and is left alone for the POC.
"""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _fused_paged_merge_kernel(
    out_ptr,  # [num_seqs, num_heads_out, HEAD_SIZE] final attention output
    peer_ptrs,  # [CPX] int64 base pointers of every rank's symm-mem buffer
    starscream_rank,  # this rank's index within the group
    num_heads,  # H, the global (pre-scatter) head count
    out_stride_0: tl.int64,  # elements between consecutive sequences in out
    out_stride_1: tl.int64,  # elements between consecutive heads in out
    HEAD_SIZE: tl.constexpr,
    DEXT: tl.constexpr,  # HEAD_SIZE + 2
    BLOCK_D: tl.constexpr,  # next_pow2(HEAD_SIZE)
    CPX: tl.constexpr,  # number of XCDs == group world size
    SCATTER_HEADS: tl.constexpr,  # merge only this rank's head shard
):
    seq_idx = tl.program_id(0)

    if SCATTER_HEADS:
        # Grid dim 1 is num_heads // CPX, so program_id(1) is a LOCAL head
        # index. Lift it to the global index the peer records are addressed by;
        # the local index is where it lands in `out`. Because the grid itself is
        # narrowed, the peer loads below are never issued for heads this rank
        # does not own -- that is where the CPX-fold traffic saving comes from.
        # Masking the store alone would not save a byte.
        out_head = tl.program_id(1)
        head_idx = starscream_rank * (num_heads // CPX) + out_head
    else:
        # POC/benchmark mode: every rank merges the full head range, so the
        # result can be diffed against a full-context single-rank reference.
        head_idx = tl.program_id(1)
        out_head = head_idx

    d = tl.arange(0, BLOCK_D)
    dim_mask = d < HEAD_SIZE

    # Every peer's buffer holds its own [num_seqs, num_heads, DEXT] record, so
    # the per-(seq, head) source offset is the same on all of them.
    src_base = seq_idx.to(tl.int64) * (num_heads * DEXT) + head_idx * DEXT

    # Pass 1: global max over the per-XCD local maxima.
    #
    # An empty rank emits M = SS_NEG_HUGE (-3e38) with L = 0 and a zero
    # numerator. That stays finite through `m_p - overall_max`, so the rescale
    # below underflows to 0 rather than producing NaN, and the rank drops out.
    overall_max = float("-inf")
    for p in tl.static_range(CPX):
        peer = tl.load(peer_ptrs + p).to(tl.pointer_type(tl.float32))
        overall_max = tl.maximum(
            overall_max, tl.load(peer + src_base + HEAD_SIZE + 1)
        )

    # Pass 2: rescale each XCD's numerator and exp-sum to the common max and
    # accumulate. Same formulation as reduce_segments(starscream_flag=True).
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    overall_expsum = tl.zeros([], dtype=tl.float32)
    for p in tl.static_range(CPX):
        peer = tl.load(peer_ptrs + p).to(tl.pointer_type(tl.float32))
        num_p = tl.load(peer + src_base + d, mask=dim_mask, other=0.0)
        l_p = tl.load(peer + src_base + HEAD_SIZE)
        m_p = tl.load(peer + src_base + HEAD_SIZE + 1)
        scale = tl.exp(m_p - overall_max)
        acc += num_p * scale
        overall_expsum += l_p * scale

    # Exact-zero guard rather than the HIP reducer's `+ 1e-6`: an all-empty row
    # yields 0 either way, and dropping the epsilon removes a ~1e-6/L bias.
    acc = tl.where(overall_expsum == 0.0, 0.0, acc / overall_expsum)

    # Every program stores unconditionally under both modes: the head shard is
    # selected by the grid bound above, not by a predicate here.
    off = seq_idx.to(tl.int64) * out_stride_0 + out_head * out_stride_1 + d
    tl.store(out_ptr + off, acc.to(out_ptr.dtype.element_ty), mask=dim_mask)


def get_symm_meta(
    num_seqs: int,
    num_heads: int,
    head_size: int,
    group,
):
    """Allocate the Starscream record *inside* symmetric memory.

    Returns ``(meta_view, buf)`` where ``meta_view`` is a
    ``[num_seqs, num_heads, head_size + 2]`` fp32 view to hand to
    ``paged_attention_rocm(..., starscream_meta_out=meta_view)``, and ``buf`` is
    the :class:`SymmMemAllGatherBuffer` to pass to :func:`fused_paged_merge`.

    Returns ``(None, None)`` when symmetric memory is unavailable or the group
    has a single rank, so the caller can fall back to the RCCL path.

    ``group`` needs only the GroupCoordinator surface the symm-mem manager
    touches: ``device_group``, ``unique_name``, ``world_size``,
    ``rank_in_group``, ``device``.
    """
    from vllm.distributed.device_communicators.symm_mem_allgather import (
        get_symm_mem_allgather_manager,
    )

    mgr = get_symm_mem_allgather_manager(group)
    if mgr is None:
        return None, None

    numel = num_seqs * num_heads * (head_size + 2)
    buf = mgr.get_buffer(numel, torch.float32)
    meta = buf.local_view((num_seqs, num_heads, head_size + 2))
    return meta, buf


def fused_paged_merge(
    out: torch.Tensor,
    buf,
    group,
    num_seqs: int,
    num_heads: int,
    head_size: int,
    scatter_heads: bool = False,
    host_barrier: bool = False,
) -> torch.Tensor:
    """Merge every XCD's record straight out of peer symmetric memory.

    No RCCL collective and no gathered scratch tensor: the kernel dereferences
    ``buf.peer_ptrs`` directly. ``buf`` must be the buffer whose ``local_view``
    was passed to ``paged_attention_rocm`` as ``starscream_meta_out``, so the
    HIP reduce kernel has already published this rank's record in place.

    ``num_heads`` is always the GLOBAL head count -- it addresses the peer
    records, whose layout does not change with the output sharding.

    ``scatter_heads`` selects the output ownership:

    * ``True``  -- this rank merges heads ``[rank*chunk, (rank+1)*chunk)`` and
      ``out`` must be ``[num_seqs, num_heads // world, head_size]``. Each head
      is merged once across the device, matching what SPX produces and what a
      TP-sharded ``o_proj`` consumes. Peer traffic is ``world``x lower.
    * ``False`` -- every rank merges every head and ``out`` must be full width.
      Diffable against a single-rank reference; ``world``x redundant.

    Uses the on-device signal-pad barrier unconditionally rather than
    ``starscream_symm_reduce._barrier``. That helper honours
    ``VLLM_STARSCREAM_SIGNAL_PAD_BARRIER``, which defaults to False and would
    put *two* ``dist.barrier`` RCCL collectives per decode step on the critical
    path -- worse than the single all_gather this path exists to remove. Set
    ``host_barrier=True`` only to A/B the barrier itself.
    """
    from vllm.attention.ops.starscream_symm_reduce import signal_pad_barrier

    def barrier():
        if host_barrier:
            from vllm.distributed.device_communicators.symm_mem_allgather import (
                symm_mem_barrier,
            )

            symm_mem_barrier(group)
        else:
            signal_pad_barrier(buf)

    cpx = buf.world_size

    if scatter_heads:
        if num_heads % cpx != 0:
            raise ValueError(
                f"scatter_heads needs num_heads ({num_heads}) divisible by the "
                f"CPX world size ({cpx}); pad or fall back to scatter_heads=False"
            )
        heads_out = num_heads // cpx
    else:
        heads_out = num_heads
    if out.shape[:2] != (num_seqs, heads_out):
        raise ValueError(
            f"out has shape {tuple(out.shape)}; expected "
            f"({num_seqs}, {heads_out}, {head_size}) for "
            f"scatter_heads={scatter_heads}"
        )

    # Barrier 1: every peer's record is visible before anyone reads it.
    barrier()

    # Grid dim 1 is the number of heads THIS rank merges. Narrowing it is the
    # whole optimisation: an unlaunched program issues no peer loads.
    _fused_paged_merge_kernel[(num_seqs, heads_out)](
        out_ptr=out,
        peer_ptrs=buf.peer_ptrs,
        starscream_rank=buf.rank_in_group,
        num_heads=num_heads,
        out_stride_0=out.stride(0),
        out_stride_1=out.stride(1),
        HEAD_SIZE=head_size,
        DEXT=head_size + 2,
        BLOCK_D=triton.next_power_of_2(head_size),
        CPX=cpx,
        SCATTER_HEADS=scatter_heads,
        num_warps=1,
    )

    # Barrier 2: no rank may start the next decode step -- which writes its
    # record straight back into this buffer -- until every peer has finished
    # reading. Load-bearing here in a way it is not on the Triton path, where
    # the producer writes to a separate tensor and only a copy touches symm-mem.
    #
    # This is the price of the zero-copy production above, and it is the one
    # place the fused path could still be tightened: double-buffering the record
    # (step N+1 publishes into the other bank) would make barrier 2 redundant
    # and halve the barrier count. That needs a *device-side* bank toggle, not a
    # host-side one -- under CUDA graph capture host Python runs once, so a host
    # toggle would freeze at its capture value. Same trap the signal pad's phase
    # counter already documents. Left for the productisation pass.
    barrier()
    return out
