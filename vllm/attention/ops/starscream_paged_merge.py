# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cross-XCD (CPX / Starscream) merge for the HIP paged-attention kernel.

In CPX mode one physical MI300X is exposed as 8 devices, one per XCD. Each rank
runs the ordinary ROCm paged-attention decode kernel over its own slice of the
KV context. Every rank's `paged_attention_ll4mi_reduce_kernel` then emits a
partial-softmax record instead of a final output:

    [0 : head_size]   un-normalized numerator, rescaled to this rank's max
    [head_size]       L, the local exp_sum (rescaled to the same max)
    [head_size + 1]   M, the local max logit

This module all-gathers those records and finishes the softmax. It is the same
math as the second `reduce_segments` call on the Triton path, but kept
standalone: the Triton record layout is tied to that kernel's `HEAD_SIZE_PADDED`
and segment masking, which the HIP producer does not share.
"""

import torch
import triton
import triton.language as tl

# Finite stand-in for -inf, matching SS_NEG_HUGE in csrc/rocm/attention.cu.
# Keeps an all-empty merge at exp(M - M) = 1 rather than NaN.
NEG_HUGE = -3.0e38

# Extra fp32 slots appended to each (seq, head) record: L and M.
META_EXTRA = 2


@triton.jit
def _starscream_merge_kernel(
    meta_ptr,  # [world_size, num_seqs, num_heads, HEAD_SIZE + 2] fp32
    out_ptr,  # [num_seqs, num_heads, HEAD_SIZE]
    stride_rank,  # elements between consecutive ranks in meta
    stride_out_row,  # elements between consecutive (seq, head) rows in out
    HEAD_SIZE: tl.constexpr,
    RECORD: tl.constexpr,  # HEAD_SIZE + 2
    WORLD_SIZE: tl.constexpr,
):
    # One program per (seq, head) pair.
    pid = tl.program_id(0)
    offs = tl.arange(0, HEAD_SIZE)

    acc = tl.zeros([HEAD_SIZE], dtype=tl.float32)
    l_g = 0.0
    # Literal rather than the module constant: keeps this kernel free of
    # captured globals. Must match SS_NEG_HUGE in csrc/rocm/attention.cu.
    m_g = -3.0e38

    for r in tl.static_range(WORLD_SIZE):
        base = meta_ptr + r * stride_rank + pid * RECORD
        num = tl.load(base + offs).to(tl.float32)
        l_r = tl.load(base + HEAD_SIZE).to(tl.float32)
        m_r = tl.load(base + HEAD_SIZE + 1).to(tl.float32)

        m_new = tl.maximum(m_g, m_r)
        alpha = tl.exp(m_g - m_new)
        beta = tl.exp(m_r - m_new)
        acc = acc * alpha + num * beta
        l_g = l_g * alpha + l_r * beta
        m_g = m_new

    # Exact-zero guard rather than the HIP reducer's `+ 1e-6`. Two reasons:
    # an all-empty row yields 0 either way, and the epsilon puts a systematic
    # ~1e-6/L bias on every element -- which would show up as a constant offset
    # when A/B-ing this path against the fused symm-mem merge in
    # starscream_paged_symm_merge, whose guard is exactly this. The two merges
    # still differ in rounding (sequential recurrence here, two-pass rescale
    # there), so they are close-but-not-bit-equal by construction; this at
    # least removes the one difference that is a bias rather than noise.
    res = tl.where(l_g == 0.0, 0.0, acc / l_g)
    tl.store(out_ptr + pid * stride_out_row + offs, res.to(out_ptr.dtype.element_ty))


def make_meta_buffer(
    num_seqs: int,
    num_heads: int,
    head_size: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Allocate the per-rank record buffer passed to `paged_attention_rocm`."""
    return torch.empty(
        (num_seqs, num_heads, head_size + META_EXTRA),
        dtype=torch.float32,
        device=device,
    )


def starscream_merge(
    meta_local: torch.Tensor,
    out: torch.Tensor,
    world_size: int,
    group=None,
    gathered: torch.Tensor | None = None,
) -> torch.Tensor:
    """All-gather the per-rank records and write the merged attention output.

    Args:
        meta_local: [num_seqs, num_heads, head_size + 2] fp32, this rank's
            record as emitted by the HIP reduce kernel.
        out: [num_seqs, num_heads, head_size] destination.
        world_size: number of CPX ranks (XCDs) participating.
        group: optional torch.distributed process group.
        gathered: optional pre-allocated
            [world_size * num_seqs, num_heads, head_size + 2] scratch buffer,
            so the steady-state path allocates nothing.
    """
    num_seqs, num_heads, record = meta_local.shape
    head_size = record - META_EXTRA
    assert out.shape == (num_seqs, num_heads, head_size), out.shape

    if world_size > 1:
        import torch.distributed as dist

        meta_local = meta_local.contiguous()
        if gathered is None:
            gathered = torch.empty(
                (world_size * num_seqs, num_heads, record),
                dtype=meta_local.dtype,
                device=meta_local.device,
            )
        dist.all_gather_into_tensor(gathered, meta_local, group=group)
    else:
        gathered = meta_local.contiguous()

    grid = (num_seqs * num_heads,)
    _starscream_merge_kernel[grid](
        gathered,
        out,
        num_seqs * num_heads * record,
        head_size,
        HEAD_SIZE=head_size,
        RECORD=record,
        WORLD_SIZE=world_size,
        num_warps=1,
    )
    return out
