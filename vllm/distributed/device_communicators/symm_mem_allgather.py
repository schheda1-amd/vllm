# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Symmetric-memory allgather infrastructure for Starscream (CPX+NPS4).

This module provides the plumbing needed to run an intra-physical-GPU,
inter-XCD allgather over the DCP ``GroupCoordinator`` using PyTorch
symmetric memory (``torch.distributed._symmetric_memory``) instead of the
RCCL/NCCL collective. It is the setup layer consumed by the fused Triton
kernels in ``vllm/attention/ops/starscream_symm_reduce.py``.

Design notes
------------
* Buffers are allocated lazily and cached per ``(numel, dtype)``. vLLM batch
  shapes vary token-to-token, so we key on element count rather than a fixed
  preallocation.
* We expose per-peer device pointers (``hdl.buffer_ptrs``) as an int64 tensor
  so a Triton kernel can dereference remote XCD buffers directly. On ROCm the
  multicast path is typically disabled (``TORCH_SYMM_MEM_DISABLE_MULTICAST=1``),
  so the peer-pointer path is the portable choice.
* Cross-rank ordering is handled by the caller. We provide a light barrier
  helper; the data movement itself is pure symmetric memory + Triton.

Nothing in here is imported at module load beyond torch; the symmetric-memory
submodule is probed defensively so importing this file never hard-fails on a
platform without symm-mem support.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from vllm.logger import init_logger

logger = init_logger(__name__)

try:
    import torch.distributed._symmetric_memory as torch_symm_mem

    symm_mem_available = True
except ImportError:
    torch_symm_mem = None
    symm_mem_available = False


class SymmMemAllGatherBuffer:
    """A cached symmetric-memory allgather buffer bound to one group.

    A single instance owns:
      * ``buffer`` - a symm-mem tensor of ``numel`` elements (flat), which the
        local rank writes its shard into (or that a producer kernel writes
        directly).
      * ``handle`` - the rendezvous handle. ``handle.buffer_ptrs`` are the base
        device pointers of *every* rank's buffer, in group-rank order.
      * ``peer_ptrs`` - ``buffer_ptrs`` materialized as an int64 CUDA tensor for
        consumption inside Triton kernels.

    The buffer is flat; callers reshape views as needed. Semantics of the
    higher-level allgather (concatenation order == group rank order) are
    enforced by the consuming kernel, not here.
    """

    def __init__(
        self,
        numel: int,
        dtype: torch.dtype,
        device: torch.device,
        group_name: str,
        world_size: int,
        rank_in_group: int,
        device_group=None,
    ) -> None:
        assert symm_mem_available, (
            "torch.distributed._symmetric_memory is not available; "
            "cannot allocate a symmetric-memory allgather buffer."
        )
        self.numel = numel
        self.dtype = dtype
        self.device = device
        self.group_name = group_name
        self.world_size = world_size
        self.rank_in_group = rank_in_group
        # torch ProcessGroup used only for the one-time cold-path init barrier.
        self._group_for_init = device_group

        # Local shard staging buffer (symm-mem): each rank owns `numel` elems.
        self.buffer = torch_symm_mem.empty(numel, dtype=dtype, device=device)
        self.handle = torch_symm_mem.rendezvous(self.buffer, group=group_name)

        # Per-peer base pointers, group-rank ordered, for Triton peer reads.
        self.peer_ptrs = torch.tensor(
            self.handle.buffer_ptrs, dtype=torch.int64, device=device
        )

        # Optional multicast pointer (may be 0/unsupported, esp. on ROCm).
        self.multicast_ptr = getattr(self.handle, "multicast_ptr", 0)

        # --- Signal pad for the on-device (RCCL-free) barrier. ---
        # CUDA-graph-safe, sense-reversing (double-buffered) barrier.
        #
        # Why device-side + self-resetting: under CUDA graph capture, host
        # Python runs ONCE, so a host-incremented generation counter passed as a
        # kernel arg freezes at its capture value and the barrier degenerates to
        # a no-op on every replay. The barrier must therefore carry NO host
        # state: the phase lives in device memory and the kernel advances it, so
        # each graph replay re-executes a correct barrier.
        #
        # Layout: 2 banks of `world_size` int32 slots. Bank b, slot j on rank
        # r's pad = "rank j has arrived at bank b for rank r". Consecutive
        # barriers alternate banks (phase parity), so a fast rank's arrival for
        # barrier N+1 lands in the other bank and cannot clobber a slow rank
        # still finishing barrier N. Double-buffering suffices because a rank
        # cannot lap another by two barriers (finishing N+1 requires the peer to
        # participate in N+1). Each bank self-resets (zeroed) after use.
        self.signal_pad = torch_symm_mem.empty(
            2 * world_size, dtype=torch.int32, device=device
        )
        self.signal_pad.zero_()
        self.signal_handle = torch_symm_mem.rendezvous(
            self.signal_pad, group=group_name
        )
        self.signal_peer_ptrs = torch.tensor(
            self.signal_handle.buffer_ptrs, dtype=torch.int64, device=device
        )
        # Device-side phase counter (per-rank LOCAL, not symm-mem). The kernel
        # reads it, uses (phase & 1) to pick the bank, and stores phase+1 -- so
        # it advances correctly on every graph replay. Absolute value is
        # irrelevant; only parity matters, and parity alternates from any start.
        self.phase = torch.zeros(1, dtype=torch.int32, device=device)
        # One-time cross-rank sync so every rank has finished zeroing its pad
        # before any peer can arrive into it. Cold-path (per-allocation) only.
        torch.cuda.synchronize()
        dist.barrier(group=self._group_for_init)
        torch.cuda.synchronize()

    def local_view(self, shape: torch.Size | tuple[int, ...]) -> torch.Tensor:
        """A view of this rank's own staging buffer with the given shape."""
        return self.buffer[: _prod(shape)].view(*shape)


def _prod(shape) -> int:
    out = 1
    for s in shape:
        out *= int(s)
    return out


class SymmMemAllGatherManager:
    """Lazily allocates and caches symm-mem allgather buffers for a group.

    One manager is created per DCP ``GroupCoordinator`` (see
    :func:`get_symm_mem_allgather_manager`). Buffers are cached by
    ``(numel, dtype)`` so repeated forward passes with the same meta-tensor
    shape reuse the same rendezvous'd allocation.
    """

    def __init__(
        self,
        group_name: str,
        device: torch.device,
        world_size: int,
        rank_in_group: int,
        device_group=None,
    ) -> None:
        self.group_name = group_name
        self.device = device
        self.world_size = world_size
        self.rank_in_group = rank_in_group
        self.device_group = device_group
        self._cache: dict[tuple[int, torch.dtype], SymmMemAllGatherBuffer] = {}

    def get_buffer(
        self, numel: int, dtype: torch.dtype
    ) -> SymmMemAllGatherBuffer:
        key = (numel, dtype)
        buf = self._cache.get(key)
        if buf is None:
            buf = SymmMemAllGatherBuffer(
                numel=numel,
                dtype=dtype,
                device=self.device,
                group_name=self.group_name,
                world_size=self.world_size,
                rank_in_group=self.rank_in_group,
                device_group=self.device_group,
            )
            self._cache[key] = buf
        return buf


# One manager per group name. Keyed on the DCP GroupCoordinator's group name
# so that distinct groups (e.g. hypothetical topologies) get isolated buffers.
_MANAGERS: dict[str, SymmMemAllGatherManager] = {}


def is_symm_mem_available() -> bool:
    return symm_mem_available


def _resolve_group_name(group) -> str:
    """Return the process-group name symm-mem rendezvous expects.

    ``group`` is a vLLM ``GroupCoordinator``. Symmetric memory rendezvous keys
    on the underlying torch ``ProcessGroup``'s ``group_name``.
    """
    device_group = group.device_group
    # torch ProcessGroup exposes `group_name` once registered.
    name = getattr(device_group, "group_name", None)
    if name is None:
        # Fall back to the coordinator's unique name (also registered).
        name = group.unique_name
    return name


def get_symm_mem_allgather_manager(group) -> SymmMemAllGatherManager | None:
    """Get (or lazily create) the symm-mem manager for a GroupCoordinator.

    Returns ``None`` if symmetric memory is unavailable or the group has only
    one rank (nothing to gather).
    """
    if not symm_mem_available:
        return None
    if group.world_size <= 1:
        return None

    group_name = _resolve_group_name(group)
    mgr = _MANAGERS.get(group_name)
    if mgr is None:
        try:
            torch_symm_mem.enable_symm_mem_for_group(group_name)
        except Exception:
            # Deprecated / no-op on newer torch; safe to ignore.
            pass
        mgr = SymmMemAllGatherManager(
            group_name=group_name,
            device=group.device,
            world_size=group.world_size,
            rank_in_group=group.rank_in_group,
            device_group=group.device_group,
        )
        _MANAGERS[group_name] = mgr
    return mgr


def symm_mem_barrier(group) -> None:
    """Host-side barrier over the group's device process group.

    Used to bracket the symm-mem allgather so every peer has published its
    shard before remote reads occur. This is a control-plane sync only; the
    payload movement is pure symm-mem + Triton and never touches RCCL.
    """
    dist.barrier(group=group.device_group)
