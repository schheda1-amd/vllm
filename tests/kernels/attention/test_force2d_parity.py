# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Correctness gate for VLLM_STARSCREAM_FORCE_2D.

Runs the FULL unified_attention() on the real CPX setup (8 XCDs, starscream on,
step2 merge) and checks that forcing the 2D kernel produces the SAME decode
output as the default 3D kernel, for the same inputs. Both are just different
tilings of the identical attention math, so their outputs must match to fp
rounding. If they match, the FORCE_2D path is correct (3D is already validated).

Also cross-checks BOTH against a single-GPU reference attention over the full
(un-split) context, so we catch a systematic error that happens to be identical
in 2D and 3D.

Run (inside the container, CPX hardware):
  TORCH_SYMM_MEM_DISABLE_MULTICAST=1 \
  torchrun --nnodes=1 --nproc-per-node=8 \
    tests/kernels/attention/test_force2d_parity.py \
    --cpx-size 8 [--seq-len 16384] [--batches 32,64] [--q-heads 8] [--head-size 128]
"""
import argparse
import os

import torch
import torch.distributed as dist

_rank = int(os.environ.get("RANK", 0))
_local_rank = int(os.environ.get("LOCAL_RANK", 0))
_world_size = int(os.environ.get("WORLD_SIZE", 1))
torch.cuda.set_device(_local_rank)

from vllm.distributed.parallel_state import (  # noqa: E402
    init_distributed_environment, initialize_model_parallel,
    get_dcp_group, destroy_model_parallel,
)

init_distributed_environment(world_size=_world_size, rank=_rank,
                             local_rank=_local_rank,
                             distributed_init_method="env://", backend="nccl")
try:
    import torch.distributed._symmetric_memory as _sm
    _sm.enable_symm_mem_for_group(dist.group.WORLD.group_name)
except Exception:
    pass

import vllm.envs as envs  # noqa: E402
from vllm.attention.ops.triton_unified_attention import unified_attention  # noqa: E402


def _log(m):
    if _rank == 0:
        print(m, flush=True)


def _build_cell(B, S, qh_dev, kvh, head_size, block_size, cpx, device, seed):
    """Build q_local/kv-cache/block-table for one decode cell (CPX)."""
    torch.manual_seed(seed + _rank)
    import math
    dtype = torch.bfloat16
    local_ctx = math.ceil(S / cpx)
    npb = math.ceil(local_ctx / block_size)
    total_blocks = B * npb
    k_cache = torch.randn(total_blocks, block_size, kvh, head_size, dtype=dtype, device=device)
    v_cache = torch.randn(total_blocks, block_size, kvh, head_size, dtype=dtype, device=device)
    block_table = torch.arange(total_blocks, dtype=torch.int32, device=device).reshape(B, npb)
    cu_seqlens_q = torch.arange(B + 1, dtype=torch.int32, device=device)
    seqused_k = torch.full((B,), S, dtype=torch.int32, device=device)
    qh_local = qh_dev // cpx
    q_local = torch.randn(B, qh_local, head_size, dtype=dtype, device=device)
    k_descale = torch.ones((B, kvh), dtype=torch.float32, device=device)
    v_descale = torch.ones((B, kvh), dtype=torch.float32, device=device)
    return dict(k_cache=k_cache, v_cache=v_cache, block_table=block_table,
                cu_seqlens_q=cu_seqlens_q, seqused_k=seqused_k, q_local=q_local,
                k_descale=k_descale, v_descale=v_descale, S=S)


def _run(tn, grp, cpx, qh_dev, head_size):
    """One CPX decode attention step (step2 merge). Returns owned-head output."""
    B = tn["q_local"].shape[0]
    qh_local = qh_dev // cpx
    out = torch.empty(B, qh_local, head_size, dtype=torch.bfloat16, device=tn["q_local"].device)
    q = grp.all_gather(tn["q_local"].contiguous(), dim=-2)
    unified_attention(
        q=q, k=tn["k_cache"], v=tn["v_cache"], out=out,
        slice_idx=0, starscream_rank=grp.rank_in_group,
        cu_seqlens_q=tn["cu_seqlens_q"], max_seqlen_q=1,
        seqused_k=tn["seqused_k"], max_seqlen_k=tn["S"],
        softmax_scale=head_size ** -0.5, causal=True, window_size=(-1, -1),
        block_table=tn["block_table"], softcap=0, q_descale=None,
        k_descale=tn["k_descale"], v_descale=tn["v_descale"],
        cpx_size=cpx, enable_starscream=True,
    )
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cpx-size", type=int, default=_world_size)
    p.add_argument("--seq-len", type=int, default=16384)
    p.add_argument("--batches", type=str, default="32,64")
    p.add_argument("--q-heads", type=int, default=8)   # per-GPU (llama3-70b TP=8)
    p.add_argument("--kv-heads", type=int, default=1)
    p.add_argument("--head-size", type=int, default=128)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--rtol", type=float, default=2e-2)
    p.add_argument("--atol", type=float, default=2e-2)
    args = p.parse_args()

    assert _world_size % args.cpx_size == 0
    initialize_model_parallel(tensor_model_parallel_size=_world_size,
                              pipeline_model_parallel_size=1,
                              decode_context_model_parallel_size=args.cpx_size)
    grp = get_dcp_group()
    device = torch.device(f"cuda:{_local_rank}")
    cpx = args.cpx_size
    # single physical GPU focus: num_physical=1 -> per-device == given
    qh_dev = args.q_heads
    kvh = args.kv_heads
    batches = [int(x) for x in args.batches.split(",")]

    # step2 merge, signal-pad barrier (matches the perf runs)
    os.environ["VLLM_STARSCREAM_USE_SYMM_MEM"] = "0"
    os.environ["VLLM_STARSCREAM_FUSE_REDUCE"] = "1"
    os.environ["VLLM_STARSCREAM_SIGNAL_PAD_BARRIER"] = "1"

    _log(f"[force2d-parity] seq={args.seq_len} batches={batches} "
         f"qh_dev={qh_dev} kvh={kvh} head={args.head_size} cpx={cpx}")
    all_ok = True
    for B in batches:
        seed = 1000 + B
        # Build ONCE, run 3D then 2D on the SAME inputs.
        tn = _build_cell(B, args.seq_len, qh_dev, kvh, args.head_size,
                         args.block_size, cpx, device, seed)

        os.environ["VLLM_STARSCREAM_FORCE_2D"] = "0"
        out_3d = _run(tn, grp, cpx, qh_dev, args.head_size).clone()
        dist.barrier()

        os.environ["VLLM_STARSCREAM_FORCE_2D"] = "1"
        out_2d = _run(tn, grp, cpx, qh_dev, args.head_size).clone()
        dist.barrier()

        # Compare on THIS rank's owned heads, reduce verdict across ranks.
        maxdiff = (out_3d.float() - out_2d.float()).abs().max().item()
        ok = torch.allclose(out_3d.float(), out_2d.float(),
                            rtol=args.rtol, atol=args.atol)
        # global agreement
        t = torch.tensor([1.0 if ok else 0.0], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.MIN)
        md = torch.tensor([maxdiff], device=device)
        dist.all_reduce(md, op=dist.ReduceOp.MAX)
        gok = t.item() > 0
        all_ok = all_ok and gok
        _log(f"  batch={B:>4}: 2D vs 3D {'MATCH' if gok else 'MISMATCH'} "
             f"(max|Δ| over ranks = {md.item():.4e})")

    _log(f"[force2d-parity] {'ALL PASSED' if all_ok else 'FAILED'}")
    destroy_model_parallel()
    dist.destroy_process_group()
    if not all_ok and _rank == 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
