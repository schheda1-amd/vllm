# Starscream symmetric-memory allgather + fused reduce

Context for continuing this work (including on a fresh Claude session on the
remote cluster). This documents the symmetric-memory replacement for the RCCL
cross-XCD allgather in the Starscream (CPX+NPS4) decode attention path, and the
fusion of that allgather with `reduce_segments`.

## Docker container

Launch the container from the directory containing the vllm repo (it is mounted
at `/workspace/vllm`):

```bash
docker run \
    -it --rm \
    --device /dev/dri --device /dev/kfd \
    --network host --ipc host \
    --group-add video --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged \
    -v .:/workspace/vllm \
    --shm-size 128G \
    rocm/vllm:rocm7.13.0_gfx94X-dcgpu_ubuntu24.04_py3.13_pytorch_2.10.0_vllm_0.19.1 /bin/bash
```

The container ships a pre-built vllm. Use `PYTHONPATH` to override it with the
mounted source tree (no rebuild needed — the compiled `vllm._C` extension in the
system install is still used, warnings about it are harmless):

```bash
PYTHONPATH=/workspace/vllm \
TORCH_SYMM_MEM_DISABLE_MULTICAST=1 \
torchrun --nnodes=1 --nproc-per-node=8 \
  /workspace/vllm/tests/kernels/attention/test_starscream_symm_reduce.py \
  --mode all --cpx-size 8 --num-tokens 32 --num-heads 64 --head-size 128 --seq-len 4096
```

## Background

Starscream shards each sequence's KV context across the XCDs (vGPUs) of one
physical AMD Instinct GPU running in CPX+NPS4 mode, instead of replicating the
KV cache when `num_kv_heads < TP degree`. Each XCD computes attention over its
local KV slice and produces **partial, un-normalized** results. Those partials
must be merged across the XCDs of the group using online softmax.

The XCD group is the **DCP group** (`get_dcp_group()`); by construction it is a
contiguous block of ranks within one TP group, i.e. the XCDs of a single
physical GPU — provided `decode_context_parallel_size == tp_size //
num_kv_heads`. The allgather is therefore **intra-physical-GPU, inter-XCD**.

In the stock branch (`vllm/attention/ops/triton_unified_attention.py`, tail of
`unified_attention`, ~line 1101), the merge is two steps:

```python
starscream_metadata = get_dcp_group().all_gather(starscream_meta_out, dim=-2)  # RCCL
reduce_segments[...](..., starscream_flag=True, NUM_SEGMENTS_PER_SEQ=cpx_size)  # merge+scatter
```

Goal: replace the RCCL allgather with a symmetric-memory Triton allgather, and
then fuse the allgather and `reduce_segments` into a single kernel. Symmetric
memory is chosen because it lets us write fused communication+compute kernels in
Triton (peer buffers are directly addressable via device pointers).

**The original `reduce_segments` kernel is never modified.** All new logic lives
in separate modules.

## Meta-tensor layout (contract between producer and consumer)

Each rank produces `starscream_meta_out` of shape `[T, H, 1, Dext]` (fp32),
`Dext = next_pow2(head_size) + 2`. Per `(token, head)`, the `Dext` contiguous
slots are:

| slots         | contents                                             |
|---------------|------------------------------------------------------|
| `[0 : HP]`    | numerator `Σ_j exp(s_j − m_local)·V_j` (un-normalized)|
| `[HP]`        | `L` = exp-sum `Σ_j exp(s_j − m_local)`               |
| `[HP + 1]`    | `M` = `m_local` (local max logit)                    |

`HP = next_pow2(head_size)`. The merge rescales every XCD's partials to a common
max before dividing — the online-softmax invariant.

## Files

| File | Role |
|------|------|
| `vllm/distributed/device_communicators/symm_mem_allgather.py` | Infra: lazy per-`(numel,dtype)` symm-mem buffer + rendezvous, per-peer pointer int64 tensor, host barrier. One manager per DCP group. |
| `vllm/attention/ops/starscream_symm_reduce.py` | `symm_mem_all_gather()` (Step 1) and `fused_all_gather_reduce_segments()` (Step 2), plus their Triton kernels. |
| `vllm/attention/ops/triton_unified_attention.py` | Gated dispatch at the tail (~line 1101). `reduce_segments` untouched. |
| `vllm/envs.py` | Two flags (both default `0`). |

## Three runtime paths (flag-selectable)

| Path | Env flag(s) | What runs |
|------|-------------|-----------|
| **C — baseline (default)** | neither | Stock RCCL allgather → unmodified `reduce_segments`. Golden reference. |
| **Step 1 — symm-mem allgather** | `VLLM_STARSCREAM_USE_SYMM_MEM=1` | Symm-mem Triton allgather → **unmodified** `reduce_segments`. Isolates the collective swap. |
| **Step 2 — fused** | `VLLM_STARSCREAM_FUSE_REDUCE=1` | Single fused kernel: peer-read allgather + online-softmax merge + head-scatter. Returns before `reduce_segments`. |

Both new paths fall back to Path C automatically if symmetric memory is
unavailable (`get_symm_mem_allgather_manager()` returns `None`), so nothing
hard-breaks. `VLLM_STARSCREAM_FUSE_REDUCE` takes precedence over
`VLLM_STARSCREAM_USE_SYMM_MEM` when both are set.

### What "Step 2" fuses

`reduce_segments` in its `starscream_flag=True` branch does exactly:
1. **Merge** — online softmax: `overall_max = max(segm_max)`, rescale each
   segment by `exp(m − max)`, sum numerators, divide by the global exp-sum
   (`triton_unified_attention.py:787-816`).
2. **Scatter** — write only this rank's owned heads into `out` at the local
   index (`:818-825`).

The fused kernel (`_fused_allgather_reduce_kernel`) folds the allgather into
that: instead of gathering all peers' partials into a `[T,H,CPX,Dext]` tensor
and having `reduce_segments` load from it, the fused kernel **reads each peer's
partials directly over symm-mem peer pointers inside the same kernel** that does
the merge and scatter. One launch, no intermediate gathered buffer, no separate
collective.

## Tests

Test file: `tests/kernels/attention/test_starscream_symm_reduce.py`

The test is self-contained and driven by `torchrun` directly — no pytest, no
Ray. It builds the `starscream_meta_out` tensor exactly as
`triton_unified_attention.py` does (same shape, same slot layout), then invokes
the new paths and compares against the RCCL baseline (`get_dcp_group().all_gather`
→ `reduce_segments`).

### Contained test cases

| Test | What it checks |
|------|---------------|
| `test_step1_output_matches_baseline` | (a) symm-mem gathered tensor == RCCL gathered tensor; (b) reduce_segments output fed from symm-mem gather == baseline output |
| `test_step2_output_matches_baseline` | fused kernel output == baseline output |
| `test_randomized_inputs` | both Step 1 and Step 2 vs. baseline across 8 independent random seeds |

### Run commands (inside the ROCm/PyTorch docker container)

```bash
# --- Step 1 only ---
VLLM_STARSCREAM_USE_SYMM_MEM=1 \
TORCH_SYMM_MEM_DISABLE_MULTICAST=1 \
torchrun --nnodes=1 --nproc-per-node=8 \
  tests/kernels/attention/test_starscream_symm_reduce.py \
  --mode step1 --cpx-size 8 --num-tokens 32 --num-heads 64 --head-size 128 --seq-len 4096

# --- Step 2 only ---
VLLM_STARSCREAM_FUSE_REDUCE=1 \
TORCH_SYMM_MEM_DISABLE_MULTICAST=1 \
torchrun --nnodes=1 --nproc-per-node=8 \
  tests/kernels/attention/test_starscream_symm_reduce.py \
  --mode step2 --cpx-size 8 --num-tokens 32 --num-heads 64 --head-size 128 --seq-len 4096

# --- Both (recommended first run) ---
TORCH_SYMM_MEM_DISABLE_MULTICAST=1 \
torchrun --nnodes=1 --nproc-per-node=8 \
  tests/kernels/attention/test_starscream_symm_reduce.py \
  --mode all --cpx-size 8 --num-tokens 32 --num-heads 64 --head-size 128 --seq-len 4096

# --- Tighter tolerance if you want to catch ULP noise ---
TORCH_SYMM_MEM_DISABLE_MULTICAST=1 \
torchrun --nnodes=1 --nproc-per-node=8 \
  tests/kernels/attention/test_starscream_symm_reduce.py \
  --mode all --cpx-size 8 --rtol 1e-5 --atol 1e-5
```

All parameters:

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | `all` | `step1`, `step2`, or `all` |
| `--num-tokens` | 16 | T dimension |
| `--num-heads` | 32 | H (must be divisible by `--cpx-size`) |
| `--head-size` | 128 | head dimension |
| `--cpx-size` | world_size | XCDs in the group; must equal `--nproc-per-node` |
| `--seq-len` | 4096 | full sequence length (informational for this test) |
| `--rtol` / `--atol` | 1e-3 | match tolerances |
| `--trials` | 8 | random trial count for randomized test |

### Test progression (intended)

1. **Functional correctness** — Step 1 first (smallest change: only the
   collective is swapped), then Step 2.
2. **Output matching for randomized inputs** — `--mode all` covers this.
3. **Performance runs** — Step 2 is the target; Step 1 is a useful intermediate
   data point.

## Semantic-parity notes (for the output-matching stage)

- Fused merge math is a line-by-line mirror of `reduce_segments`'s
  `starscream_flag=True` branch: `overall_max = max`, rescale by
  `exp(m − max)`, sum numerators + exp-sums, `where(expsum == 0, 0,
  acc / expsum)`, then the identical head-scatter (`chunk = H // cpx`, local
  index).
- Segment order == group-rank order in both the symm-mem allgather and the RCCL
  allgather, so gathered layouts match.
- fp8 path (`USE_FP8`) carries the same `out_scale_inv` + clamp.
- Numerical note: the batch formulation (`reduce_segments`) and the sequential
  accumulation (fused kernel) reduce in a potentially different order. Results
  should match to fp32 rounding; if the output-matching stage sees tiny ULP
  diffs, that is the expected cause, not a semantic break.

## Cluster prerequisites (verify on remote — not verified locally)

1. **PyTorch ≥ 2.9** with symmetric-memory support on ROCm
   (`torch.distributed._symmetric_memory`). The reference kernels in
   `~/symmetric-memory/gemm+ag_symm.py` establish the working idiom.
2. **`TORCH_SYMM_MEM_DISABLE_MULTICAST=1`** in the container. The kernels here
   use per-peer `buffer_ptrs` (not the multicast pointer), so they are
   compatible with multicast disabled — which is the ROCm default.
3. Peer-pointer deref idiom used: `tl.load(peer_ptrs + p).to(tl.pointer_type(
   tl.float32))`, exactly as in the reference GEMM+AG kernels.

## Sync model — two barrier implementations (flag-selectable)

The symm-mem allgather (step1) and fused kernel (step2) are each bracketed by two
barriers: one *before* the peer reads (every peer has published its shard) and
one *after* (no peer overwrites its staging buffer before others finish reading).
A single dispatch helper `_barrier(group, buf)` in `starscream_symm_reduce.py`
selects the implementation:

**Default — host `dist.barrier`** (`VLLM_STARSCREAM_SIGNAL_PAD_BARRIER=0`).
`dist.barrier(group.device_group)`. Correctness-safe and simple, BUT
`dist.barrier` is *itself an RCCL collective*. So this path replaces one RCCL
all_gather (which fuses data movement + sync) with a symm-mem copy + **two** RCCL
barriers. That fixed 2-collective latency dominates on small messages and is why,
in the first perf run, symm-mem lost to baseline on small batches and only won on
large ones (batch>=128, seq=8192) — the exact inversion of the expected profile.

**Fast — on-device signal-pad barrier** (`VLLM_STARSCREAM_SIGNAL_PAD_BARRIER=1`).
Replaces `dist.barrier` with a single-warp Triton kernel
(`_signal_pad_barrier_kernel`) that arrives/waits through symmetric-memory peer
pointers — no RCCL, no host round-trip. This is what should restore the expected
"symm-mem wins the small/latency-bound cases" profile.

Algorithm (all-to-all, monotonic generation):
- Each rank owns an int32 signal pad of `world_size` slots (a second symm-mem
  allocation in `SymmMemAllGatherBuffer`, with `signal_peer_ptrs`).
- A per-buffer generation counter `_gen`, incremented each barrier (`next_gen()`).
  Because barriers are collective, every rank passes the same `gen`.
- **arrive:** vectorized `atomic_xchg(gen, release, sys)` into slot `my_rank` of
  every peer's pad.
- **wait:** spin on `atomic_add(...,0, acquire, sys)` over my own pad until the
  min across slots `>= gen`.
- `>=` compare + never resetting to 0 avoids the classic barrier reset race; a
  fast rank entering `gen+1` can only raise a slot, never starve a slow rank.
- One-time cold-path `dist.barrier` after zeroing the pad at allocation, so no
  peer arrives before all pads are zeroed.

### HARDWARE VALIDATION REQUIRED (MI300/CDNA, ROCm)

The signal-pad barrier's correctness hinges on Triton lowering `sem="release"/
"acquire"` with `scope="sys"` to **true system-scope** acquire/release across
XCDs. If the ROCm Triton build silently narrows `scope="sys"` to device scope,
cross-XCD data visibility is not guaranteed and you can get stale peer reads under
load. VALIDATE by running the correctness test with the flag ON before trusting
any perf number:

```bash
PYTHONPATH=/workspace/vllm \
TORCH_SYMM_MEM_DISABLE_MULTICAST=1 \
VLLM_STARSCREAM_SIGNAL_PAD_BARRIER=1 \
torchrun --nnodes=1 --nproc-per-node=8 \
  /workspace/vllm/tests/kernels/attention/test_starscream_symm_reduce.py \
  --mode all --cpx-size 8 --num-tokens 32 --num-heads 64 --head-size 128 --seq-len 4096
```

Step 1 must still show `0.00e+00` and step 2 ULP-only diffs. A hang = the wait
never observing arrive (scope/ordering issue); wrong output = visibility issue.

### Possible further speedups (measure first)
- Fold arrive into the producer copy and wait into the consumer kernel, removing
  the separate barrier launch entirely (true in-kernel handshake). Bigger change.
- Drop the trailing (post-read) barrier: the next call's publish + its
  pre-barrier may already order buffer reuse. A/B once the basic version passes.
