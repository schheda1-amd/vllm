# Starscream attention performance benchmark (SPX vs CPX+NPS4)

Benchmarks decode-attention wall-clock for **Llama3-70B (TP=8)** on AMD Instinct,
comparing **a single physical GPU** in two compute-partition modes and, within
CPX, three cross-XCD merge implementations. Context for a fresh session picking
this up on the remote node.

## Focus: ONE physical GPU

The primary comparison is a **single physical GPU's attention shard**:

| Mode | Devices used | Attention per device | Cross-XCD merge |
|------|--------------|----------------------|-----------------|
| **SPX** | 1 physical GPU (1 rank) | over the **full** context | none |
| **CPX+NPS4** | that same GPU as 8 XCDs (8 ranks) | over **1/cpx** of the context | yes — 3 variants |

We are NOT sweeping the whole node. SPX runs 1 rank on GPU 0; CPX runs 8 ranks on
GPU 0's 8 XCDs (`CUDA_VISIBLE_DEVICES=0` vs `0..7`). The question: can the same
silicon, subdivided into 8 XCDs, beat its own SPX attention latency once the
cross-XCD merge is done via symmetric memory instead of RCCL?

CPX merge variants (toggled by env flag, benchmarked back-to-back per cell):

| Variant | Merge path |
|---------|-----------|
| `rccl`  | RCCL/NCCL allgather → `reduce_segments` (stock) |
| `step1` | symmetric-memory allgather → `reduce_segments` (collective swap) |
| `step2` | fused symmetric-memory allgather + reduce (single kernel) |

The signal-pad barrier (`VLLM_STARSCREAM_SIGNAL_PAD_BARRIER=1`, see the impl
README) applies to step1/step2 and removes RCCL from the merge sync path; toggle
it via `SIGNAL_PAD=1` on the launcher.

**Thesis.** CPX partitioning cuts per-XCD attention compute ~cpx×, but makes the
intra-GPU cross-XCD communication explicit. RCCL is a poor fit for that
intra-physical-GPU peer traffic and can bottleneck. Symmetric memory
(step1/step2, ideally with the signal-pad barrier) gives CPX a performant merge
so the GPU in CPX mode stays competitive with — or beats — the same GPU in SPX on
decode attention. The benchmark quantifies this across workload shapes.

## Model shape: Llama3-70B under TP=8, one GPU

Llama3-70B: 64 q-heads, 8 kv-heads (GQA), head_dim 128. Under TP=8 each GPU owns:

| Param | Whole model | Per GPU (TP=8) |
|-------|-------------|----------------|
| q-heads | 64 | **8** |
| kv-heads | 8 | **1** |
| head_size | 128 | 128 |

So one GPU runs GQA with **8 q-heads sharing 1 kv-head** — the defaults
(`--total-q-heads 8 --total-kv-heads 1 --head-size 128`). This is exactly the
Starscream regime: `kv-heads (1) < cpx (8)`, so in CPX the single KV head would
otherwise be replicated across all 8 XCDs; Starscream splits its context instead.

Decode GEMM shapes on that GPU (batch B, context S), for reference:
* **SPX:** QKᵀ `[8,128]×[128,S]→[8,S]`; P·V `[8,S]×[S,128]→[8,128]`.
* **CPX (per XCD):** query all-gathered to 8 q-heads, context split 8×:
  QKᵀ `[8,128]×[128,S/8]→[8,S/8]`; P·V `[8,S/8]×[S/8,128]→[8,128]` partial →
  merge across 8 XCDs → each XCD outputs `[B,1,128]`. The S dimension is cut 8×
  per XCD and the 8 XCDs run concurrently.

## What is measured (and what is NOT)

* **Measured:** wall-clock of a single **decode** attention step — start to end
  of the attention op. In CPX this includes the query-head all-gather (RCCL,
  constant across merge variants), the per-XCD partial attention, and the merge.
* **Reduction across ranks:** `MAX` — in CPX the slowest XCD is the critical path
  that gates the physical GPU. Each rank reports the median over `--iters` timed
  runs (after `--warmup`); the harness takes the max across ranks. In SPX there
  is one rank, so max is trivially that rank.
* **Decode only.** Every sequence contributes exactly one query token
  (`max_seqlen_q = 1`). This is deliberate: the Starscream cross-XCD path is a
  **decode** path — prefill (`max_seqlen_q > 1`) uses a different branch and is
  out of scope here. The prefill:decode ratios in the workload table describe
  where each workload spends its time; this benchmark targets the decode phase
  those ratios are dominated by.
* **NOT measured:** prefill attention, MoE/FFN layers, end-to-end model latency,
  throughput/tokens-per-second. This is an attention-kernel microbenchmark of
  the decode step only, on a single physical GPU.

## Parameters and shapes

* **Head counts = model config (Llama3-70B, TP=8), NOT the workload table.** The
  workload table only gives sequence shapes → `--seq-lens` / `--batch-sizes`.
  Head counts come from the architecture → `--total-q-heads 8 --total-kv-heads 1
  --head-size 128` (the single-GPU TP=8 shard; these are the defaults). Because
  the benchmark runs one physical GPU, `num_physical = 1`, so the `--total-*`
  values ARE the per-GPU values (no further division).
* Llama3-70B uses standard GQA, so unlike an MLA model these head counts map
  directly onto the `triton_unified_attention` path Starscream lives in — this is
  a faithful shape, not a synthetic stand-in.
* Per-device head derivation: `qh_dev = total_q_heads / num_physical` (= 8),
  `kvh = total_kv_heads / num_physical` (= 1); in CPX local q-heads per XCD =
  `qh_dev / cpx` (= 1).
* Sweeps: `--seq-lens` (context length) × `--batch-sizes` (decode batch =
  concurrent sequences). The merge cost itself depends on batch size, num_heads,
  and head_size only; seq_len drives the upstream attention compute, which is
  what changes the merge:total ratio between workloads.
* Scenario mapping (informational labels in the output), from the workload
  table:

  | seq_len | scenario | input/output tokens |
  |---------|----------|---------------------|
  | 256     | Short Chat / Q&A | 256 / 256 |
  | 8192    | Summarization / Reasoning | 8192 / 256–2048 |
  | 131072  | Writing / Large Coding | 131072 / 10240–32768 |
  | 2621440 | Advanced Coding | 2621440 / 1024–10240 |

## OOM policy

No extrapolation. For each `(batch, seq_len)` cell, all ranks attempt allocation,
then all-reduce (`MIN`) the success flag. If *any* rank OOMs, all ranks skip that
cell together (this avoids a collective hang) and it is logged `SKIP(OOM)` and
written as a blank latency in the CSV. Enabling larger batches / longer contexts
is a separate task (it would need paging or a different allocation strategy).

## Docker container

Same container as the correctness tests:

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

The container ships a pre-built vllm; the benchmark runs against the mounted
source via `PYTHONPATH=/workspace/vllm` (warnings about `vllm._C` are harmless).

## Running

**Two separate runs, one per hardware partition mode.** SPX and CPX are hardware
modes set with `rocm-smi`, NOT software flags — switch the hardware before each
run. Both runs use a SINGLE physical GPU (SPX: 1 rank on GPU 0; CPX: 8 ranks on
GPU 0's XCDs). Compare the two CSVs offline.

### Convenience launcher

```bash
# SPX — 1 rank on one physical GPU (set hardware first)
rocm-smi --setcomputepartition SPX
./tests/kernels/attention/run_starscream_bench.sh spx      # mode defaults to spx

# CPX+NPS4 — that same GPU's 8 XCDs, with the fast signal-pad barrier
rocm-smi --setcomputepartition CPX
SIGNAL_PAD=1 ./tests/kernels/attention/run_starscream_bench.sh cpx
```

The launcher pins devices automatically (SPX `CUDA_VISIBLE_DEVICES=0`,
nproc=1; CPX `0,1,2,3,4,5,6,7`, nproc=cpx_size) and defaults to the Llama3-70B
per-GPU shape (q=8, kv=1, head_size=128). Override via env vars (see script
header): `DEVICES`, `SEQ_LENS`, `BATCH_SIZES`, `Q_HEADS`, `KV_HEADS`, `CPX_SIZE`,
`SIGNAL_PAD`, `ITERS`, `CSV`. Example — a different physical GPU + quick sweep:

```bash
DEVICES=8,9,10,11,12,13,14,15 SEQ_LENS=8192 BATCH_SIZES=1,32 SIGNAL_PAD=1 \
  ./tests/kernels/attention/run_starscream_bench.sh cpx
```

Default CSV output: `/workspace/bench_spx.csv` and `/workspace/bench_cpx.csv`.

### Manual invocation (equivalent)

```bash
# SPX — one physical GPU, 1 rank
rocm-smi --setcomputepartition SPX
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=/workspace/vllm \
torchrun --nnodes=1 --nproc-per-node=1 \
  tests/kernels/attention/bench_starscream_attention.py \
  --mode spx --total-q-heads 8 --total-kv-heads 1 --head-size 128 \
  --seq-lens 256,8192,131072 --batch-sizes 1,8,32,64,128,256,512,1024 \
  --csv /workspace/bench_spx.csv

# CPX — that GPU's 8 XCDs, signal-pad barrier on
rocm-smi --setcomputepartition CPX
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
PYTHONPATH=/workspace/vllm \
TORCH_SYMM_MEM_DISABLE_MULTICAST=1 \
VLLM_STARSCREAM_SIGNAL_PAD_BARRIER=1 \
torchrun --nnodes=1 --nproc-per-node=8 \
  tests/kernels/attention/bench_starscream_attention.py \
  --mode cpx --cpx-size 8 --total-q-heads 8 --total-kv-heads 1 --head-size 128 \
  --seq-lens 256,8192,131072 --batch-sizes 1,8,32,64,128,256,512,1024 \
  --csv /workspace/bench_cpx.csv
```

**Built-in sanity check.** Before the sweep, the harness automatically runs one
small cell end-to-end (default `seq_len=8192, batch=1`, all variants) so a broken
setup — bad shapes, symmetric memory unavailable, a collective hang — fails fast
before large KV caches are allocated. It prints `[sanity] ...` lines and aborts
the run if that cell OOMs or errors. Controls:

* `--skip-sanity` — skip it entirely
* `--sanity-seq-len N` / `--sanity-batch N` — change the probe cell

No manual warm-up run is needed; just launch the sweep.

## Output format

Console: one row per `(seq_len, batch)`, one column per variant (`spx_us` in SPX
mode; `rccl_us`, `step1_us`, `step2_us` in CPX mode), latency in microseconds,
plus the scenario label.

CSV (tidy long format, ready for plotting — SPX and CPX files concatenate
directly):

```
mode,seq_len,batch,variant,latency_us
spx,8192,1,spx,12.34
cpx,8192,1,rccl,45.67
cpx,8192,1,step1,18.90
cpx,8192,1,step2,15.23
cpx,131072,1024,rccl,          <- blank latency = SKIP(OOM)
```

To compare SPX vs CPX for a cell, join on `(seq_len, batch)`: SPX has one
`variant=spx` row; CPX has three (`rccl`/`step1`/`step2`).

## Related

* Implementation + correctness: `vllm/attention/ops/STARSCREAM_SYMM_MEM_README.md`
* Correctness test: `tests/kernels/attention/test_starscream_symm_reduce.py`
* Next step after perf: on-device signal-pad barrier to remove the host
  `dist.barrier` from the merge critical path (see the implementation README).
