# Starscream attention performance benchmark (SPX vs CPX+NPS4)

Benchmarks decode-attention wall-clock for DeepSeek-R1-style shapes on AMD
Instinct GPUs, comparing two hardware compute-partition modes and, within CPX,
three cross-XCD merge implementations. Context for a fresh session picking this
up on the remote node.

## What is being compared

| Mode | Devices | Attention per device | Cross-XCD merge |
|------|---------|----------------------|-----------------|
| **SPX** | 8 physical GPUs | over the **full** context | none |
| **CPX+NPS4** | 64 vGPUs (8 XCDs/GPU) | over **1/cpx** of the context | yes — 3 variants |

CPX merge variants (toggled by env flag, benchmarked back-to-back per cell):

| Variant | Merge path |
|---------|-----------|
| `rccl`  | RCCL/NCCL allgather → `reduce_segments` (stock) |
| `step1` | symmetric-memory allgather → `reduce_segments` (collective swap) |
| `step2` | fused symmetric-memory allgather + reduce (single kernel) |

**Thesis.** CPX partitioning cuts per-XCD attention compute ~cpx×, but makes the
intra-GPU cross-XCD communication explicit. RCCL is a poor fit for that
intra-physical-GPU peer traffic and can bottleneck. Symmetric memory
(step1/step2) gives CPX a performant merge so CPX stays competitive with — or
beats — SPX on end-to-end attention. The benchmark quantifies this across
workload shapes.

## What is measured (and what is NOT)

* **Measured:** wall-clock of a single **decode** attention step — start to end
  of the attention op. In CPX this includes the query-head all-gather (RCCL,
  constant across merge variants), the per-XCD partial attention, and the merge.
* **Reduction across ranks:** `MAX` — the slowest XCD is the critical path that
  gates its physical GPU. Each rank reports the median over `--iters` timed
  runs (after `--warmup`); the harness takes the max across all ranks.
* **Decode only.** Every sequence contributes exactly one query token
  (`max_seqlen_q = 1`). This is deliberate: the Starscream cross-XCD path is a
  **decode** path — prefill (`max_seqlen_q > 1`) uses a different branch and is
  out of scope here. The prefill:decode ratios in the workload table describe
  where each workload spends its time; this benchmark targets the decode phase
  those ratios are dominated by.
* **NOT measured:** prefill attention, MoE/FFN layers, end-to-end model latency,
  throughput/tokens-per-second. This is an attention-kernel microbenchmark of
  the decode step only.

## Parameters and shapes

* **Head counts come from the MODEL CONFIG, not the workload table.** The
  workload table only gives sequence shapes (input/output tokens, prefill:decode
  ratio) → these map to `--seq-lens` / `--batch-sizes`. Head counts, head size,
  and KV-head count are DeepSeek-R1 *architecture* parameters →
  `--total-q-heads` / `--total-kv-heads` / `--head-size`. The `128/128/128`
  defaults are placeholders.
* **KV-heads caveat (important).** DeepSeek-R1 uses MLA (Multi-head Latent
  Attention): KV is a compressed latent, not N discrete KV heads, so there is no
  clean `num_key_value_heads`. Moreover vLLM routes MLA models through a separate
  MLA attention backend, not the `triton_unified_attention` path where Starscream
  lives. Consequences:
    * This benchmark is a **synthetic characterization** of the Starscream merge
      at representative shapes — not a measurement of DeepSeek-R1's real
      attention backend.
    * **Relative results are robust** (SPX vs CPX vs rccl/step1/step2 trends hold
      regardless of the exact head count — the thesis).
    * **Absolute microseconds are not** — they scale with the head config. For
      absolute fidelity, get `num_attention_heads`, `num_key_value_heads` (or the
      MLA qk/v head dims), and `head_dim` from the target model config and pass
      them in.
* Per-device head counts are derived: in CPX the gathered per-XCD query heads =
  `total_q_heads / num_physical_gpus`, and local heads = gathered / cpx.
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
modes set with `rocm-smi`, NOT software flags — you must switch the hardware
before each run. Compare the two CSVs offline.

### Convenience launcher

```bash
# SPX (8 GPUs) — set hardware first
rocm-smi --setcomputepartition SPX
./tests/kernels/attention/run_starscream_bench.sh spx      # mode defaults to spx

# CPX+NPS4 (64 vGPUs) — set hardware first
rocm-smi --setcomputepartition CPX
./tests/kernels/attention/run_starscream_bench.sh cpx
```

Override sweeps / shapes via env vars (see the script header for the full list):

```bash
SEQ_LENS=8192 BATCH_SIZES=1,32 ITERS=20 \
  ./tests/kernels/attention/run_starscream_bench.sh cpx
```

Default CSV output: `/workspace/bench_spx.csv` and `/workspace/bench_cpx.csv`.

### Manual invocation (equivalent)

```bash
# SPX
rocm-smi --setcomputepartition SPX
PYTHONPATH=/workspace/vllm \
torchrun --nnodes=1 --nproc-per-node=8 \
  tests/kernels/attention/bench_starscream_attention.py \
  --mode spx --total-q-heads 128 --total-kv-heads 128 --head-size 128 \
  --seq-lens 256,8192,131072 --batch-sizes 1,8,32,64,128,256,512,1024 \
  --csv /workspace/bench_spx.csv

# CPX
rocm-smi --setcomputepartition CPX
PYTHONPATH=/workspace/vllm \
TORCH_SYMM_MEM_DISABLE_MULTICAST=1 \
torchrun --nnodes=1 --nproc-per-node=64 \
  tests/kernels/attention/bench_starscream_attention.py \
  --mode cpx --cpx-size 8 --total-q-heads 128 --total-kv-heads 128 --head-size 128 \
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
