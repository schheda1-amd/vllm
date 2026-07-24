# SPX vs CPX Attention Benchmark

Benchmarks the Triton decode-attention kernel on one MI300X under two GPU
compute-partition modes. Config is fixed: 128 q-heads, 8 kv-heads (GQA 16),
head_size 128, block_size 16, bf16.

## 1. Build the container

From the repo root (compiles the kernels; `--network=host` is required):

```bash
DOCKER_BUILDKIT=1 docker build --network=host \
  -f docker/Dockerfile.rocm \
  --build-arg ARG_PYTORCH_ROCM_ARCH=gfx942 \
  --target final -t vllm-rocm:starscream .
```

## 2. Start the container

```bash
docker run --rm -it --network=host \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  --security-opt seccomp=unconfined --shm-size=16g \
  -v $(pwd):/workspace/vllm -w /workspace/vllm \
  vllm-rocm:starscream bash
```

## 3. Run

`./run_spx_vs_cpx.sh <mode> [metric]`
  - mode:   `spx` | `cpx-baseline` | `cpx`
  - metric: `latency` (default) | `bandwidth` | `both`

Set the matching hardware partition first, and run one mode at a time.

```bash
# SPX baseline
rocm-smi --setcomputepartition SPX
./run_spx_vs_cpx.sh spx both

# CPX baseline (CPX hardware, Starscream OFF)
rocm-smi --setcomputepartition CPX
./run_spx_vs_cpx.sh cpx-baseline both

# CPX + Starscream (the proposed path)
rocm-smi --setcomputepartition CPX
./run_spx_vs_cpx.sh cpx both
```

Outputs (repo root):
- latency   -> `bench_<mode>.csv`      (per-decode-step us, CUDA-graph timed, MAX over XCDs)
- bandwidth -> `bandwidth_<mode>.csv`  (achieved HBM GB/s via rocprofv3 counters)

## Notes

- Default shapes match `test_harness_1.py`: seq_len {8192, 131072} x batch
  {1, 32, 64, 128, 1024}. Override with `SEQ_LENS=...` / `BATCH_SIZES=...`.
- `cpx` runs the `step2` Starscream path (fused symm-mem allgather + reduce).
- **If a CPX run hangs or CUDA-graph capture fails**, retry with the host
  barrier: `SIGNAL_PAD=0 ./run_spx_vs_cpx.sh cpx both`. Default `SIGNAL_PAD=1`
  uses the on-device signal-pad barrier (keeps RCCL off the merge path, the
  intended symm-mem path); `0` falls back to a host `dist.barrier` that is
  always safe but slightly slower.
