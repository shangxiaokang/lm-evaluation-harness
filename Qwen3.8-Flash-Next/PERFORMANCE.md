# Qwen3.8-Flash-Next SGLang performance tests

These scripts compare BF16, NVFP4 online, and serialized NVFP4 offline with
the same SGLang scheduling and cache settings.

- `bf16` loads the BF16 checkpoint without weight quantization.
- `nvfp4_online` loads that BF16 checkpoint, quantizes eligible routed MoE
  weights during startup, and uses online per-token activation scaling.
- `nvfp4_offline` loads the serialized NVIDIA checkpoint, uses its saved
  scales, and must resolve to `modelopt_mixed` by default.

## Files

- `launch_perf_server.sh`: launch one precision mode.
- `bench_perf_serving.sh`: serving throughput and latency matrix.
- `bench_perf_one_batch.sh`: optional fixed-batch benchmark.
- `run_perf_all.sh`: launch and benchmark all modes sequentially.
- `summarize_perf.py`: aggregate repeated serving results against BF16.
- `perf_common.sh`: shared model paths and controlled server settings.

The existing `run_sglang*.sh` files remain the accuracy tests. Do not use
their wall-clock time as the primary performance result.

## Prerequisites

The default paths are:

```text
SGLang source: /lustre/fsw/general_sa/xshang/sglang
Python:        SGLang .venv/bin/python, otherwise python3
BF16:          /lustre/fsw/general_sa/xshang/huggingface/Qwen3.8-Flash-Next
NVFP4 offline: /lustre/fsw/general_sa/xshang/huggingface/Qwen3.8-Flash-Next-NVFP4
```

The SGLang build must support `Qwen4ExpForConditionalGeneration`,
`nvfp4_online`, and the NVIDIA mixed-precision checkpoint loader. PyPI
`sglang==0.5.19` is too old. Do not add `--language-model-only` for this model.

## Controlled comparison

The defaults intentionally fix the following across all modes:

- 4 GB200 GPUs, TP=4 and DP=1.
- BF16 KV cache and BF16 Mamba state.
- `flashinfer_trtllm` MoE runner.
- FlashInfer GDN prefill/decode.
- PLE embedding resident on GPU for all modes.
- Radix cache disabled and cache flushed before every measured run.
- Page size 64, maximum 32 running requests, and identical CUDA Graph limit.

This isolates the precision recipe better than the checkpoint defaults. It is
still not a layer-for-layer comparison: `nvfp4_online` converts routed MoE
experts while the NVIDIA offline checkpoint may use a mixed recipe. Always
save `server_info.json` and report the resolved quantization.

`run_perf_all.sh` validates the resolved quantization and every controlled
server setting before sending benchmark traffic. It also rejects incomplete
requests, missing repeats, and mismatched workload matrices before generating
speedups.

Run a short smoke test first:

```bash
REPEATS=1 \
CONCURRENCIES="1 4" \
WORKLOADS="balanced:512:128" \
bash run_perf_all.sh
```

Run the default full matrix:

```bash
bash run_perf_all.sh
```

Add the optional fixed-batch diagnostic with:

```bash
RUN_ONE_BATCH=1 bash run_perf_all.sh
```

The output is placed under:

```text
results/performance/<timestamp>/
```

Each mode contains its server log, full `/server_info`, time to ready, GPU
telemetry, raw benchmark JSONL, and benchmark log. The top-level `summary.csv`
and `summary.md` contain medians and speedups relative to BF16;
`server_summary.csv` and `server_summary.md` contain startup and memory data.
The fixed-batch JSONL is deliberately not folded into the serving summary.

## Reading the results

- Use concurrency 1 for single-request latency and concurrency 32 for the
  saturated-throughput comparison.
- For `prefill`, focus on input tokens/s and TTFT; for `decode`, focus on
  output tokens/s and TPOT/ITL; use `balanced` as the mixed serving case.
- Compare `weight_gb_per_worker_max`, token capacity, and simultaneous peak HBM
  in `server_summary.csv` alongside throughput. A smaller checkpoint otherwise
  receives a larger KV pool under the same memory fraction.
- Online startup includes BF16-to-NVFP4 weight conversion. Treat startup and
  steady-state serving as separate measurements.

## Run one mode manually

Terminal 1:

```bash
bash launch_perf_server.sh bf16
bash launch_perf_server.sh nvfp4_online
bash launch_perf_server.sh nvfp4_offline
```

Only run one server at a time. In terminal 2:

```bash
MODE=bf16 bash bench_perf_serving.sh bf16
```

Both client scripts query `/server_info` and abort if the running model,
quantization mode, or any controlled setting does not match the mode label.

For a fixed-batch sweep against the running server:

```bash
BATCH_SIZES="1 4 16 32" \
INPUT_LENS="1024" \
OUTPUT_LENS="256" \
bash bench_perf_one_batch.sh bf16
```

## Useful overrides

Override the Python executable if SGLang was installed into another environment:

```bash
PYTHON_BIN=/path/to/python bash run_perf_all.sh
```

Run a checkpoint-adaptive memory/cache comparison separately. The kernel
backends remain fixed so that this is not a fully native/default comparison:

```bash
RUN_ID="native_$(date +%Y%m%d_%H%M%S)" \
KV_CACHE_DTYPE=auto \
PLE_OFFLOAD_EMBEDDING=auto \
DISABLE_RADIX_CACHE=0 \
MAMBA_RADIX_CACHE_STRATEGY=auto \
MAX_MAMBA_CACHE_SIZE=auto \
bash run_perf_all.sh
```

When Radix cache is enabled with `extra_buffer_lazy`, the default scripts
allocate four Mamba state slots per running request. Override
`MAX_MAMBA_CACHE_SIZE` if the resolved server configuration requires a
different value. For checkpoint-native automatic sizing, set both
`MAMBA_RADIX_CACHE_STRATEGY=auto` and `MAX_MAMBA_CACHE_SIZE=auto`.

Use an exclusive GPU allocation. `gpu_metrics.csv` is filtered to
`CUDA_VISIBLE_DEVICES`, but another process on those devices still contaminates
memory, utilization, power, and latency measurements.

The default mode order is fixed, so `time_to_ready_seconds.txt` is not a fair
cold-start comparison: BF16 and online reuse the same files and can benefit
from the host page cache. For startup experiments, repeat with rotated orders
and state whether the filesystem/page cache was warm, for example:

```bash
MODES="nvfp4_offline bf16 nvfp4_online" RUN_ID=order_2 bash run_perf_all.sh
```

To compare total GPU memory rather than KV capacity, first find the BF16 token
capacity in `server_info.json`, then rerun every mode with the same cap:

```bash
MAX_TOTAL_TOKENS=<common-safe-value> bash run_perf_all.sh
```

Other common overrides include:

```text
MODES="bf16 nvfp4_online nvfp4_offline"
WORKLOADS="balanced:1024:256 prefill:4096:32 decode:128:1024"
CONCURRENCIES="1 4 16 32"
REPEATS=3
REQUEST_MULTIPLIER=10
WARMUP_REQUESTS=auto
OUTPUT_DETAILS=0
RUN_ONE_BATCH=0
```

Do not pass `--disable-ignore-eos`: the benchmark intentionally forces the
requested output length. `--random-range-ratio 1` is also intentional; it
makes every request use the exact configured ISL and OSL.
