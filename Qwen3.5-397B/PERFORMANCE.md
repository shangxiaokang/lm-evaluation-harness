# Qwen3.5-397B-A17B SGLang tests

This directory contains lm-eval accuracy scripts and reproducible SGLang
serving benchmarks for local Qwen3.5-397B checkpoints.

## Checkpoints and precision modes

| Mode | Checkpoint | SGLang quantization | Activation scale |
|---|---|---|---|
| BF16 | `/lustre/fsw/general_sa/xshang/huggingface/Qwen3.5-397B-A17B` | none | BF16 |
| NVFP4 online | BF16 checkpoint above | `nvfp4_online` | online per-token FP32; routed experts convert while loading |
| NVFP4 offline | `/lustre/fsw/general_sa/xshang/huggingface/Qwen3.5-397B-A17B-NVFP4` | `modelopt_fp4` | serialized checkpoint scales |

The specified NVIDIA NVFP4 checkpoint is the ModelOpt FP4 format, not a
`modelopt_mixed` checkpoint. Do not change the offline mode to
`modelopt_mixed`.

## Accuracy evaluation

Run from a GB200/B200 node after installing SGLang from source and lm-eval:

```bash
bash run_sglang_hf.sh
bash run_sglang.sh
bash run_sglang_nvfp4_offline.sh
```

All scripts default to `TASK=arc_easy` and `BATCH_SIZE=4`. Override, for
example:

```bash
TASK=mmlu_pro BATCH_SIZE=2 LIMIT=16 bash run_sglang_nvfp4_offline.sh
```

`run_sglang_hf.sh` defaults to 8 GPUs because Qwen3.5-397B BF16 needs about
800GB of weight memory. The two NVFP4 scripts default to 4 GPUs. Qwen3.5 is
not currently supported by SGLang's `--language-model-only` allow-list, so
these scripts intentionally do not pass that option.

## Serving benchmark

`run_perf_all.sh` uses four physical GPUs with:

```text
outer TP=4, DP=4, EP=4, MoE DP=1
DP-Attention=on, DP LM head=on
effective Attention TP=1, effective MoE TP=1
```

It uses the SGLang-tested ModelOpt FP4 topology:
`flashinfer_cutedsl` MoE runner plus FlashInfer A2A. Radix cache is disabled
for fixed-length synthetic traffic; BF16 Mamba state is held in BF16 with the
`extra_buffer` strategy.

The default 4-GPU run benchmarks only online and offline NVFP4:

```bash
bash run_perf_all.sh
```

BF16 is intentionally not in this default because 4x B200/GB200 lacks enough
HBM. A fair three-mode comparison needs enough memory for BF16, normally eight
B200/GB200 GPUs; keep EP and DP at four and increase the outer TP width:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MODES="bf16 nvfp4_online nvfp4_offline" \
TP_SIZE=8 DP_SIZE=4 EP_SIZE=4 \
bash run_perf_all.sh
```

This 8-GPU command has effective Attention TP=2 and MoE TP=2. On a 4x B300
system with enough HBM, use `ALLOW_UNSAFE_BF16=1` to permit BF16 in the
four-GPU topology.

### Workloads

| Workload | Input tokens | Output tokens | Total | Main measurements |
|---|---:|---:|---:|---|
| balanced | 12,288 | 12,288 | 24,576 | total throughput, E2E, TTFT, TPOT |
| prefill | 16,384 | 1,024 | 17,408 | input throughput, TTFT |
| decode | 1,024 | 16,384 | 17,408 | output throughput, TPOT, ITL |

The server context limit is deliberately fixed at 32,768 even though the model
supports a larger native context. Default concurrency is `1 4 8`, each point
is repeated three times, and the default matrix is deliberately long-running.
A small smoke run is:

```bash
REPEATS=1 CONCURRENCIES="1" MIN_PROMPTS=1 bash run_perf_all.sh
```

Set `RUN_ONE_BATCH=1` to also collect `one_batch.jsonl` for fixed-batch
12K/12K traffic.

## File flow and reports

```text
run_perf_all.sh
  -> launch_perf_server.sh
  -> bench_perf_serving.sh
  -> summarize_perf.py
  -> results/performance/<run-id>/{summary.md,summary.csv,server_summary.md}
```

`summary.md` aggregates throughput and latency medians across repeats.
`server_summary.md` records resolved quantization, effective topology,
startup time, and peak GPU memory. Both the orchestrator and per-mode benchmark
validate `/server_info`; a changed SGLang default fails early instead of
silently changing the comparison.

\n