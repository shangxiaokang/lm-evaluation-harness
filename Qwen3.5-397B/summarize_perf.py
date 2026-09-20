#!/usr/bin/env python3
"""Summarize Qwen3.5-397B-A17B SGLang performance results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


MODE_ORDER = {"bf16": 0, "nvfp4_online": 1, "nvfp4_offline": 2}
WORKLOAD_ORDER = {"balanced": 0, "prefill": 1, "decode": 2}
EXPECTED_QUANTIZATION = {
    "bf16": {None, "", "none", "unquant"},
    "nvfp4_online": {"nvfp4_online"},
    "nvfp4_offline": {"modelopt_fp4"},
}
MEDIAN_METRICS = (
    "duration",
    "request_throughput",
    "input_throughput",
    "output_throughput",
    "total_throughput",
    "median_ttft_ms",
    "p95_ttft_ms",
    "p99_ttft_ms",
    "median_tpot_ms",
    "p95_tpot_ms",
    "p99_tpot_ms",
    "median_itl_ms",
    "p95_itl_ms",
    "p99_itl_ms",
    "median_e2e_latency_ms",
    "p95_e2e_latency_ms",
    "p99_e2e_latency_ms",
)
SERVER_CONFIG_FIELDS = (
    "version",
    "model_path",
    "dtype",
    "quantization",
    "tp_size",
    "dp_size",
    "ep_size",
    "moe_dp_size",
    "enable_dp_attention",
    "enable_dp_lm_head",
    "moe_dense_tp_size",
    "moe_a2a_backend",
    "kv_cache_dtype",
    "attention_backend",
    "moe_runner_backend",
    "fp4_gemm_runner_backend",
    "fp8_gemm_runner_backend",
    "bf16_gemm_backend",
    "page_size",
    "disable_radix_cache",
    "context_length",
    "chunked_prefill_size",
    "mem_fraction_static",
    "max_running_requests",
    "cuda_graph_max_bs_decode",
    "max_total_tokens",
    "max_mamba_cache_size",
    "mamba_ssm_dtype",
    "mamba_radix_cache_strategy",
    "speculative_algorithm",
)


def parse_tag(tag: str | None) -> dict[str, str]:
    if not tag:
        return {}
    parsed: dict[str, str] = {}
    for item in tag.split("|"):
        if "=" in item:
            key, value = item.split("=", 1)
            parsed[key] = value
    return parsed


def numeric(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def median(records: list[dict[str, Any]], key: str) -> float | None:
    values = [
        value
        for record in records
        if (value := numeric(record.get(key))) is not None
    ]
    return statistics.median(values) if values else None


def fmt(value: Any, digits: int = 2) -> str:
    return "" if value is None else f"{float(value):.{digits}f}"


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def effective_parallel_sizes(
    info: dict[str, Any],
) -> tuple[int | None, int | None]:
    try:
        tp_size = int(info["tp_size"])
        dp_size = int(info.get("dp_size", 1))
        ep_size = int(info.get("ep_size", 1))
        moe_dp_size = int(info.get("moe_dp_size", 1))
    except (KeyError, TypeError, ValueError):
        return None, None

    enable_dp_attention = info.get("enable_dp_attention", False)
    if isinstance(enable_dp_attention, str):
        enable_dp_attention = enable_dp_attention.lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    attention_divisor = dp_size if enable_dp_attention else 1
    moe_divisor = ep_size * moe_dp_size
    if (
        min(tp_size, attention_divisor, moe_divisor) <= 0
        or tp_size % attention_divisor
        or tp_size % moe_divisor
    ):
        return None, None
    return tp_size // attention_divisor, tp_size // moe_divisor


def selected_server_info(info: dict[str, Any]) -> dict[str, Any]:
    selected = {key: info.get(key) for key in SERVER_CONFIG_FIELDS}
    attention_tp_size, moe_tp_size = effective_parallel_sizes(info)
    selected["effective_attention_tp_size"] = attention_tp_size
    selected["effective_moe_tp_size"] = moe_tp_size
    selected["startup_time"] = info.get("startup_time")
    selected["workers"] = [
        {
            "effective_max_running_requests_per_dp": state.get(
                "effective_max_running_requests_per_dp"
            ),
            "memory_usage": state.get("memory_usage"),
            "startup_time": state.get("startup_time"),
        }
        for state in info.get("internal_states", [])
        if isinstance(state, dict)
    ]
    return selected


def env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def env_bool(name: str, default: str) -> bool:
    value = env(name, default).lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {value!r}")


def normalize_dtype(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    normalized = value.lower().removeprefix("torch.")
    return {"bf16": "bfloat16", "fp16": "float16"}.get(
        normalized, normalized
    )


def config_matches(actual: Any, expected: Any) -> bool:
    if isinstance(expected, bool):
        if isinstance(actual, str):
            lowered = actual.lower()
            if lowered in {"1", "true", "yes", "on"}:
                actual = True
            elif lowered in {"0", "false", "no", "off"}:
                actual = False
        return actual is expected
    if isinstance(expected, int):
        return numeric(actual) == float(expected)
    if isinstance(expected, float):
        value = numeric(actual)
        return value is not None and math.isclose(
            value, expected, rel_tol=1e-6, abs_tol=1e-9
        )
    return normalize_dtype(actual) == normalize_dtype(expected)


def validate_controlled_config(info: dict[str, Any], mode: str) -> None:
    tp_size = int(env("TP_SIZE", "4"))
    dp_size = int(env("DP_SIZE", "4"))
    ep_size = int(env("EP_SIZE", "4"))
    moe_dp_size = int(env("MOE_DP_SIZE", "1"))
    enable_dp_attention = env_bool("ENABLE_DP_ATTENTION", "0")
    requested_chunked_prefill_size = int(
        env("CHUNKED_PREFILL_SIZE", "16384")
    )
    resolved_chunked_prefill_size = requested_chunked_prefill_size
    if enable_dp_attention:
        resolved_chunked_prefill_size //= dp_size

    expected: dict[str, Any] = {
        "tp_size": tp_size,
        "dp_size": dp_size,
        "ep_size": ep_size,
        "moe_dp_size": moe_dp_size,
        "enable_dp_attention": enable_dp_attention,
        "enable_dp_lm_head": env_bool("ENABLE_DP_LM_HEAD", "0"),
        "mem_fraction_static": float(env("MEM_FRACTION_STATIC", "0.80")),
        "context_length": int(env("CONTEXT_LENGTH", "32768")),
        "page_size": int(env("PAGE_SIZE", "64")),
        "chunked_prefill_size": resolved_chunked_prefill_size,
        "max_running_requests": int(env("MAX_RUNNING_REQUESTS", "8")),
        "cuda_graph_max_bs_decode": int(
            env("CUDA_GRAPH_MAX_BS_DECODE", env("MAX_RUNNING_REQUESTS", "8"))
        ),
        "disable_radix_cache": env_bool("DISABLE_RADIX_CACHE", "1"),
    }

    optional_string_fields = {
        "dtype": env("DTYPE", "bfloat16"),
        "kv_cache_dtype": env("KV_CACHE_DTYPE", "bfloat16"),
        "attention_backend": env("ATTENTION_BACKEND", "trtllm_mha"),
        "moe_runner_backend": env(
            "MOE_RUNNER_BACKEND", "flashinfer_cutedsl"
        ),
        "moe_a2a_backend": env("MOE_A2A_BACKEND", "flashinfer"),
        "mamba_ssm_dtype": env("MAMBA_SSM_DTYPE", "bfloat16"),
        "mamba_radix_cache_strategy": env(
            "MAMBA_RADIX_CACHE_STRATEGY", "extra_buffer"
        ),
    }
    for key, value in optional_string_fields.items():
        if value and value.lower() != "auto":
            expected[key] = value

    mamba_size = env("MAX_MAMBA_CACHE_SIZE", "auto")
    if mamba_size.lower() != "auto":
        expected["max_mamba_cache_size"] = int(mamba_size)

    max_total_tokens = env("MAX_TOTAL_TOKENS", "")
    if max_total_tokens:
        expected["max_total_tokens"] = int(max_total_tokens)

    moe_dense_tp_size = env("MOE_DENSE_TP_SIZE", "")
    if moe_dense_tp_size:
        expected["moe_dense_tp_size"] = int(moe_dense_tp_size)

    expected_model = env(
        "NVFP4_MODEL" if mode == "nvfp4_offline" else "BF16_MODEL",
        (
            "/lustre/fsw/general_sa/xshang/huggingface/"
            + (
                "Qwen3.5-397B-A17B-NVFP4"
                if mode == "nvfp4_offline"
                else "Qwen3.5-397B-A17B"
            )
        ),
    )
    actual_model = info.get("model_path")
    if not isinstance(actual_model, str) or os.path.realpath(
        actual_model
    ) != os.path.realpath(expected_model):
        raise ValueError(
            f"controlled-config mismatch: model_path={actual_model!r}, "
            f"expected {expected_model!r}"
        )

    errors = []
    for key, expected_value in expected.items():
        if key not in info:
            errors.append(f"{key}: missing (expected {expected_value!r})")
            continue
        actual_value = info[key]
        if not config_matches(actual_value, expected_value):
            errors.append(f"{key}: got {actual_value!r}, expected {expected_value!r}")

    attention_tp_size, moe_tp_size = effective_parallel_sizes(info)
    expected_attention_tp_size = int(
        env(
            "ATTENTION_TP_SIZE",
            str(tp_size // dp_size if enable_dp_attention else tp_size),
        )
    )
    expected_moe_tp_size = int(
        env("MOE_TP_SIZE", str(tp_size // (ep_size * moe_dp_size)))
    )
    if attention_tp_size != expected_attention_tp_size:
        errors.append(
            "effective_attention_tp_size: got {!r}, expected {!r}".format(
                attention_tp_size, expected_attention_tp_size
            )
        )
    if moe_tp_size != expected_moe_tp_size:
        errors.append(
            "effective_moe_tp_size: got {!r}, expected {!r}".format(
                moe_tp_size, expected_moe_tp_size
            )
        )

    states = [
        state
        for state in info.get("internal_states", [])
        if isinstance(state, dict)
    ]
    attention_dp_size = dp_size if enable_dp_attention else 1
    expected_effective = (
        expected["max_running_requests"] // attention_dp_size
    )
    if not states:
        errors.append("internal_states: missing")
    for index, state in enumerate(states):
        actual_effective = state.get("effective_max_running_requests_per_dp")
        if not config_matches(actual_effective, expected_effective):
            errors.append(
                "internal_states[{}].effective_max_running_requests_per_dp: "
                "got {!r}, expected {!r}".format(
                    index, actual_effective, expected_effective
                )
            )

    if errors:
        raise ValueError(
            "SGLang changed a controlled benchmark setting:\n  - "
            + "\n  - ".join(errors)
        )


def validate_server_info(
    path: Path,
    mode: str,
    expected_quantization: str | None,
    controlled: bool,
) -> None:
    info = read_json(path)
    quantization = info.get("quantization")
    normalized = (
        quantization.lower() if isinstance(quantization, str) else quantization
    )
    if expected_quantization is None:
        allowed = EXPECTED_QUANTIZATION[mode]
    elif expected_quantization.lower() in {"none", "unquant", "null", ""}:
        allowed = EXPECTED_QUANTIZATION["bf16"]
    else:
        allowed = {expected_quantization.lower()}
    if normalized not in allowed:
        choices = ", ".join(
            sorted("null" if item is None else item for item in allowed)
        )
        raise ValueError(
            f"{mode} resolved quantization={quantization!r}; expected one of: "
            f"{choices}"
        )
    if controlled:
        validate_controlled_config(info, mode)
    print(json.dumps(selected_server_info(info), indent=2, sort_keys=True))


def load_records(run_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for result_file in sorted(run_dir.glob("*/serving.jsonl")):
        fallback_mode = result_file.parent.name
        with result_file.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON in {result_file}:{line_number}: {exc}"
                    ) from exc
                tag = parse_tag(record.get("tag"))
                record["_mode"] = tag.get("mode", fallback_mode)
                record["_workload"] = tag.get("workload", "unknown")
                record["_repeat"] = int(tag["repeat"]) if "repeat" in tag else None
                record["_expected_prompts"] = (
                    int(tag["n"]) if "n" in tag else None
                )
                records.append(record)
    return records


def record_group_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        record["_workload"],
        record.get("random_input_len"),
        record.get("random_output_len"),
        record.get("max_concurrency"),
    )


def validate_records(
    records: list[dict[str, Any]],
    expected_modes: list[str] | None,
    expected_repeats: int | None,
) -> None:
    errors = []
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)

    for record in records:
        mode = str(record["_mode"])
        group_key = (mode, *record_group_key(record))
        grouped[group_key].append(record)

        expected_prompts = record.get("_expected_prompts")
        completed = record.get("completed")
        if expected_prompts is None:
            errors.append(f"{group_key}: result tag has no n=<num_prompts>")
        elif numeric(completed) != float(expected_prompts):
            errors.append(
                f"{group_key}: completed={completed!r}, "
                f"expected {expected_prompts}"
            )

        tag = parse_tag(record.get("tag"))
        tag_expectations = {
            "isl": record.get("random_input_len"),
            "osl": record.get("random_output_len"),
            "c": record.get("max_concurrency"),
        }
        for tag_key, result_value in tag_expectations.items():
            try:
                tag_value = int(tag[tag_key])
            except (KeyError, ValueError):
                errors.append(f"{group_key}: invalid or missing tag field {tag_key}")
                continue
            if numeric(result_value) != float(tag_value):
                errors.append(
                    f"{group_key}: tag {tag_key}={tag_value}, "
                    f"result has {result_value!r}"
                )

    for group_key, items in grouped.items():
        repeats = [item.get("_repeat") for item in items]
        if None in repeats or len(set(repeats)) != len(repeats):
            errors.append(f"{group_key}: invalid/duplicate repeats {repeats}")
        if expected_repeats is not None:
            expected_repeat_values = set(range(1, expected_repeats + 1))
            if set(repeats) != expected_repeat_values:
                errors.append(
                    f"{group_key}: repeats={repeats}, "
                    f"expected {sorted(expected_repeat_values)}"
                )

    present_modes = {str(record["_mode"]) for record in records}
    if expected_modes is not None:
        requested_modes = set(expected_modes)
        if present_modes != requested_modes:
            errors.append(
                f"modes present={sorted(present_modes)}, "
                f"expected={sorted(requested_modes)}"
            )
        reference_keys: set[tuple[Any, ...]] | None = None
        reference_mode = ""
        for mode in expected_modes:
            mode_keys = {
                record_group_key(record)
                for record in records
                if record["_mode"] == mode
            }
            if reference_keys is None:
                reference_keys = mode_keys
                reference_mode = mode
            elif mode_keys != reference_keys:
                errors.append(
                    f"{mode} workload matrix differs from {reference_mode}: "
                    f"missing={sorted(reference_keys - mode_keys, key=repr)!r}, "
                    f"extra={sorted(mode_keys - reference_keys, key=repr)!r}"
                )

    if errors:
        raise ValueError(
            "benchmark result validation failed:\n  - " + "\n  - ".join(errors)
        )


def aggregate(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = (
            record["_mode"],
            record["_workload"],
            record.get("random_input_len"),
            record.get("random_output_len"),
            record.get("max_concurrency"),
        )
        groups[key].append(record)

    rows: list[dict[str, Any]] = []
    for (mode, workload, input_len, output_len, concurrency), items in groups.items():
        row: dict[str, Any] = {
            "mode": mode,
            "workload": workload,
            "input_len": input_len,
            "output_len": output_len,
            "max_concurrency": concurrency,
            "repeats": len(items),
            "completed_min": min(int(item.get("completed", 0)) for item in items),
        }
        for metric in MEDIAN_METRICS:
            row[metric] = median(items, metric)
        rows.append(row)

    baseline = {
        (
            row["workload"],
            row["input_len"],
            row["output_len"],
            row["max_concurrency"],
        ): row
        for row in rows
        if row["mode"] == "bf16"
    }
    for row in rows:
        base = baseline.get(
            (
                row["workload"],
                row["input_len"],
                row["output_len"],
                row["max_concurrency"],
            )
        )
        for metric in ("output_throughput", "total_throughput"):
            base_value = None if base is None else base.get(metric)
            value = row.get(metric)
            row[f"{metric}_vs_bf16"] = (
                value / base_value if value is not None and base_value else None
            )
        for metric in (
            "median_ttft_ms",
            "median_tpot_ms",
            "p99_e2e_latency_ms",
        ):
            base_value = None if base is None else base.get(metric)
            value = row.get(metric)
            row[f"{metric}_speedup_vs_bf16"] = (
                base_value / value if value and base_value is not None else None
            )

    rows.sort(
        key=lambda row: (
            WORKLOAD_ORDER.get(str(row["workload"]), 99),
            int(row["max_concurrency"] or 0),
            MODE_ORDER.get(str(row["mode"]), 99),
        )
    )
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "mode",
        "workload",
        "input_len",
        "output_len",
        "max_concurrency",
        "repeats",
        "completed_min",
        *MEDIAN_METRICS,
        "output_throughput_vs_bf16",
        "total_throughput_vs_bf16",
        "median_ttft_ms_speedup_vs_bf16",
        "median_tpot_ms_speedup_vs_bf16",
        "p99_e2e_latency_ms_speedup_vs_bf16",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# Qwen3.5-397B-A17B SGLang performance summary",
        "",
        "Values are medians across runs. Speedup values above 1 are better.",
        "",
        "| Mode | Workload | ISL | OSL | C | Runs | Output tok/s | vs BF16 | Total tok/s | TTFT median ms | TTFT p99 ms | TPOT median ms | TPOT p99 ms | E2E p99 ms |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {mode} | {workload} | {input_len} | {output_len} | "
            "{max_concurrency} | {repeats} | {output} | {speedup} | "
            "{total} | {ttft} | {ttft_p99} | {tpot} | {tpot_p99} | "
            "{e2e_p99} |".format(
                **row,
                output=fmt(row.get("output_throughput")),
                speedup=fmt(row.get("output_throughput_vs_bf16"), 3),
                total=fmt(row.get("total_throughput")),
                ttft=fmt(row.get("median_ttft_ms")),
                ttft_p99=fmt(row.get("p99_ttft_ms")),
                tpot=fmt(row.get("median_tpot_ms")),
                tpot_p99=fmt(row.get("p99_tpot_ms")),
                e2e_p99=fmt(row.get("p99_e2e_latency_ms")),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def state_values(states: list[dict[str, Any]], key: str) -> list[float]:
    values = []
    for state in states:
        memory = state.get("memory_usage")
        if isinstance(memory, dict) and (value := numeric(memory.get(key))) is not None:
            values.append(value)
    return values


def graph_memory_values(states: list[dict[str, Any]]) -> list[float]:
    values = []
    for state in states:
        memory = state.get("memory_usage")
        graph = memory.get("graph") if isinstance(memory, dict) else None
        if isinstance(graph, dict):
            values.append(sum(numeric(value) or 0.0 for value in graph.values()))
    return values


def startup_value(startup: Any, key: str) -> float | None:
    return numeric(startup.get(key)) if isinstance(startup, dict) else None


def cuda_graph_time(startup: Any) -> float | None:
    graph = startup.get("cuda_graph") if isinstance(startup, dict) else None
    if not isinstance(graph, dict):
        return None
    return sum(numeric(value) or 0.0 for value in graph.values())


def gpu_peak_hbm(gpu_file: Path) -> tuple[float | None, str]:
    if not gpu_file.is_file():
        return None, ""
    per_device: dict[str, float] = {}
    per_timestamp: dict[str, float] = defaultdict(float)
    with gpu_file.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            index = (row.get("index") or "").strip()
            timestamp = (row.get("timestamp") or "").strip()
            try:
                memory_mib = float((row.get("memory_used_mib") or "").strip())
            except ValueError:
                continue
            per_device[index] = max(per_device.get(index, 0.0), memory_mib)
            if timestamp:
                per_timestamp[timestamp] += memory_mib
    if not per_device:
        return None, ""
    by_device = ";".join(
        f"gpu{index}={memory_mib / 1024:.3f}GiB"
        for index, memory_mib in sorted(per_device.items())
    )
    peak_total_mib = (
        max(per_timestamp.values())
        if per_timestamp
        else sum(per_device.values())
    )
    return peak_total_mib / 1024, by_device


def collect_server_rows(run_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for info_path in sorted(run_dir.glob("*/server_info.json")):
        mode = info_path.parent.name
        info = read_json(info_path)
        states = [
            state
            for state in info.get("internal_states", [])
            if isinstance(state, dict)
        ]
        startup = info.get("startup_time")
        weight_values = state_values(states, "weight")
        kvcache_values = state_values(states, "kvcache")
        token_values = state_values(states, "token_capacity")
        graph_values = graph_memory_values(states)
        running_values = [
            value
            for state in states
            if (
                value := numeric(
                    state.get("effective_max_running_requests_per_dp")
                )
            )
            is not None
        ]
        ready_file = info_path.parent / "time_to_ready_seconds.txt"
        ready_value = (
            float(ready_file.read_text(encoding="utf-8").strip())
            if ready_file.is_file()
            else None
        )
        peak_hbm, peak_hbm_by_device = gpu_peak_hbm(
            info_path.parent / "gpu_metrics.csv"
        )
        attention_tp_size, moe_tp_size = effective_parallel_sizes(info)
        rows.append(
            {
                "mode": mode,
                "time_to_ready_s": ready_value,
                "sglang_version": info.get("version"),
                "quantization": info.get("quantization"),
                "dtype": info.get("dtype"),
                "kv_cache_dtype": info.get("kv_cache_dtype"),
                "tp_size": info.get("tp_size"),
                "dp_size": info.get("dp_size"),
                "ep_size": info.get("ep_size"),
                "moe_dp_size": info.get("moe_dp_size"),
                "enable_dp_attention": info.get("enable_dp_attention"),
                "enable_dp_lm_head": info.get("enable_dp_lm_head"),
                "effective_attention_tp_size": attention_tp_size,
                "effective_moe_tp_size": moe_tp_size,
                "moe_dense_tp_size": info.get("moe_dense_tp_size"),
                "moe_a2a_backend": info.get("moe_a2a_backend"),
                "attention_backend": info.get("attention_backend"),
                "moe_runner_backend": info.get("moe_runner_backend"),
                "fp4_gemm_runner_backend": info.get(
                    "fp4_gemm_runner_backend"
                ),
                "fp8_gemm_runner_backend": info.get(
                    "fp8_gemm_runner_backend"
                ),
                "bf16_gemm_backend": info.get("bf16_gemm_backend"),
                "disable_radix_cache": info.get("disable_radix_cache"),
                "max_running_requests": info.get("max_running_requests"),
                "max_mamba_cache_size": info.get("max_mamba_cache_size"),
                "effective_max_running_requests_per_dp_min": (
                    min(running_values) if running_values else None
                ),
                "weight_gb_per_worker_max": (
                    max(weight_values) if weight_values else None
                ),
                "kvcache_gb_per_worker_max": (
                    max(kvcache_values) if kvcache_values else None
                ),
                "graph_gb_per_worker_max": (
                    max(graph_values) if graph_values else None
                ),
                "token_capacity_per_worker_min": (
                    int(min(token_values)) if token_values else None
                ),
                "gpu_peak_hbm_gib_sum": peak_hbm,
                "gpu_peak_hbm_by_device": peak_hbm_by_device,
                "load_weight_s": startup_value(startup, "load_weight"),
                "kv_cache_allocation_s": startup_value(
                    startup, "kv_cache_allocation"
                ),
                "cuda_graph_s": cuda_graph_time(startup),
                "scheduler_e2e_s": startup_value(startup, "scheduler_e2e"),
                "tokenizer_e2e_s": startup_value(startup, "tokenizer_e2e"),
            }
        )
    rows.sort(key=lambda row: MODE_ORDER.get(str(row["mode"]), 99))
    return rows


def write_server_summary(rows: list[dict[str, Any]], run_dir: Path) -> None:
    if not rows:
        return
    csv_path = run_dir / "server_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Server startup and memory summary",
        "",
        "Memory breakdown values are per-worker maxima from `/server_info`.",
        "",
        "| Mode | Quantization | Topology | Ready s | Load weight s | "
        "Weight GB | KV GB | Token capacity | Peak HBM GiB (sum) |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {mode} | {quantization} | {topology} | {ready} | {load} | "
            "{weight} | "
            "{kv} | {tokens} | {peak} |".format(
                **row,
                topology="Attn TP{} / MoE TP{} / EP{}".format(
                    row.get("effective_attention_tp_size") or "?",
                    row.get("effective_moe_tp_size") or "?",
                    row.get("ep_size") or "?",
                ),
                ready=fmt(row.get("time_to_ready_s")),
                load=fmt(row.get("load_weight_s")),
                weight=fmt(row.get("weight_gb_per_worker_max"), 3),
                kv=fmt(row.get("kvcache_gb_per_worker_max"), 3),
                tokens=row.get("token_capacity_per_worker_min") or "",
                peak=fmt(row.get("gpu_peak_hbm_gib_sum"), 3),
            )
        )
    (run_dir / "server_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", nargs="?", type=Path)
    parser.add_argument("--validate-server-info", type=Path)
    parser.add_argument("--mode", choices=tuple(MODE_ORDER))
    parser.add_argument("--expected-quantization")
    parser.add_argument("--validate-controlled-config", action="store_true")
    parser.add_argument(
        "--expected-modes", nargs="+", choices=tuple(MODE_ORDER)
    )
    parser.add_argument("--expected-repeats", type=int)
    args = parser.parse_args()

    if args.validate_server_info is not None:
        if args.mode is None:
            parser.error("--validate-server-info requires --mode")
        validate_server_info(
            args.validate_server_info.resolve(),
            args.mode,
            args.expected_quantization,
            args.validate_controlled_config,
        )
        return 0

    if args.mode is not None or args.expected_quantization is not None:
        parser.error("--mode/--expected-quantization require --validate-server-info")
    if args.validate_controlled_config:
        parser.error("--validate-controlled-config requires --validate-server-info")
    if args.expected_repeats is not None and args.expected_repeats <= 0:
        parser.error("--expected-repeats must be positive")

    if args.run_dir is None:
        parser.error("run_dir is required when not validating server info")
    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        parser.error(f"run directory does not exist: {run_dir}")

    records = load_records(run_dir)
    if not records:
        parser.error(f"no */serving.jsonl records found under {run_dir}")

    validate_records(records, args.expected_modes, args.expected_repeats)
    rows = aggregate(records)
    csv_path = run_dir / "summary.csv"
    markdown_path = run_dir / "summary.md"
    write_csv(rows, csv_path)
    write_markdown(rows, markdown_path)
    write_server_summary(collect_server_rows(run_dir), run_dir)
    print(f"Wrote {csv_path}")
    print(f"Wrote {markdown_path}")
    print(f"Wrote {run_dir / 'server_summary.csv'}")
    print(f"Wrote {run_dir / 'server_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
