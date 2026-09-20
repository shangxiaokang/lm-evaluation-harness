#!/usr/bin/env bash
# Shared configuration for Qwen3.5-397B-A17B SGLang performance tests.

set -euo pipefail

PERF_SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"

die() {
  echo "Error: $*" >&2
  exit 1
}

is_true() {
  case "${1,,}" in
    1|true|yes|on) return 0 ;;
    0|false|no|off) return 1 ;;
    *) die "expected a boolean value, got: $1" ;;
  esac
}

SGLANG_SOURCE="${SGLANG_SOURCE:-/lustre/fsw/general_sa/xshang/sglang}"
PYTHON_BIN="${PYTHON_BIN:-${SGLANG_SOURCE}/.venv/bin/python}"
[[ -x "${PYTHON_BIN}" ]] || die "Python is not executable: ${PYTHON_BIN}"

BF16_MODEL="${BF16_MODEL:-/lustre/fsw/general_sa/xshang/huggingface/Qwen3.5-397B-A17B}"
NVFP4_MODEL="${NVFP4_MODEL:-/lustre/fsw/general_sa/xshang/huggingface/Qwen3.5-397B-A17B-NVFP4}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${BF16_MODEL}}"

export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export HF_HOME="${HF_HOME:-/lustre/fsw/general_sa/xshang/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export PYTHONPATH="${SGLANG_SOURCE}/python${PYTHONPATH:+:${PYTHONPATH}}"

# SGLang represents EP as a sub-dimension of the outer TP group. With DPA,
# TP=4, DP=4, EP=4 means four physical GPUs, Attention TP=1 and MoE TP=1.
TP_SIZE="${TP_SIZE:-4}"
DP_SIZE="${DP_SIZE:-4}"
EP_SIZE="${EP_SIZE:-4}"
MOE_DP_SIZE="${MOE_DP_SIZE:-1}"
ENABLE_DP_ATTENTION="${ENABLE_DP_ATTENTION:-1}"
ENABLE_DP_LM_HEAD="${ENABLE_DP_LM_HEAD:-1}"
MOE_DENSE_TP_SIZE="${MOE_DENSE_TP_SIZE:-}"
MOE_A2A_BACKEND="${MOE_A2A_BACKEND:-flashinfer}"

validate_parallel_topology() {
  local name value
  for name in TP_SIZE DP_SIZE EP_SIZE MOE_DP_SIZE; do
    value="${!name}"
    [[ "${value}" =~ ^[1-9][0-9]*$ ]] || die "${name} must be a positive integer, got: ${value}"
  done

  local moe_parallel_size=$((EP_SIZE * MOE_DP_SIZE))
  (( moe_parallel_size <= TP_SIZE )) || die "EP_SIZE * MOE_DP_SIZE must not exceed TP_SIZE"
  (( TP_SIZE % moe_parallel_size == 0 )) || die "TP_SIZE must be divisible by EP_SIZE * MOE_DP_SIZE"
  if is_true "${ENABLE_DP_ATTENTION}"; then
    (( TP_SIZE % DP_SIZE == 0 )) || die "TP_SIZE must be divisible by DP_SIZE with DP attention"
    ATTENTION_TP_SIZE=$((TP_SIZE / DP_SIZE))
  else
    ATTENTION_TP_SIZE=${TP_SIZE}
  fi
  if is_true "${ENABLE_DP_LM_HEAD}" && ! is_true "${ENABLE_DP_ATTENTION}"; then
    die "ENABLE_DP_LM_HEAD requires ENABLE_DP_ATTENTION"
  fi
  if [[ -n "${MOE_DENSE_TP_SIZE}" ]]; then
    [[ "${MOE_DENSE_TP_SIZE}" =~ ^[1-9][0-9]*$ ]] || die "MOE_DENSE_TP_SIZE must be empty or a positive integer"
    (( MOE_DENSE_TP_SIZE == 1 || MOE_DENSE_TP_SIZE == TP_SIZE )) || die "MOE_DENSE_TP_SIZE supports only 1 or TP_SIZE"
  fi
  MOE_TP_SIZE=$((TP_SIZE / moe_parallel_size))
}

validate_parallel_topology

DTYPE="${DTYPE:-bfloat16}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-bfloat16}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.80}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-32768}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-}"
PAGE_SIZE="${PAGE_SIZE:-64}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-16384}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-8}"
CUDA_GRAPH_MAX_BS_DECODE="${CUDA_GRAPH_MAX_BS_DECODE:-${MAX_RUNNING_REQUESTS}}"
MAMBA_SSM_DTYPE="${MAMBA_SSM_DTYPE:-bfloat16}"
MAMBA_RADIX_CACHE_STRATEGY="${MAMBA_RADIX_CACHE_STRATEGY:-extra_buffer}"
MAMBA_TRACK_INTERVAL="${MAMBA_TRACK_INTERVAL:-128}"
MAX_MAMBA_CACHE_SIZE="${MAX_MAMBA_CACHE_SIZE:-auto}"
DISABLE_RADIX_CACHE="${DISABLE_RADIX_CACHE:-1}"
MOE_RUNNER_BACKEND="${MOE_RUNNER_BACKEND:-flashinfer_cutedsl}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-trtllm_mha}"
MODEL_LOADER_THREADS="${MODEL_LOADER_THREADS:-64}"
WATCHDOG_TIMEOUT="${WATCHDOG_TIMEOUT:-1200}"
SERVER_RANDOM_SEED="${SERVER_RANDOM_SEED:-42}"

HOST="${HOST:-127.0.0.1}"
BENCH_HOST="${BENCH_HOST:-127.0.0.1}"
PORT="${PORT:-30000}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.5-397b-perf}"
LOG_LEVEL="${LOG_LEVEL:-info}"
SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT:-7200}"
PERF_OUTPUT_ROOT="${PERF_OUTPUT_ROOT:-${PERF_SCRIPT_DIR}/results/performance}"

resolve_mode() {
  local requested_mode="${1:-}"
  MODE_SERVER_ARGS=()
  case "${requested_mode}" in
    bf16)
      MODE="bf16"
      MODEL_PATH="${BF16_MODEL}"
      EXPECTED_QUANTIZATION="none"
      ;;
    nvfp4_online)
      MODE="nvfp4_online"
      MODEL_PATH="${BF16_MODEL}"
      EXPECTED_QUANTIZATION="nvfp4_online"
      MODE_SERVER_ARGS+=(--quantization nvfp4_online)
      ;;
    nvfp4_offline)
      MODE="nvfp4_offline"
      MODEL_PATH="${NVFP4_MODEL}"
      EXPECTED_QUANTIZATION="modelopt_fp4"
      MODE_SERVER_ARGS+=(--quantization modelopt_fp4)
      ;;
    *) die "mode must be bf16, nvfp4_online, or nvfp4_offline" ;;
  esac
}

validate_mode_inputs() {
  [[ -d "${SGLANG_SOURCE}/python/sglang" ]] || die "SGLang source tree not found under ${SGLANG_SOURCE}"
  [[ -f "${MODEL_PATH}/config.json" ]] || die "config.json not found under ${MODEL_PATH}"
  [[ -f "${TOKENIZER_PATH}/tokenizer_config.json" ]] || die "tokenizer_config.json not found under ${TOKENIZER_PATH}"

  if [[ "${MODE}" == "nvfp4_offline" ]]; then
    [[ -f "${MODEL_PATH}/hf_quant_config.json" ]] || die "offline NVFP4 requires ${MODEL_PATH}/hf_quant_config.json"
  elif [[ -f "${MODEL_PATH}/hf_quant_config.json" ]]; then
    die "${MODE} must use the BF16 checkpoint, but hf_quant_config.json exists"
  fi

  if [[ "${MODE}" == "bf16" && "${TP_SIZE}" == "4" && "${DP_SIZE}" == "4" && "${EP_SIZE}" == "4" ]] && ! is_true "${ALLOW_UNSAFE_BF16:-0}"; then
    die "Qwen3.5-397B BF16 needs at least 8x B200/GB200 GPUs. The requested 4-GPU EP4+DP4 topology is only safe for NVFP4. Set ALLOW_UNSAFE_BF16=1 only on 4x B300-class HBM."
  fi

  local visible_gpus required_gpus
  if is_true "${ENABLE_DP_ATTENTION}"; then
    required_gpus=${TP_SIZE}
  else
    required_gpus=$((TP_SIZE * DP_SIZE))
  fi
  IFS=',' read -r -a visible_gpus <<< "${CUDA_VISIBLE_DEVICES}"
  (( ${#visible_gpus[@]} >= required_gpus )) || die "parallel topology needs ${required_gpus} GPUs; CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
}

build_server_args() {
  SERVER_ARGS=(
    --model-path "${MODEL_PATH}"
    --served-model-name "${SERVED_MODEL_NAME}"
    --trust-remote-code
    --host "${HOST}"
    --port "${PORT}"
    --tp-size "${TP_SIZE}"
    --dp-size "${DP_SIZE}"
    --ep-size "${EP_SIZE}"
    --moe-dp-size "${MOE_DP_SIZE}"
    --moe-a2a-backend "${MOE_A2A_BACKEND}"
    --dtype "${DTYPE}"
    --kv-cache-dtype "${KV_CACHE_DTYPE}"
    --context-length "${CONTEXT_LENGTH}"
    --mem-fraction-static "${MEM_FRACTION_STATIC}"
    --page-size "${PAGE_SIZE}"
    --chunked-prefill-size "${CHUNKED_PREFILL_SIZE}"
    --max-running-requests "${MAX_RUNNING_REQUESTS}"
    --cuda-graph-max-bs-decode "${CUDA_GRAPH_MAX_BS_DECODE}"
    --mamba-ssm-dtype "${MAMBA_SSM_DTYPE}"
    --mamba-radix-cache-strategy "${MAMBA_RADIX_CACHE_STRATEGY}"
    --mamba-track-interval "${MAMBA_TRACK_INTERVAL}"
    --attention-backend "${ATTENTION_BACKEND}"
    --moe-runner-backend "${MOE_RUNNER_BACKEND}"
    --model-loader-extra-config "$(printf '{"enable_multithread_load":true,"num_threads":%s}' "${MODEL_LOADER_THREADS}")"
    --watchdog-timeout "${WATCHDOG_TIMEOUT}"
    --random-seed "${SERVER_RANDOM_SEED}"
    --log-level "${LOG_LEVEL}"
  )
  [[ -z "${MAX_TOTAL_TOKENS}" ]] || SERVER_ARGS+=(--max-total-tokens "${MAX_TOTAL_TOKENS}")
  [[ -z "${MOE_DENSE_TP_SIZE}" ]] || SERVER_ARGS+=(--moe-dense-tp-size "${MOE_DENSE_TP_SIZE}")
  is_true "${ENABLE_DP_ATTENTION}" && SERVER_ARGS+=(--enable-dp-attention)
  is_true "${ENABLE_DP_LM_HEAD}" && SERVER_ARGS+=(--enable-dp-lm-head)
  [[ "${MAX_MAMBA_CACHE_SIZE}" == "auto" ]] || SERVER_ARGS+=(--max-mamba-cache-size "${MAX_MAMBA_CACHE_SIZE}")
  is_true "${DISABLE_RADIX_CACHE}" && SERVER_ARGS+=(--disable-radix-cache)
  SERVER_ARGS+=("${MODE_SERVER_ARGS[@]}")
}

export SGLANG_SOURCE PYTHON_BIN BF16_MODEL NVFP4_MODEL TOKENIZER_PATH
export TP_SIZE DP_SIZE EP_SIZE MOE_DP_SIZE ATTENTION_TP_SIZE MOE_TP_SIZE
export ENABLE_DP_ATTENTION ENABLE_DP_LM_HEAD MOE_DENSE_TP_SIZE MOE_A2A_BACKEND
export DTYPE KV_CACHE_DTYPE MEM_FRACTION_STATIC CONTEXT_LENGTH MAX_TOTAL_TOKENS
export PAGE_SIZE CHUNKED_PREFILL_SIZE MAX_RUNNING_REQUESTS CUDA_GRAPH_MAX_BS_DECODE
export MAMBA_SSM_DTYPE MAMBA_RADIX_CACHE_STRATEGY MAMBA_TRACK_INTERVAL MAX_MAMBA_CACHE_SIZE
export DISABLE_RADIX_CACHE MOE_RUNNER_BACKEND ATTENTION_BACKEND MODEL_LOADER_THREADS WATCHDOG_TIMEOUT
export SERVER_RANDOM_SEED HOST BENCH_HOST PORT SERVED_MODEL_NAME LOG_LEVEL SERVER_READY_TIMEOUT PERF_OUTPUT_ROOT
