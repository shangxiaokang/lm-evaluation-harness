#!/usr/bin/env bash

# Shared configuration for Qwen3.8-Flash-Next SGLang performance tests.

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
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "${SGLANG_SOURCE}/.venv/bin/python" ]]; then
    PYTHON_BIN="${SGLANG_SOURCE}/.venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    die "no Python found; set PYTHON_BIN=/path/to/python"
  fi
fi

BF16_MODEL="${BF16_MODEL:-/lustre/fsw/general_sa/xshang/huggingface/Qwen3.8-Flash-Next}"
NVFP4_MODEL="${NVFP4_MODEL:-/lustre/fsw/general_sa/xshang/huggingface/Qwen3.8-Flash-Next-NVFP4}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${BF16_MODEL}}"

export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export HF_HOME="${HF_HOME:-/lustre/fsw/general_sa/xshang/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export PYTHONPATH="${SGLANG_SOURCE}/python${PYTHONPATH:+:${PYTHONPATH}}"

TP_SIZE="${TP_SIZE:-4}"
DP_SIZE="${DP_SIZE:-1}"
EP_SIZE="${EP_SIZE:-1}"
MOE_DP_SIZE="${MOE_DP_SIZE:-1}"
ENABLE_DP_ATTENTION="${ENABLE_DP_ATTENTION:-0}"
ENABLE_DP_LM_HEAD="${ENABLE_DP_LM_HEAD:-0}"
MOE_DENSE_TP_SIZE="${MOE_DENSE_TP_SIZE:-}"
MOE_A2A_BACKEND="${MOE_A2A_BACKEND:-none}"

validate_parallel_topology() {
  local name value
  for name in TP_SIZE DP_SIZE EP_SIZE MOE_DP_SIZE; do
    value="${!name}"
    [[ "${value}" =~ ^[1-9][0-9]*$ ]] || \
      die "${name} must be a positive integer, got: ${value}"
  done

  local moe_parallel_size=$((EP_SIZE * MOE_DP_SIZE))
  (( moe_parallel_size <= TP_SIZE )) || \
    die "EP_SIZE * MOE_DP_SIZE must not exceed TP_SIZE"
  (( TP_SIZE % moe_parallel_size == 0 )) || \
    die "TP_SIZE=${TP_SIZE} must be divisible by EP_SIZE * MOE_DP_SIZE=${moe_parallel_size}"
  if (( MOE_DP_SIZE > 1 && EP_SIZE > 1 )); then
    (( moe_parallel_size == TP_SIZE )) || \
      die "EP_SIZE * MOE_DP_SIZE must equal TP_SIZE when both exceed 1"
  fi

  if is_true "${ENABLE_DP_ATTENTION}"; then
    (( TP_SIZE % DP_SIZE == 0 )) || \
      die "TP_SIZE=${TP_SIZE} must be divisible by DP_SIZE=${DP_SIZE} with DP attention"
    ATTENTION_TP_SIZE=$((TP_SIZE / DP_SIZE))
  else
    ATTENTION_TP_SIZE=${TP_SIZE}
  fi
  if is_true "${ENABLE_DP_LM_HEAD}" && \
     ! is_true "${ENABLE_DP_ATTENTION}"; then
    die "ENABLE_DP_LM_HEAD requires ENABLE_DP_ATTENTION"
  fi

  if [[ -n "${MOE_DENSE_TP_SIZE}" ]]; then
    [[ "${MOE_DENSE_TP_SIZE}" =~ ^[1-9][0-9]*$ ]] || \
      die "MOE_DENSE_TP_SIZE must be empty or a positive integer"
    (( MOE_DENSE_TP_SIZE == 1 || MOE_DENSE_TP_SIZE == TP_SIZE )) || \
      die "MOE_DENSE_TP_SIZE currently supports only 1 or TP_SIZE"
  fi

  MOE_TP_SIZE=$((TP_SIZE / moe_parallel_size))
}

validate_parallel_topology

DTYPE="${DTYPE:-bfloat16}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-bfloat16}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.85}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-8192}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-}"

PAGE_SIZE="${PAGE_SIZE:-64}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-8192}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-32}"
CUDA_GRAPH_MAX_BS_DECODE="${CUDA_GRAPH_MAX_BS_DECODE:-${MAX_RUNNING_REQUESTS}}"

MAMBA_SSM_DTYPE="${MAMBA_SSM_DTYPE:-bfloat16}"
MAMBA_FULL_MEMORY_RATIO="${MAMBA_FULL_MEMORY_RATIO:-0.9}"
MAMBA_RADIX_CACHE_STRATEGY="${MAMBA_RADIX_CACHE_STRATEGY:-extra_buffer_lazy}"
DISABLE_RADIX_CACHE="${DISABLE_RADIX_CACHE:-1}"
MAMBA_SLOTS_PER_REQUEST="${MAMBA_SLOTS_PER_REQUEST:-4}"
if [[ -z "${MAX_MAMBA_CACHE_SIZE:-}" ]]; then
  if is_true "${DISABLE_RADIX_CACHE}"; then
    MAX_MAMBA_CACHE_SIZE="${MAX_RUNNING_REQUESTS}"
  else
    MAX_MAMBA_CACHE_SIZE="$((MAX_RUNNING_REQUESTS * MAMBA_SLOTS_PER_REQUEST))"
  fi
fi

LINEAR_ATTN_PREFILL_BACKEND="${LINEAR_ATTN_PREFILL_BACKEND:-flashinfer}"
LINEAR_ATTN_DECODE_BACKEND="${LINEAR_ATTN_DECODE_BACKEND:-flashinfer}"
MOE_RUNNER_BACKEND="${MOE_RUNNER_BACKEND:-flashinfer_trtllm}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-trtllm_mha}"
SERVER_RANDOM_SEED="${SERVER_RANDOM_SEED:-42}"

# Controlled comparison default: keep every PLE table resident on GPU.
# Set to 1 to offload all three modes, or auto to let SGLang decide per model.
PLE_OFFLOAD_EMBEDDING="${PLE_OFFLOAD_EMBEDDING:-0}"

HOST="${HOST:-127.0.0.1}"
BENCH_HOST="${BENCH_HOST:-127.0.0.1}"
PORT="${PORT:-30000}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.8-flash-next-perf}"
LOG_LEVEL="${LOG_LEVEL:-info}"
SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT:-3600}"

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
      if [[ -n "${OFFLINE_QUANTIZATION:-}" && "${OFFLINE_QUANTIZATION}" != "auto" ]]; then
        EXPECTED_QUANTIZATION="${OFFLINE_QUANTIZATION}"
        MODE_SERVER_ARGS+=(--quantization "${OFFLINE_QUANTIZATION}")
      fi
      ;;
    *)
      die "mode must be bf16, nvfp4_online, or nvfp4_offline"
      ;;
  esac
}

validate_mode_inputs() {
  [[ -x "${PYTHON_BIN}" ]] || die "Python is not executable: ${PYTHON_BIN}"
  [[ -d "${SGLANG_SOURCE}/python/sglang" ]] || \
    die "SGLang source tree not found under ${SGLANG_SOURCE}"
  [[ -f "${MODEL_PATH}/config.json" ]] || \
    die "config.json not found under ${MODEL_PATH}"
  [[ -f "${TOKENIZER_PATH}/tokenizer_config.json" ]] || \
    die "tokenizer_config.json not found under ${TOKENIZER_PATH}"

  if [[ "${MODE}" == "nvfp4_offline" ]]; then
    [[ -f "${MODEL_PATH}/hf_quant_config.json" ]] || \
      die "offline NVFP4 requires ${MODEL_PATH}/hf_quant_config.json"
  elif [[ -f "${MODEL_PATH}/hf_quant_config.json" ]]; then
    die "${MODE} must use the BF16 checkpoint, but hf_quant_config.json exists"
  fi

  local visible_gpus
  local required_gpus
  if is_true "${ENABLE_DP_ATTENTION}"; then
    required_gpus=${TP_SIZE}
  else
    required_gpus=$((TP_SIZE * DP_SIZE))
  fi
  IFS=',' read -r -a visible_gpus <<< "${CUDA_VISIBLE_DEVICES}"
  (( ${#visible_gpus[@]} >= required_gpus )) || \
    die "parallel topology needs ${required_gpus} GPUs; CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
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
    --mamba-full-memory-ratio "${MAMBA_FULL_MEMORY_RATIO}"
    --mamba-radix-cache-strategy "${MAMBA_RADIX_CACHE_STRATEGY}"
    --linear-attn-prefill-backend "${LINEAR_ATTN_PREFILL_BACKEND}"
    --linear-attn-decode-backend "${LINEAR_ATTN_DECODE_BACKEND}"
    --moe-runner-backend "${MOE_RUNNER_BACKEND}"
    --random-seed "${SERVER_RANDOM_SEED}"
    --log-level "${LOG_LEVEL}"
  )

  if [[ -n "${MAX_TOTAL_TOKENS}" ]]; then
    SERVER_ARGS+=(--max-total-tokens "${MAX_TOTAL_TOKENS}")
  fi
  if [[ -n "${MOE_DENSE_TP_SIZE}" ]]; then
    SERVER_ARGS+=(--moe-dense-tp-size "${MOE_DENSE_TP_SIZE}")
  fi
  if is_true "${ENABLE_DP_ATTENTION}"; then
    SERVER_ARGS+=(--enable-dp-attention)
  fi
  if is_true "${ENABLE_DP_LM_HEAD}"; then
    SERVER_ARGS+=(--enable-dp-lm-head)
  fi
  if [[ -n "${ATTENTION_BACKEND}" ]]; then
    SERVER_ARGS+=(--attention-backend "${ATTENTION_BACKEND}")
  fi
  if [[ "${MAX_MAMBA_CACHE_SIZE}" != "auto" ]]; then
    SERVER_ARGS+=(--max-mamba-cache-size "${MAX_MAMBA_CACHE_SIZE}")
  fi

  case "${PLE_OFFLOAD_EMBEDDING,,}" in
    auto) ;;
    1|true|yes|on) SERVER_ARGS+=(--ple-offload-embedding) ;;
    0|false|no|off) SERVER_ARGS+=(--no-ple-offload-embedding) ;;
    *) die "PLE_OFFLOAD_EMBEDDING must be 0, 1, or auto" ;;
  esac

  if is_true "${DISABLE_RADIX_CACHE}"; then
    SERVER_ARGS+=(--disable-radix-cache)
  fi

  SERVER_ARGS+=("${MODE_SERVER_ARGS[@]}")
}

export SGLANG_SOURCE PYTHON_BIN BF16_MODEL NVFP4_MODEL TOKENIZER_PATH
export TP_SIZE DP_SIZE EP_SIZE MOE_DP_SIZE ATTENTION_TP_SIZE MOE_TP_SIZE
export ENABLE_DP_ATTENTION ENABLE_DP_LM_HEAD MOE_DENSE_TP_SIZE MOE_A2A_BACKEND
export DTYPE KV_CACHE_DTYPE MEM_FRACTION_STATIC CONTEXT_LENGTH
export MAX_TOTAL_TOKENS PAGE_SIZE CHUNKED_PREFILL_SIZE MAX_RUNNING_REQUESTS
export CUDA_GRAPH_MAX_BS_DECODE MAX_MAMBA_CACHE_SIZE MAMBA_SSM_DTYPE
export MAMBA_FULL_MEMORY_RATIO MAMBA_RADIX_CACHE_STRATEGY DISABLE_RADIX_CACHE
export MAMBA_SLOTS_PER_REQUEST LINEAR_ATTN_PREFILL_BACKEND
export LINEAR_ATTN_DECODE_BACKEND MOE_RUNNER_BACKEND ATTENTION_BACKEND
export SERVER_RANDOM_SEED
export PLE_OFFLOAD_EMBEDDING HOST BENCH_HOST PORT SERVED_MODEL_NAME LOG_LEVEL
export SERVER_READY_TIMEOUT PERF_OUTPUT_ROOT
