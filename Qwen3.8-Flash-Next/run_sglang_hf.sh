#!/bin/bash
# =============================================================================
# lm-eval SGLang: local Qwen3.8-Flash-Next BF16 on 4x GB200
# =============================================================================
# Loads the BF16 checkpoint without weight quantization. Text-only mode skips
# the vision tower, and the large PLE n-gram embedding is kept in host memory.
#
# Requires a Qwen4Exp-capable SGLang build.
#
# E.g. on the node:
#   bash run_sglang_hf.sh
#   LIMIT=16 bash run_sglang_hf.sh
#   TASK=arc_easy bash run_sglang_hf.sh
# =============================================================================

set -euo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export HF_HOME="${HF_HOME:-/lustre/fsw/general_sa/xshang/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
HARNESS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${HARNESS_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

MODEL_PATH="${MODEL_PATH:-/lustre/fsw/general_sa/xshang/huggingface/Qwen3.8-Flash-Next}"
TASK="${TASK:-mmlu_pro}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_FEWSHOT="${NUM_FEWSHOT:-0}"
DTYPE="${DTYPE:-bfloat16}"
TP_SIZE="${TP_SIZE:-4}"
DP_SIZE="${DP_SIZE:-1}"
MEM_FRACTION="${MEM_FRACTION:-0.9}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-True}"
LIMIT="${LIMIT:-}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"

LANGUAGE_MODEL_ONLY="${LANGUAGE_MODEL_ONLY:-True}"
PLE_OFFLOAD_EMBEDDING="${PLE_OFFLOAD_EMBEDDING:-True}"

# Qwen3.8 uses QSA plus Gated DeltaNet. It needs three Mamba slots per request.
PAGE_SIZE="${PAGE_SIZE:-64}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-4096}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-32}"
MAMBA_SLOTS_PER_REQ="${MAMBA_SLOTS_PER_REQ:-3}"
MAX_MAMBA_CACHE_SIZE="${MAX_MAMBA_CACHE_SIZE:-$((MAX_RUNNING_REQUESTS * MAMBA_SLOTS_PER_REQ))}"
MAMBA_SSM_DTYPE="${MAMBA_SSM_DTYPE:-bfloat16}"
MAMBA_FULL_MEMORY_RATIO="${MAMBA_FULL_MEMORY_RATIO:-0.9}"
MAMBA_RADIX_CACHE_STRATEGY="${MAMBA_RADIX_CACHE_STRATEGY:-extra_buffer_lazy}"

OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/results/sglang_hf}"
LOG_FILE="${SCRIPT_DIR}/sglang_hf_eval_${TASK}_TP${TP_SIZE}_DP${DP_SIZE}_bs${BATCH_SIZE}.log"

mkdir -p "${OUTPUT_DIR}" "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}"

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "Error: no config.json under MODEL_PATH=${MODEL_PATH}" >&2
  exit 1
fi

if [[ -f "${MODEL_PATH}/hf_quant_config.json" ]]; then
  echo "Error: ${MODEL_PATH} looks like a quantized ModelOpt checkpoint." >&2
  echo "This script requires the BF16 Qwen3.8-Flash-Next checkpoint." >&2
  exit 1
fi

IFS=',' read -r -a _sglang_gpus <<< "${CUDA_VISIBLE_DEVICES}"
REQUIRED_GPUS=$((TP_SIZE * DP_SIZE))
if [[ "${#_sglang_gpus[@]}" -lt "${REQUIRED_GPUS}" ]]; then
  echo "Error: TP_SIZE=${TP_SIZE}, DP_SIZE=${DP_SIZE} require at least" >&2
  echo "       ${REQUIRED_GPUS} visible GPUs, but CUDA_VISIBLE_DEVICES=" >&2
  echo "       ${CUDA_VISIBLE_DEVICES} has ${#_sglang_gpus[@]}." >&2
  exit 1
fi

MODEL_ARGS="pretrained=${MODEL_PATH},dtype=${DTYPE},trust_remote_code=${TRUST_REMOTE_CODE},tp_size=${TP_SIZE},dp_size=${DP_SIZE},mem_fraction_static=${MEM_FRACTION},max_model_len=${MAX_MODEL_LEN},kv_cache_dtype=${KV_CACHE_DTYPE},language_model_only=${LANGUAGE_MODEL_ONLY},ple_offload_embedding=${PLE_OFFLOAD_EMBEDDING},page_size=${PAGE_SIZE},chunked_prefill_size=${CHUNKED_PREFILL_SIZE},max_running_requests=${MAX_RUNNING_REQUESTS},max_mamba_cache_size=${MAX_MAMBA_CACHE_SIZE},mamba_ssm_dtype=${MAMBA_SSM_DTYPE},mamba_full_memory_ratio=${MAMBA_FULL_MEMORY_RATIO},mamba_radix_cache_strategy=${MAMBA_RADIX_CACHE_STRATEGY}"

if command -v lm_eval >/dev/null 2>&1; then
  LM_EVAL=(lm_eval)
else
  LM_EVAL=(python3 -m lm_eval)
fi

echo "=============================================="
echo "lm-eval SGLang backend (Qwen3.8 BF16, 4x GB200)"
echo "=============================================="
echo "Model path:  ${MODEL_PATH}"
echo "Task:        ${TASK}"
echo "GPUs:        ${CUDA_VISIBLE_DEVICES}"
echo "TP / DP:     ${TP_SIZE} / ${DP_SIZE}"
echo "Batch size:  ${BATCH_SIZE}"
echo "Max len:     ${MAX_MODEL_LEN}"
echo "Mem frac:    ${MEM_FRACTION}"
echo "Dtype:       ${DTYPE}"
echo "KV cache:    ${KV_CACHE_DTYPE}"
echo "Text only:   ${LANGUAGE_MODEL_ONLY}"
echo "PLE offload: ${PLE_OFFLOAD_EMBEDDING}"
echo "Mamba cache: ${MAX_MAMBA_CACHE_SIZE}"
echo "Limit:       ${LIMIT:-none (full task)}"
echo "model_args:  ${MODEL_ARGS}"
echo "Output dir:  ${OUTPUT_DIR}"
echo "Log file:    ${LOG_FILE}"
echo "=============================================="

cd "${HARNESS_DIR}"

EVAL_CMD=(
  "${LM_EVAL[@]}"
  --model sglang
  --model_args "${MODEL_ARGS}"
  --tasks "${TASK}"
  --batch_size "${BATCH_SIZE}"
  --num_fewshot "${NUM_FEWSHOT}"
  --output_path "${OUTPUT_DIR}"
)

if [[ -n "${LIMIT}" ]]; then
  EVAL_CMD+=(--limit "${LIMIT}")
fi

"${EVAL_CMD[@]}" 2>&1 | tee "${LOG_FILE}"

echo "=============================================="
echo "Evaluation completed."
echo "Results: ${OUTPUT_DIR}"
echo "Log:     ${LOG_FILE}"
echo "=============================================="
