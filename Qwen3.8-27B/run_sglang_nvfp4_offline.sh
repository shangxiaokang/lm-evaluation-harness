#!/bin/bash
# =============================================================================
# lm-eval SGLang: Qwen3.8-27B offline NVFP4/FP8 W4A4 on 2x GB200
# =============================================================================

set -euo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export HF_HOME="${HF_HOME:-/lustre/fsw/general_sa/xshang/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"

# Keep the checkpoint's NVFP4 layers in W4A4 execution mode.
export SGLANG_FLASHINFER_CUTEDSL_NVFP4_W4A16=0

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
HARNESS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${HARNESS_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

MODEL_PATH="${MODEL_PATH:-/lustre/fsw/general_sa/xshang/huggingface/Qwen3.8-27B-NVFP4}"
TASK="${TASK:-mmlu_pro}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_FEWSHOT="${NUM_FEWSHOT:-0}"
DTYPE="${DTYPE:-auto}"
TP_SIZE="${TP_SIZE:-2}"
DP_SIZE="${DP_SIZE:-1}"
MEM_FRACTION="${MEM_FRACTION:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-True}"
LIMIT="${LIMIT:-}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"

LANGUAGE_MODEL_ONLY="${LANGUAGE_MODEL_ONLY:-True}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-2048}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-16}"
MAMBA_SLOTS_PER_REQ="${MAMBA_SLOTS_PER_REQ:-5}"
MAX_MAMBA_CACHE_SIZE="${MAX_MAMBA_CACHE_SIZE:-$((MAX_RUNNING_REQUESTS * MAMBA_SLOTS_PER_REQ))}"
MAMBA_SSM_DTYPE="${MAMBA_SSM_DTYPE:-float32}"
MAMBA_RADIX_CACHE_STRATEGY="${MAMBA_RADIX_CACHE_STRATEGY:-extra_buffer}"

QUANTIZATION="${QUANTIZATION:-modelopt_mixed}"
FP4_GEMM_BACKEND="${FP4_GEMM_BACKEND:-auto}"

OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/results/sglang_nvfp4_w4a4}"
LOG_FILE="${SCRIPT_DIR}/sglang_nvfp4_w4a4_eval_${TASK}_TP${TP_SIZE}_DP${DP_SIZE}_bs${BATCH_SIZE}.log"

mkdir -p "${OUTPUT_DIR}" "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}"

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "Error: no config.json under MODEL_PATH=${MODEL_PATH}" >&2
  exit 1
fi

if ! grep -Eq \
  '"quant_algo"[[:space:]]*:[[:space:]]*"MIXED_PRECISION"' \
  "${MODEL_PATH}/config.json"; then
  echo "Error: checkpoint is not ModelOpt MIXED_PRECISION: ${MODEL_PATH}" >&2
  exit 1
fi

IFS=',' read -r -a VISIBLE_GPUS <<< "${CUDA_VISIBLE_DEVICES}"
REQUIRED_GPUS=$((TP_SIZE * DP_SIZE))
if [[ "${#VISIBLE_GPUS[@]}" -lt "${REQUIRED_GPUS}" ]]; then
  echo "Error: ${REQUIRED_GPUS} GPUs required, but ${#VISIBLE_GPUS[@]} are visible." >&2
  exit 1
fi

MODEL_ARGS="pretrained=${MODEL_PATH},dtype=${DTYPE},trust_remote_code=${TRUST_REMOTE_CODE},tp_size=${TP_SIZE},dp_size=${DP_SIZE},mem_fraction_static=${MEM_FRACTION},max_model_len=${MAX_MODEL_LEN},kv_cache_dtype=${KV_CACHE_DTYPE},language_model_only=${LANGUAGE_MODEL_ONLY},chunked_prefill_size=${CHUNKED_PREFILL_SIZE},max_running_requests=${MAX_RUNNING_REQUESTS},max_mamba_cache_size=${MAX_MAMBA_CACHE_SIZE},mamba_ssm_dtype=${MAMBA_SSM_DTYPE},mamba_radix_cache_strategy=${MAMBA_RADIX_CACHE_STRATEGY},quantization=${QUANTIZATION},fp4_gemm_runner_backend=${FP4_GEMM_BACKEND}"

if command -v lm_eval >/dev/null 2>&1; then
  LM_EVAL=(lm_eval)
else
  LM_EVAL=(python3 -m lm_eval)
fi

echo "=================================================="
echo "Qwen3.8-27B offline NVFP4 W4A4 evaluation"
echo "=================================================="
echo "Model:       ${MODEL_PATH}"
echo "Quant:       ${QUANTIZATION}"
echo "FP4 mode:    W4A4"
echo "FP4 backend: ${FP4_GEMM_BACKEND}"
echo "Task:        ${TASK}"
echo "GPUs:        ${CUDA_VISIBLE_DEVICES}"
echo "TP / DP:     ${TP_SIZE} / ${DP_SIZE}"
echo "Batch size:  ${BATCH_SIZE}"
echo "Max len:     ${MAX_MODEL_LEN}"
echo "Model args:  ${MODEL_ARGS}"
echo "=================================================="

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

echo "Evaluation completed."
echo "Results: ${OUTPUT_DIR}"
echo "Log:     ${LOG_FILE}"
