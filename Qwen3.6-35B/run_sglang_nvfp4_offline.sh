#!/bin/bash
# =============================================================================
# lm-eval SGLang: NVFP4 OFFLINE (serialized ModelOpt checkpoint)
# =============================================================================
# Diff vs run_sglang.sh (NVFP4 ONLINE):
#   Online:  BF16/FP8 weights + quantization=nvfp4_online
#            + moe_runner_backend=flashinfer_trtllm  (SM100/SM103)
#   Offline: pre-quantized NVFP4 ckpt + quantization=modelopt_mixed
#            + moe_runner_backend=flashinfer_trtllm  (GB200/SM100)
#
# Do not set quantization=nvfp4_online on this checkpoint.
#
# E.g. on a 2-GPU node:
#   bash run_sglang_nvfp4_offline.sh
#   LIMIT=16 bash run_sglang_nvfp4_offline.sh
#   TASK=mmlu_pro bash run_sglang_nvfp4_offline.sh
# =============================================================================

set -euo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export HF_HOME="${HF_HOME:-/lustre/fsw/general_sa/xshang/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
HARNESS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${HARNESS_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

MODEL_PATH="${MODEL_PATH:-/lustre/fsw/general_sa/xshang/huggingface/Qwen3.6-35B-A3B-NVFP4}"
TASK="${TASK:-mmlu_pro}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_FEWSHOT="${NUM_FEWSHOT:-0}"
DTYPE="${DTYPE:-auto}"
TP_SIZE="${TP_SIZE:-2}"
DP_SIZE="${DP_SIZE:-1}"
MEM_FRACTION="${MEM_FRACTION:-0.8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-True}"
LIMIT="${LIMIT:-}"
# SGLang: auto / fp8_e4m3 / fp8_e5m2. auto reads FP8 KV from ModelOpt config.
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
# GB200 uses the FlashInfer TensorRT-LLM MoE backend.
MOE_RUNNER_BACKEND="${MOE_RUNNER_BACKEND:-flashinfer_trtllm}"
# MIXED_PRECISION: attention FP8 + MoE W4A16_NVFP4
QUANTIZATION="${QUANTIZATION:-modelopt_mixed}"

OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/results/sglang_nvfp4_offline}"
LOG_FILE="${SCRIPT_DIR}/sglang_nvfp4_offline_eval_${TASK}_TP${TP_SIZE}_DP${DP_SIZE}_bs${BATCH_SIZE}.log"

mkdir -p "${OUTPUT_DIR}" "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}"

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "Error: no config.json under MODEL_PATH=${MODEL_PATH}" >&2
  echo "Offline NVFP4 needs the serialized Hugging Face checkpoint, e.g.:" >&2
  echo "  nvidia/Qwen3.6-35B-A3B-NVFP4" >&2
  exit 1
fi

if [[ ! -f "${MODEL_PATH}/hf_quant_config.json" ]]; then
  echo "Error: no hf_quant_config.json under MODEL_PATH=${MODEL_PATH}" >&2
  echo "This script loads a pre-quantized ModelOpt NVFP4 checkpoint." >&2
  echo "For online NVFP4 from BF16/FP8, use: bash ${SCRIPT_DIR}/run_sglang.sh" >&2
  exit 1
fi

MODEL_ARGS="pretrained=${MODEL_PATH},dtype=${DTYPE},trust_remote_code=${TRUST_REMOTE_CODE},tp_size=${TP_SIZE},dp_size=${DP_SIZE},mem_fraction_static=${MEM_FRACTION},max_model_len=${MAX_MODEL_LEN},kv_cache_dtype=${KV_CACHE_DTYPE},moe_runner_backend=${MOE_RUNNER_BACKEND}"

if [[ "${QUANTIZATION}" != "none" ]]; then
  MODEL_ARGS="${MODEL_ARGS},quantization=${QUANTIZATION}"
fi

if command -v lm_eval >/dev/null 2>&1; then
  LM_EVAL=(lm_eval)
else
  LM_EVAL=(python3 -m lm_eval)
fi

echo "=============================================="
echo "lm-eval SGLang backend (NVFP4 OFFLINE)"
echo "=============================================="
echo "Model path: ${MODEL_PATH}"
echo "Quant:      ${QUANTIZATION} (serialized ModelOpt)"
echo "MoE backend:${MOE_RUNNER_BACKEND}"
echo "Task:       ${TASK}"
echo "GPUs:       ${CUDA_VISIBLE_DEVICES}"
echo "TP / DP:    ${TP_SIZE} / ${DP_SIZE}"
echo "Batch size: ${BATCH_SIZE}"
echo "Max len:    ${MAX_MODEL_LEN}"
echo "Mem frac:   ${MEM_FRACTION}"
echo "Dtype:      ${DTYPE}"
echo "KV cache:   ${KV_CACHE_DTYPE}"
echo "Limit:      ${LIMIT:-none (full task)}"
echo "model_args: ${MODEL_ARGS}"
echo "HF_HOME:    ${HF_HOME}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Log file:   ${LOG_FILE}"
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
