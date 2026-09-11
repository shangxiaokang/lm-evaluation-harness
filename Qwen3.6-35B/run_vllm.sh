#!/bin/bash
# =============================================================================
# lm-eval vLLM backend: Qwen3.6-35B arc_easy on 8x 5K-pro
# =============================================================================
# E.g. local NVFP4:
#   bash run_vllm.sh
# E.g. Hugging Face BF16:
#   MODEL_PATH=Qwen/Qwen3.6-35B-A3B QUANTIZATION=none bash run_vllm.sh
# E.g. smoke test:
#   LIMIT=16 bash run_vllm.sh
# E.g. if EngineCore still loads dist-packages flash_attn:
#   python3 -m pip uninstall -y flash-attn
#   bash run_vllm.sh
# =============================================================================

set -euo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export HF_HOME="${HF_HOME:-/lustre/raplab/client/xshang/workspace/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/lustre/raplab/client/xshang/workspace/cache/vllm}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
HARNESS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# NGC flash_attn is ABI-incompatible with this PyTorch. Prefixed stub makes
# `flash_attn.ops.triton.rotary` a ModuleNotFoundError, which vLLM suppresses.
STUB_FLASH_ATTN="${STUB_FLASH_ATTN:-1}"
if [[ "${STUB_FLASH_ATTN}" == "1" ]]; then
  export PYTHONPATH="${SCRIPT_DIR}/shims:${HARNESS_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
else
  export PYTHONPATH="${HARNESS_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
fi

MODEL_PATH="${MODEL_PATH:-/lustre/raplab/client/xshang/workspace/huggingface/Qwen3.6-35B-A3B-NVFP4}"
TASK="${TASK:-mmlu_pro}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_FEWSHOT="${NUM_FEWSHOT:-0}"
DTYPE="${DTYPE:-auto}"
TP_SIZE="${TP_SIZE:-1}"
DP_SIZE="${DP_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-True}"
LIMIT="${LIMIT:-}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
# Qwen3.6 is a VL checkpoint. Vision RoPE imports the image's broken flash_attn.
# Skip the vision tower for text tasks such as arc_easy.
LANGUAGE_MODEL_ONLY="${LANGUAGE_MODEL_ONLY:-True}"

OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/results/vllm}"
LOG_FILE="${SCRIPT_DIR}/vllm_eval_${TASK}_TP${TP_SIZE}_DP${DP_SIZE}_bs${BATCH_SIZE}.log"

mkdir -p "${OUTPUT_DIR}" "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}" "${VLLM_CACHE_ROOT}"

is_local_ckpt=0
if [[ -d "${MODEL_PATH}" ]]; then
  is_local_ckpt=1
  if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
    echo "Error: no config.json under MODEL_PATH=${MODEL_PATH}" >&2
    exit 1
  fi
fi

# vLLM uses tensor_parallel_size / data_parallel_size / gpu_memory_utilization.
MODEL_ARGS="pretrained=${MODEL_PATH},dtype=${DTYPE},trust_remote_code=${TRUST_REMOTE_CODE},tensor_parallel_size=${TP_SIZE},data_parallel_size=${DP_SIZE},gpu_memory_utilization=${GPU_MEMORY_UTILIZATION},max_model_len=${MAX_MODEL_LEN},kv_cache_dtype=${KV_CACHE_DTYPE},language_model_only=${LANGUAGE_MODEL_ONLY}"

if [[ -n "${QUANTIZATION:-}" && "${QUANTIZATION}" != "none" ]]; then
  MODEL_ARGS="${MODEL_ARGS},quantization=${QUANTIZATION}"
elif [[ "${is_local_ckpt}" -eq 1 && -f "${MODEL_PATH}/hf_quant_config.json" ]]; then
  # NVIDIA ModelOpt NVFP4: vLLM flag is quantization=modelopt
  MODEL_ARGS="${MODEL_ARGS},quantization=modelopt"
fi

if command -v lm_eval >/dev/null 2>&1; then
  LM_EVAL=(lm_eval)
else
  LM_EVAL=(python3 -m lm_eval)
fi

echo "=============================================="
echo "lm-eval vLLM backend"
echo "=============================================="
echo "Model:      ${MODEL_PATH}"
echo "Task:       ${TASK}"
echo "GPUs:       ${CUDA_VISIBLE_DEVICES}"
echo "TP / DP:    ${TP_SIZE} / ${DP_SIZE}"
echo "Batch size: ${BATCH_SIZE}"
echo "Max len:    ${MAX_MODEL_LEN}"
echo "GPU util:   ${GPU_MEMORY_UTILIZATION}"
echo "Dtype:      ${DTYPE}"
echo "KV cache:   ${KV_CACHE_DTYPE}"
echo "Text only:  ${LANGUAGE_MODEL_ONLY}"
echo "FA stub:    ${STUB_FLASH_ATTN} (${SCRIPT_DIR}/shims/flash_attn)"
echo "Limit:      ${LIMIT:-none (full task)}"
echo "model_args: ${MODEL_ARGS}"
echo "HF_HOME:    ${HF_HOME}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Log file:   ${LOG_FILE}"
echo "=============================================="

cd "${HARNESS_DIR}"

EVAL_CMD=(
  "${LM_EVAL[@]}"
  --model vllm
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
