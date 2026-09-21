#!/usr/bin/env bash
# Qwen3.5-397B-A17B BF16 accuracy evaluation with SGLang.
#
# The 397B BF16 checkpoint needs at least 8x B200/GB200-class GPUs. This
# script therefore defaults to TP=8, DP=1. It intentionally does not set
# language_model_only: current SGLang does not allow it for Qwen3.5.

set -euo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export HF_HOME="${HF_HOME:-/lustre/fsw/general_sa/xshang/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
HARNESS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${HARNESS_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

MODEL_PATH="${MODEL_PATH:-/lustre/fsw/general_sa/xshang/huggingface/Qwen3.5-397B-A17B}"
TASK="${TASK:-arc_easy}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_FEWSHOT="${NUM_FEWSHOT:-0}"
DTYPE="${DTYPE:-bfloat16}"
TP_SIZE="${TP_SIZE:-8}"
DP_SIZE="${DP_SIZE:-1}"
MEM_FRACTION="${MEM_FRACTION:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-True}"
LIMIT="${LIMIT:-}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
PAGE_SIZE="${PAGE_SIZE:-64}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-2048}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-16}"
MAMBA_SSM_DTYPE="${MAMBA_SSM_DTYPE:-bfloat16}"
MAMBA_RADIX_CACHE_STRATEGY="${MAMBA_RADIX_CACHE_STRATEGY:-extra_buffer}"
MAMBA_TRACK_INTERVAL="${MAMBA_TRACK_INTERVAL:-128}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-trtllm_mha}"
# flashinfer_cutedsl only supports the NVFP4 MoE paths in this SGLang build.
# BF16 uses the FlashInfer TensorRT-LLM MoE implementation instead.
MOE_RUNNER_BACKEND="${MOE_RUNNER_BACKEND:-flashinfer_trtllm}"
# This accuracy script uses DP=1. FlashInfer A2A requires DP=TP together with
# DP attention, so use the regular single-DP dispatch path.
MOE_A2A_BACKEND="${MOE_A2A_BACKEND:-none}"

OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/results/sglang_hf}"
LOG_FILE="${SCRIPT_DIR}/sglang_hf_eval_${TASK}_TP${TP_SIZE}_DP${DP_SIZE}_bs${BATCH_SIZE}.log"

mkdir -p "${OUTPUT_DIR}" "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}"

[[ -f "${MODEL_PATH}/config.json" ]] || {
  echo "Error: no config.json under MODEL_PATH=${MODEL_PATH}" >&2
  exit 1
}
[[ ! -f "${MODEL_PATH}/hf_quant_config.json" ]] || {
  echo "Error: this BF16 script received a quantized checkpoint: ${MODEL_PATH}" >&2
  exit 1
}
if (( TP_SIZE < 8 )) && [[ "${ALLOW_UNSAFE_BF16:-0}" != "1" ]]; then
  echo "Error: Qwen3.5-397B BF16 needs TP>=8 on B200/GB200-class GPUs." >&2
  echo "Set ALLOW_UNSAFE_BF16=1 only on hardware with sufficient HBM." >&2
  exit 1
fi

IFS=',' read -r -a _sglang_gpus <<< "${CUDA_VISIBLE_DEVICES}"
REQUIRED_GPUS=$((TP_SIZE * DP_SIZE))
(( ${#_sglang_gpus[@]} >= REQUIRED_GPUS )) || {
  echo "Error: TP_SIZE=${TP_SIZE}, DP_SIZE=${DP_SIZE} require ${REQUIRED_GPUS} visible GPUs." >&2
  exit 1
}

MODEL_ARGS="pretrained=${MODEL_PATH},dtype=${DTYPE},trust_remote_code=${TRUST_REMOTE_CODE},tp_size=${TP_SIZE},dp_size=${DP_SIZE},mem_fraction_static=${MEM_FRACTION},max_model_len=${MAX_MODEL_LEN},kv_cache_dtype=${KV_CACHE_DTYPE},page_size=${PAGE_SIZE},chunked_prefill_size=${CHUNKED_PREFILL_SIZE},max_running_requests=${MAX_RUNNING_REQUESTS},mamba_ssm_dtype=${MAMBA_SSM_DTYPE},mamba_radix_cache_strategy=${MAMBA_RADIX_CACHE_STRATEGY},mamba_track_interval=${MAMBA_TRACK_INTERVAL},attention_backend=${ATTENTION_BACKEND},moe_runner_backend=${MOE_RUNNER_BACKEND},moe_a2a_backend=${MOE_A2A_BACKEND}"

if command -v lm_eval >/dev/null 2>&1; then
  LM_EVAL=(lm_eval)
else
  LM_EVAL=(python3 -m lm_eval)
fi

echo "=================================================="
echo "Qwen3.5-397B-A17B BF16 evaluation"
echo "=================================================="
echo "Model:       ${MODEL_PATH}"
echo "Task:        ${TASK}"
echo "GPUs:        ${CUDA_VISIBLE_DEVICES}"
echo "TP / DP:     ${TP_SIZE} / ${DP_SIZE}"
echo "Batch size:  ${BATCH_SIZE}"
echo "Max len:     ${MAX_MODEL_LEN}"
echo "Dtype:       ${DTYPE}"
echo "KV cache:    ${KV_CACHE_DTYPE}"
echo "MoE:         ${MOE_RUNNER_BACKEND} + ${MOE_A2A_BACKEND}"
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
[[ -z "${LIMIT}" ]] || EVAL_CMD+=(--limit "${LIMIT}")
"${EVAL_CMD[@]}" 2>&1 | tee "${LOG_FILE}"

\n
