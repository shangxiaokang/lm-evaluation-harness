#!/bin/bash
# =============================================================================
# lm-eval SGLang backend: Hugging Face Qwen/Qwen3.6-35B-A3B (BF16)
# =============================================================================
# Diff vs run_sglang.sh (local NVFP4):
#   1. pretrained=Qwen/Qwen3.6-35B-A3B  (Hub repo id, not a local directory)
#   2. Drop the local config.json check
#   3. Do not set quantization=modelopt_mixed / kv_cache_dtype=fp8
#   4. Keep HF_HOME so Hub shards cache on Lustre
#
# Qwen3.6 is hybrid (Mamba/GDN + MoE). BF16 35B-A3B is ~70 GB and does not
# leave a mamba state pool on one 5K-pro (max_mamba_cache_size=1, 5 slots/req
# → max_num_reqs=0). Use TP=8. Do not set CUDA_VISIBLE_DEVICES=0.
#
# E.g. on the node:
#   bash run_sglang_hf.sh
#   TASK=arc_challenge bash run_sglang_hf.sh
#   LIMIT=16 bash run_sglang_hf.sh
#   TP_SIZE=8 MEM_FRACTION=0.9 bash run_sglang_hf.sh
# =============================================================================

set -euo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export HF_HOME="${HF_HOME:-/lustre/raplab/client/xshang/workspace/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
HARNESS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${HARNESS_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

MODEL_ID="${MODEL_ID:-Qwen/Qwen3.6-35B-A3B}"
TASK="${TASK:-mmlu_pro}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_FEWSHOT="${NUM_FEWSHOT:-0}"
DTYPE="${DTYPE:-bfloat16}"
TP_SIZE="${TP_SIZE:-2}"
DP_SIZE="${DP_SIZE:-1}"
MEM_FRACTION="${MEM_FRACTION:-0.9}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-True}"
LIMIT="${LIMIT:-}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-32}"
# Hybrid GDN: each request uses several mamba slots (often 5). Pin the pool.
MAMBA_SLOTS_PER_REQ="${MAMBA_SLOTS_PER_REQ:-5}"
MAX_MAMBA_CACHE_SIZE="${MAX_MAMBA_CACHE_SIZE:-$((MAX_RUNNING_REQUESTS * MAMBA_SLOTS_PER_REQ))}"
MAMBA_SSM_DTYPE="${MAMBA_SSM_DTYPE:-bfloat16}"
MAMBA_FULL_MEMORY_RATIO="${MAMBA_FULL_MEMORY_RATIO:-0.9}"

OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/results/sglang_hf}"
LOG_FILE="${SCRIPT_DIR}/sglang_hf_eval_${TASK}_TP${TP_SIZE}_DP${DP_SIZE}_bs${BATCH_SIZE}.log"

mkdir -p "${OUTPUT_DIR}" "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}"

IFS=',' read -r -a _sglang_gpus <<< "${CUDA_VISIBLE_DEVICES}"
if [[ "${#_sglang_gpus[@]}" -lt "${TP_SIZE}" ]]; then
  echo "Error: BF16 Qwen3.6-35B-A3B needs TP_SIZE=${TP_SIZE} visible GPUs," >&2
  echo "       but CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} has ${#_sglang_gpus[@]}." >&2
  echo "Single-GPU leftover VRAM cannot size the hybrid mamba pool (max_num_reqs=0)." >&2
  echo "E.g. CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TP_SIZE=8 TASK=${TASK} bash run_sglang_hf.sh" >&2
  exit 1
fi

MODEL_ARGS="pretrained=${MODEL_ID},dtype=${DTYPE},trust_remote_code=${TRUST_REMOTE_CODE},tp_size=${TP_SIZE},dp_size=${DP_SIZE},mem_fraction_static=${MEM_FRACTION},max_model_len=${MAX_MODEL_LEN},max_running_requests=${MAX_RUNNING_REQUESTS},max_mamba_cache_size=${MAX_MAMBA_CACHE_SIZE},mamba_ssm_dtype=${MAMBA_SSM_DTYPE},mamba_full_memory_ratio=${MAMBA_FULL_MEMORY_RATIO}"

if command -v lm_eval >/dev/null 2>&1; then
  LM_EVAL=(lm_eval)
else
  LM_EVAL=(python3 -m lm_eval)
fi

echo "=============================================="
echo "lm-eval SGLang backend (Hugging Face Hub)"
echo "=============================================="
echo "Model id:   ${MODEL_ID}"
echo "Task:       ${TASK}"
echo "GPUs:       ${CUDA_VISIBLE_DEVICES}"
echo "TP / DP:    ${TP_SIZE} / ${DP_SIZE}"
echo "Batch size: ${BATCH_SIZE}"
echo "Max len:    ${MAX_MODEL_LEN}"
echo "Mem frac:   ${MEM_FRACTION}"
echo "Max reqs:   ${MAX_RUNNING_REQUESTS}"
echo "Mamba cache:${MAX_MAMBA_CACHE_SIZE} (ssm=${MAMBA_SSM_DTYPE}, ratio=${MAMBA_FULL_MEMORY_RATIO})"
echo "Dtype:      ${DTYPE}"
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
