#!/bin/bash
# =============================================================================
# lm-eval vLLM Backend Evaluation Script for Qwen3-4B
# Supports Tensor Parallelism (TP) and Data Parallelism (DP)
# =============================================================================

# Cache directories (avoid home directory quota issues)
export VLLM_CACHE_ROOT=/lustre/raplab/client/xshang/workspace/cache/vllm
export HF_HOME=/lustre/raplab/client/xshang/workspace/huggingface
export TRANSFORMERS_CACHE=/lustre/raplab/client/xshang/workspace/huggingface/hub
export HF_DATASETS_CACHE=/lustre/raplab/client/xshang/workspace/huggingface/datasets

# Create cache directories
mkdir -p ${VLLM_CACHE_ROOT}
mkdir -p ${HF_HOME}
mkdir -p ${TRANSFORMERS_CACHE}
mkdir -p ${HF_DATASETS_CACHE}

# Default values (can be overridden by environment variables)
TASK=${TASK:-"triviaqa"}
BATCH_SIZE=${BATCH_SIZE:-1}

# Model configuration
MODEL_NAME="Qwen/Qwen3-4B"
DTYPE="bfloat16"

# vLLM specific configuration
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}  # TP: split model across GPUs
DATA_PARALLEL_SIZE=${DATA_PARALLEL_SIZE:-1}      # DP: replicate model for parallel data processing

# Output directory
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
OUTPUT_DIR="${SCRIPT_DIR}/results/vllm"
LOG_FILE="${SCRIPT_DIR}/vllm_eval_${TASK}_bs${BATCH_SIZE}_tp${TENSOR_PARALLEL_SIZE}_dp${DATA_PARALLEL_SIZE}.log"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

echo "=============================================="
echo "lm-eval vLLM Backend Evaluation"
echo "=============================================="
echo "Model: ${MODEL_NAME}"
echo "Dtype: ${DTYPE}"
echo "Task: ${TASK}"
echo "Batch Size: ${BATCH_SIZE}"
echo "GPU Memory Utilization: ${GPU_MEMORY_UTILIZATION}"
echo "Tensor Parallel Size: ${TENSOR_PARALLEL_SIZE}"
echo "Data Parallel Size: ${DATA_PARALLEL_SIZE}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "Output Dir: ${OUTPUT_DIR}"
echo "Log File: ${LOG_FILE}"
echo "=============================================="
echo "Cache Directories:"
echo "  VLLM_CACHE_ROOT: ${VLLM_CACHE_ROOT}"
echo "  HF_HOME: ${HF_HOME}"
echo "=============================================="

# Run evaluation with vLLM backend
lm_eval --model vllm \
    --model_args pretrained=${MODEL_NAME},dtype=${DTYPE},trust_remote_code=True,gpu_memory_utilization=${GPU_MEMORY_UTILIZATION},tensor_parallel_size=${TENSOR_PARALLEL_SIZE},data_parallel_size=${DATA_PARALLEL_SIZE} \
    --tasks ${TASK} \
    --batch_size ${BATCH_SIZE} \
    --num_fewshot 0 \
    --output_path ${OUTPUT_DIR} \
    2>&1 | tee ${LOG_FILE}

echo "=============================================="
echo "Evaluation completed!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "=============================================="
