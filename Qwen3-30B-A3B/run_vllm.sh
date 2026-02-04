#!/bin/bash
# =============================================================================
# lm-eval vLLM Backend Evaluation Script for Qwen3-30B-A3B (MoE)
# =============================================================================

# Environment setup
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Default values (can be overridden by environment variables)
TASK=${TASK:-"arc_easy"}
BATCH_SIZE=${BATCH_SIZE:-16}

# Model configuration
MODEL_NAME=${MODEL_NAME:-"Qwen/Qwen3-30B-A3B"}
DTYPE=${DTYPE:-"bfloat16"}

# vLLM specific configuration
# For MoE models, tensor_parallel_size distributes experts across GPUs
# NOTE: Qwen3-30B-A3B requires TP>=8 to fit in GPU memory!
#       TP=1 will cause OOM (std::bad_alloc)
TENSOR_PARALLEL_SIZE=${TP_SIZE:-1}
DATA_PARALLEL_SIZE=${DP_SIZE:-1}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.90}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}

# Calculate total GPUs needed
TOTAL_GPUS=$((TENSOR_PARALLEL_SIZE * DATA_PARALLEL_SIZE))

# Output directory
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
OUTPUT_DIR="${SCRIPT_DIR}/results/vllm"
LOG_FILE="${SCRIPT_DIR}/vllm_eval_${TASK}_TP${TENSOR_PARALLEL_SIZE}_DP${DATA_PARALLEL_SIZE}.log"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

echo "=============================================="
echo "lm-eval vLLM Backend Evaluation (MoE)"
echo "=============================================="
echo "Model: ${MODEL_NAME}"
echo "Task: ${TASK}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Dtype: ${DTYPE}"
echo "=============================================="
echo "vLLM Configuration:"
echo "  Tensor Parallel Size: ${TENSOR_PARALLEL_SIZE}"
echo "  Data Parallel Size: ${DATA_PARALLEL_SIZE}"
echo "  Total GPUs: ${TOTAL_GPUS}"
echo "  GPU Memory Utilization: ${GPU_MEMORY_UTILIZATION}"
echo "  Max Model Length: ${MAX_MODEL_LEN}"
echo "=============================================="
echo "Output Dir: ${OUTPUT_DIR}"
echo "Log File: ${LOG_FILE}"
echo "=============================================="

# Run evaluation with vLLM backend
# For MoE models like Qwen3-30B-A3B:
# - tensor_parallel_size: distributes model (including experts) across GPUs
# - data_parallel_size: number of model replicas for parallel data processing
lm_eval --model vllm \
    --model_args pretrained=${MODEL_NAME},dtype=${DTYPE},trust_remote_code=True,gpu_memory_utilization=${GPU_MEMORY_UTILIZATION},tensor_parallel_size=${TENSOR_PARALLEL_SIZE},data_parallel_size=${DATA_PARALLEL_SIZE},max_model_len=${MAX_MODEL_LEN} \
    --tasks ${TASK} \
    --batch_size ${BATCH_SIZE} \
    --num_fewshot 0 \
    --output_path ${OUTPUT_DIR} \
    2>&1 | tee ${LOG_FILE}

echo "=============================================="
echo "Evaluation completed!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "=============================================="
