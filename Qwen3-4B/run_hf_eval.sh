#!/bin/bash
# =============================================================================
# lm-eval HuggingFace Backend Evaluation Script for Qwen3-4B
# Supports Data Parallelism with multiple GPUs
# =============================================================================
export MASTER_ADDR=localhost

# Default values (can be overridden by environment variables)
TASK=${TASK:-"gsm8k"}
BATCH_SIZE=${BATCH_SIZE:-1}
NUM_GPUS=${NUM_GPUS:-8}  # Number of GPUs for data parallelism

# Model configuration
MODEL_NAME="Qwen/Qwen3-4B"
DTYPE="bfloat16"

# Output directory
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
OUTPUT_DIR="${SCRIPT_DIR}/results/hf"
LOG_FILE="${SCRIPT_DIR}/hf_eval_${TASK}_${BATCH_SIZE}_${NUM_GPUS}gpu.log"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

echo "=============================================="
echo "lm-eval HuggingFace Backend Evaluation"
echo "=============================================="
echo "Model: ${MODEL_NAME}"
echo "Dtype: ${DTYPE}"
echo "Task: ${TASK}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Num GPUs: ${NUM_GPUS}"
echo "Output Dir: ${OUTPUT_DIR}"
echo "Log File: ${LOG_FILE}"
echo "=============================================="

# Run evaluation
if [ "$NUM_GPUS" -gt 1 ]; then
    # Data Parallel mode with accelerate
    echo "Running in Data Parallel mode with ${NUM_GPUS} GPUs..."
    accelerate launch --num_processes ${NUM_GPUS} \
        -m lm_eval --model hf \
        --model_args pretrained=${MODEL_NAME},dtype=${DTYPE},trust_remote_code=True \
        --tasks ${TASK} \
        --batch_size ${BATCH_SIZE} \
        --num_fewshot 0 \
        --output_path ${OUTPUT_DIR} \
        2>&1 | tee ${LOG_FILE}
else
    # Single GPU mode
    echo "Running in Single GPU mode..."
    export CUDA_VISIBLE_DEVICES=0
    lm_eval --model hf \
        --model_args pretrained=${MODEL_NAME},dtype=${DTYPE},trust_remote_code=True \
        --tasks ${TASK} \
        --batch_size ${BATCH_SIZE} \
        --num_fewshot 0 \
        --output_path ${OUTPUT_DIR} \
        2>&1 | tee ${LOG_FILE}
fi

echo "=============================================="
echo "Evaluation completed!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "=============================================="
