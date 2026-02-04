#!/bin/bash
# =============================================================================
# lm-eval Megatron-LM Backend Evaluation Script for Qwen3-4B
# =============================================================================
export MASTER_ADDR=localhost
export MEGATRON_PATH=/home/xshang/workspace/Megatron-LM
# Disable torch.compile/inductor (Triton version mismatch)
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Default values (can be overridden by environment variables)
TASK=${TASK:-"gsm8k,triviaqa"}
BATCH_SIZE=${BATCH_SIZE:-16}

# Model configuration
MODEL_NAME="Qwen/Qwen3-4B"
DTYPE="bfloat16"

# Megatron-LM specific configuration
CHECKPOINT_PATH=${CHECKPOINT_PATH:-"/lustre/raplab/client/xshang/workspace/huggingface/MCore/qwen3_4b"}
TOKENIZER_MODEL=${TOKENIZER_MODEL:-"${MODEL_NAME}"}
TP_SIZE=${TP_SIZE:-1}
PP_SIZE=${PP_SIZE:-1}
DEVICES=${DEVICES:-8}

# Qwen3-4B model architecture parameters (from run_config.yaml)
NUM_LAYERS=36
HIDDEN_SIZE=2560
NUM_ATTENTION_HEADS=32
NUM_KEY_VALUE_HEADS=8
FFN_HIDDEN_SIZE=9728
MAX_POSITION_EMBEDDINGS=40960
HEAD_DIM=128

# Extra args for Megatron-LM (model architecture)
# Based on checkpoint's run_config.yaml
EXTRA_ARGS="--num-layers ${NUM_LAYERS} \
--hidden-size ${HIDDEN_SIZE} \
--num-attention-heads ${NUM_ATTENTION_HEADS} \
--group-query-attention \
--num-query-groups ${NUM_KEY_VALUE_HEADS} \
--kv-channels ${HEAD_DIM} \
--ffn-hidden-size ${FFN_HIDDEN_SIZE} \
--max-position-embeddings ${MAX_POSITION_EMBEDDINGS} \
--position-embedding-type rope \
--use-rotary-position-embeddings \
--rotary-base 1000000 \
--rotary-percent 1.0 \
--normalization RMSNorm \
--norm-epsilon 1e-6 \
--swiglu \
--disable-bias-linear \
--no-bias-swiglu-fusion \
--no-position-embedding \
--qk-layernorm \
--no-rope-fusion"

# Output directory
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
OUTPUT_DIR="${SCRIPT_DIR}/results/megatron_lm"
LOG_FILE="${SCRIPT_DIR}/megatron_lm_eval_${TASK}_BS${BATCH_SIZE}_TP${TP_SIZE}_DP8.log"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

echo "=============================================="
echo "lm-eval Megatron-LM Backend Evaluation"
echo "=============================================="
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Tokenizer: ${TOKENIZER_MODEL}"
echo "Task: ${TASK}"
echo "Batch Size: ${BATCH_SIZE}"
echo "TP Size: ${TP_SIZE}"
echo "PP Size: ${PP_SIZE}"
echo "Devices: ${DEVICES}"
echo "Output Dir: ${OUTPUT_DIR}"
echo "Log File: ${LOG_FILE}"
echo "=============================================="
echo "Model Architecture:"
echo "  Num Layers: ${NUM_LAYERS}"
echo "  Hidden Size: ${HIDDEN_SIZE}"
echo "  Attention Heads: ${NUM_ATTENTION_HEADS}"
echo "  KV Heads: ${NUM_KEY_VALUE_HEADS}"
echo "  Head Dim (kv-channels): ${HEAD_DIM}"
echo "  FFN Hidden Size: ${FFN_HIDDEN_SIZE}"
echo "  QK LayerNorm: Yes (Qwen3 feature)"
echo "  RoPE Fusion: Disabled"
echo "=============================================="

# Run evaluation with Megatron-LM backend
torchrun --nproc_per_node=${DEVICES} --master_port=${MASTER_PORT} \
    -m lm_eval --model megatron_lm \
    --model_args load=${CHECKPOINT_PATH},tokenizer_model=${TOKENIZER_MODEL},micro_batch_size=${BATCH_SIZE},tensor_model_parallel_size=${TP_SIZE},pipeline_model_parallel_size=${PP_SIZE},devices=${DEVICES},extra_args="${EXTRA_ARGS}" \
    --tasks ${TASK} \
    --batch_size ${BATCH_SIZE} \
    --num_fewshot 0 \
    --output_path ${OUTPUT_DIR} \
    2>&1 | tee ${LOG_FILE}

echo "=============================================="
echo "Evaluation completed!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "=============================================="
