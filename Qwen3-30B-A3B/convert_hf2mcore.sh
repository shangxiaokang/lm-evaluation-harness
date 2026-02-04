#!/bin/bash
# =============================================================================
# Convert Qwen3-30B-A3B (MoE) from HuggingFace to MCore format
# Using Megatron-Bridge
# =============================================================================

set -e

# ============== Configuration ==============
# HuggingFace model
HF_MODEL_ID="Qwen/Qwen3-30B-A3B"

# Local paths
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
HF_CACHE_DIR="/lustre/raplab/client/xshang/workspace/cache/huggingface"
HF_LOCAL_PATH="${HF_CACHE_DIR}/hub/models--Qwen--Qwen3-30B-A3B"
MCORE_OUTPUT_PATH="/lustre/raplab/client/xshang/workspace/huggingface/MCore/qwen3_30b_a3b"

# Megatron-Bridge path
MEGATRON_BRIDGE_PATH="/lustre/raplab/client/xshang/workspace/Megatron-Bridge"
MEGATRON_PATH=${MEGATRON_BRIDGE_PATH}/3rdparty/Megatron-LM
export PYTHONPATH=$MEGATRON_PATH:$MEGATRON_BRIDGE_PATH/src:$PYTHONPATH

# Environment
export HF_HOME="${HF_CACHE_DIR}"
export TRANSFORMERS_CACHE="${HF_CACHE_DIR}/hub"
export HF_DATASETS_CACHE="${HF_CACHE_DIR}/datasets"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# ============== Print Configuration ==============
echo "=============================================="
echo "Qwen3-30B-A3B HF to MCore Conversion"
echo "=============================================="
echo "HuggingFace Model: ${HF_MODEL_ID}"
echo "MCore Output: ${MCORE_OUTPUT_PATH}"
echo "HF Cache Dir: ${HF_CACHE_DIR}"
echo "CUDA Device: ${CUDA_VISIBLE_DEVICES}"
echo "=============================================="

# ============== Step 1: Download HuggingFace Model ==============
echo ""
echo "[Step 1/2] Downloading HuggingFace model..."
echo "=============================================="

# Create cache directory
mkdir -p "${HF_CACHE_DIR}"

# Download model using huggingface-cli
python3 << PYTHON_EOF
from huggingface_hub import snapshot_download
import os

print(f"Downloading {os.environ.get('HF_MODEL_ID', 'Qwen/Qwen3-30B-A3B')}...")
snapshot_download(
    repo_id="${HF_MODEL_ID}",
    cache_dir="${HF_CACHE_DIR}",
    resume_download=True,
    local_dir=None,  # Use default cache location
)
print("Download completed!")
PYTHON_EOF

echo "Model downloaded successfully!"

# ============== Step 2: Convert to MCore Format ==============
echo ""
echo "[Step 2/2] Converting to MCore format..."
echo "=============================================="

# Create output directory
mkdir -p "${MCORE_OUTPUT_PATH}"

# Run conversion using Megatron-Bridge

python3 ${MEGATRON_BRIDGE_PATH}/examples/conversion/convert_checkpoints.py import \
    --hf-model "${HF_MODEL_ID}" \
    --megatron-path "${MCORE_OUTPUT_PATH}" \
    --torch-dtype bfloat16 \
    --trust-remote-code

echo ""
echo "=============================================="
echo "Conversion completed!"
echo "=============================================="
echo "MCore checkpoint saved to: ${MCORE_OUTPUT_PATH}"
echo ""
echo "Qwen3-30B-A3B Model Architecture:"
echo "  - Type: MoE (Mixture of Experts)"
echo "  - Hidden Size: 2048"
echo "  - Num Layers: 48"
echo "  - Attention Heads: 32"
echo "  - KV Heads: 4"
echo "  - Head Dim: 128"
echo "  - Num Experts: 128"
echo "  - Top-K Experts: 8"
echo "  - Dense FFN Size: 6144"
echo "  - MoE Expert FFN Size: 768"
echo "  - Vocab Size: 151936"
echo "=============================================="
