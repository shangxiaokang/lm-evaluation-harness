# git clone git@github.com:NVIDIA-NeMo/Megatron-Bridge.git

MEGATRON_BRIDGE_PATH=/Your/Megatron-Bridge
MEGATRON_PATH=${MEGATRON_BRIDGE_PATH}/3rdparty/Megatron-LM

export PYTHONPATH=$MEGATRON_PATH:${MEGATRON_BRIDGE_PATH}/src:$PYTHONPATH
export HF_HOME=/Your/huggingface

python ${MEGATRON_BRIDGE_PATH}/examples/conversion/convert_checkpoints.py import \
  --hf-model Qwen/Qwen3-4B \
  --megatron-path ./checkpoints/qwen3_4b \
  --torch-dtype bfloat16
