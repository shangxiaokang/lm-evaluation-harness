#!/bin/bash

# Run lm-eval locally with SGLang and a pre-quantized ModelOpt NVFP4 checkpoint.
# This script does not use Slurm, srun, containers, or torchrun.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
HARNESS_ROOT=$(dirname "$SCRIPT_DIR")
SGLANG_SRC=${SGLANG_SRC:-/lustre/fsw/portfolios/coreai/users/xshang/Quark/sglang}

if [ ! -f "${SGLANG_SRC}/python/sglang/srt/models/bailing_moe.py" ]; then
    echo "Error: SGLang source tree not found: ${SGLANG_SRC}"
    exit 1
fi
export PYTHONPATH="${SGLANG_SRC}/python:${HARNESS_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export HF_HOME=${HF_HOME:-/lustre/fsw/portfolios/coreai/projects/coreai_chef_numerics/users/xshang/huggingface}
export HF_ALLOW_CODE_EVAL=1
export HF_HUB_TRUST_REMOTE_CODE=1
export TOKENIZERS_PARALLELISM=false
export NCCL_TIMEOUT=3600000
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_FR_BUFFER_SIZE=1048576
export TORCHINDUCTOR_COMPILE_THREADS=${TORCHINDUCTOR_COMPILE_THREADS:-1}

RUN_ID=${RUN_ID:-local_$(hostname)_$$}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/tmp/triton_cache_${RUN_ID}}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-/tmp/torchinductor_cache_${RUN_ID}}
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

MODEL_BASE_PATH=${MODEL_BASE_PATH:-/lustre/fsw/portfolios/coreai/projects/coreai_chef_numerics/users/xshang/my-script/Ling/Ling-mini/checkpoint/Ling-mini-v2-SFT-NVFP4-PTQ-Last3BF16-hf}
# /lustre/fsw/portfolios/coreai/projects/coreai_chef_numerics/users/xshang/my-script/Ling/Ling-mini/checkpoint/Ling-mini-v2-SFT-NVFP4-V2-hf}
ITER_ID_INPUT=${1:-${ITER_ID:-1000}}

# MODEL_PATH takes precedence when explicitly provided. Otherwise construct an
# iteration directory such as iter_0001000 from ITER_ID=1000, 0001000, or
# iter_0001000.
if [ -n "${MODEL_PATH:-}" ]; then
    MODEL_PATH=${MODEL_PATH%/}
    ITER_DIR=$(basename "$MODEL_PATH")
else
    ITER_NUMBER=${ITER_ID_INPUT#iter_}
    if [[ ! "$ITER_NUMBER" =~ ^[0-9]+$ ]]; then
        echo "Error: ITER_ID must be numeric or use the iter_<number> format; got '${ITER_ID_INPUT}'."
        exit 1
    fi
    printf -v ITER_DIR "iter_%07d" "$((10#$ITER_NUMBER))"
    MODEL_PATH="${MODEL_BASE_PATH}/${ITER_DIR}"
fi

IFS=',' read -r -a VISIBLE_GPU_IDS <<<"$CUDA_VISIBLE_DEVICES"
TP_SIZE=${TP_SIZE:-1}
DP_SIZE=${DP_SIZE:-1}
EP_SIZE=${EP_SIZE:-1}
BATCH_SIZE=${BATCH:-4}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.8}
QUANTIZATION=${QUANTIZATION:-modelopt_fp4}
DTYPE=${DTYPE:-bfloat16}
MOE_RUNNER_BACKEND=${MOE_RUNNER_BACKEND:-flashinfer_cutlass}
ADD_BOS_TOKEN=${ADD_BOS_TOKEN:-False}

if [ "$EP_SIZE" -lt 1 ] || [ $((TP_SIZE % EP_SIZE)) -ne 0 ]; then
    echo "Error: EP_SIZE must be >= 1 and divide TP_SIZE (TP_SIZE=${TP_SIZE}, EP_SIZE=${EP_SIZE})."
    exit 1
fi

# Debug defaults. Set LIMIT=0 for a full evaluation.
TASK=${TASK:-arc_easy}
LIMIT=${LIMIT:-0}
RESULTS_DIR=${RESULTS_DIR:-${SCRIPT_DIR}/results/sglang_nvfp4_${ITER_DIR}_debug}
APPLY_CHAT_TEMPLATE=${APPLY_CHAT_TEMPLATE:-0}
AUTO_INSTALL_SGLANG=${AUTO_INSTALL_SGLANG:-0}

if [ ! -f "${MODEL_PATH}/config.json" ]; then
    echo "Error: HuggingFace config not found: ${MODEL_PATH}/config.json"
    exit 1
fi

if [ ! -f "${MODEL_PATH}/hf_quant_config.json" ]; then
    echo "Error: ModelOpt quantization config not found: ${MODEL_PATH}/hf_quant_config.json"
    exit 1
fi

python - "$MODEL_PATH" <<'PY'
import json
import pathlib
import sys

model_path = pathlib.Path(sys.argv[1])
with (model_path / "hf_quant_config.json").open() as config_file:
    quant_config = json.load(config_file)

quant_algo = quant_config.get("quantization", {}).get("quant_algo")
if quant_algo != "NVFP4":
    raise RuntimeError(
        f"Expected an NVFP4 checkpoint, but hf_quant_config.json reports {quant_algo!r}."
    )
print(f"ModelOpt quantization format: {quant_algo}")
PY

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "Error: nvidia-smi is unavailable; run this script on a GPU server."
    exit 1
fi

if ! python -c "import torch; assert torch.cuda.is_available()" >/dev/null 2>&1; then
    echo "Error: CUDA is not available in the current Python environment."
    exit 1
fi

if ! python -c "import sglang" >/dev/null 2>&1; then
    if [ "$AUTO_INSTALL_SGLANG" -ne 1 ]; then
        echo "Error: sglang is not installed."
        echo "Install a version with modelopt_fp4 support or rerun with AUTO_INSTALL_SGLANG=1."
        exit 1
    fi
    python -m pip install "sglang[all]"
fi

python -m pip install sacrebleu pytablewriter

python - <<'PY'
import inspect
import sglang
from sglang.srt.models import bailing_moe

print(f"SGLang version: {getattr(sglang, '__version__', 'unknown')}")
print(f"SGLang package: {inspect.getfile(sglang)}")
print(f"Bailing loader: {inspect.getfile(bailing_moe)}")
try:
    from sglang.srt.models.bailing_moe import BailingMoeV2ForCausalLM  # noqa: F401
except ImportError as exc:
    raise RuntimeError(
        "This SGLang version does not support BailingMoeV2ForCausalLM. "
        "Please upgrade SGLang."
    ) from exc
PY

mkdir -p "$RESULTS_DIR"
cd "$SCRIPT_DIR"

MODEL_ARGS="pretrained=${MODEL_PATH},tokenizer_path=${MODEL_PATH},tp_size=${TP_SIZE},dp_size=${DP_SIZE},ep_size=${EP_SIZE},dtype=${DTYPE},quantization=${QUANTIZATION},moe_runner_backend=${MOE_RUNNER_BACKEND},load_format=auto,max_model_len=${MAX_MODEL_LEN},mem_fraction_static=${MEM_FRACTION_STATIC},add_bos_token=${ADD_BOS_TOKEN},trust_remote_code=True"

EXTRA_EVAL_ARGS=()
if [ "$APPLY_CHAT_TEMPLATE" -eq 1 ]; then
    EXTRA_EVAL_ARGS+=(--apply_chat_template)
fi
if [ "$LIMIT" -gt 0 ]; then
    EXTRA_EVAL_ARGS+=(--limit "$LIMIT")
fi

echo "=== SGLang NVFP4 lm-eval local run ==="
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "Iteration: ${ITER_DIR}"
echo "Model path: ${MODEL_PATH}"
echo "Quantization: ${QUANTIZATION}"
echo "MoE runner backend: ${MOE_RUNNER_BACKEND}"
echo "Add BOS token: ${ADD_BOS_TOKEN}"
echo "Parallelism: TP=${TP_SIZE}, DP=${DP_SIZE}, EP=${EP_SIZE}"
echo "Model args: ${MODEL_ARGS}"
echo "Tasks: ${TASK}"
echo "Batch size: ${BATCH_SIZE}"
echo "Limit: ${LIMIT}"
echo "Results: ${RESULTS_DIR}"
echo "======================================="

# SGLang creates and manages its own GPU workers. Do not use torchrun here.
python -m lm_eval run \
    --model sglang \
    --model_args "$MODEL_ARGS" \
    --tasks "$TASK" \
    --batch_size "$BATCH_SIZE" \
    --num_fewshot 0 \
    --log_samples \
    --confirm_run_unsafe_code \
    --output_path "$RESULTS_DIR" \
    "${EXTRA_EVAL_ARGS[@]}"
