#!/bin/bash

# Run lm-eval with the SGLang backend directly on a GPU server.
# This script does not use Slurm, srun, containers, or torchrun.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
HARNESS_ROOT=$(dirname "$SCRIPT_DIR")

export PYTHONPATH="${HARNESS_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export HF_HOME=${HF_HOME:-/lustre/fsw/portfolios/coreai/projects/coreai_chef_numerics/users/xshang/huggingface}
export HF_ALLOW_CODE_EVAL=1
export HF_HUB_TRUST_REMOTE_CODE=1
export TOKENIZERS_PARALLELISM=false
export NCCL_TIMEOUT=3600000
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_FR_BUFFER_SIZE=1048576
# Avoid Python resource_tracker races from Inductor's compile worker pool at exit.
export TORCHINDUCTOR_COMPILE_THREADS=${TORCHINDUCTOR_COMPILE_THREADS:-1}

RUN_ID=${RUN_ID:-local_$(hostname)_$$}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/tmp/triton_cache_${RUN_ID}}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-/tmp/torchinductor_cache_${RUN_ID}}
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

MODEL_BASE_PATH=${MODEL_BASE_PATH:-/lustre/fsw/portfolios/coreai/projects/coreai_chef_numerics/users/xshang/my-script/Ling/Ling-mini/checkpoint/Ling-mini-v2-SFT-BF16-hf}
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
TP_SIZE=${TP_SIZE:-${#VISIBLE_GPU_IDS[@]}}
DP_SIZE=${DP_SIZE:-1}
EP_SIZE=${EP_SIZE:-1}
BATCH_SIZE=${BATCH:-4}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.8}

if [ "$EP_SIZE" -lt 1 ] || [ $((TP_SIZE % EP_SIZE)) -ne 0 ]; then
    echo "Error: EP_SIZE must be >= 1 and divide TP_SIZE (TP_SIZE=${TP_SIZE}, EP_SIZE=${EP_SIZE})."
    exit 1
fi

# Debug defaults: one task and 20 examples. Override TASK and set LIMIT=0
# after the environment has been validated.
TASK=${TASK:-arc_easy}
LIMIT=${LIMIT:-0}
RESULTS_DIR=${RESULTS_DIR:-${SCRIPT_DIR}/results/sglang_bf16_${ITER_DIR}_debug}
APPLY_CHAT_TEMPLATE=${APPLY_CHAT_TEMPLATE:-0}
AUTO_INSTALL_SGLANG=${AUTO_INSTALL_SGLANG:-0}

if [ ! -f "${MODEL_PATH}/config.json" ]; then
    echo "Error: HuggingFace config not found: ${MODEL_PATH}/config.json"
    exit 1
fi

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
        echo "Install it in the current GPU environment or rerun with AUTO_INSTALL_SGLANG=1."
        exit 1
    fi
    python -m pip install "sglang[all]"
fi

pip install sacrebleu pytablewriter

python - <<'PY'
import sglang

print(f"SGLang version: {getattr(sglang, '__version__', 'unknown')}")
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

MODEL_ARGS="pretrained=${MODEL_PATH},tokenizer_path=${MODEL_PATH},tp_size=${TP_SIZE},dp_size=${DP_SIZE},ep_size=${EP_SIZE},dtype=bfloat16,max_model_len=${MAX_MODEL_LEN},mem_fraction_static=${MEM_FRACTION_STATIC},trust_remote_code=True"

EXTRA_EVAL_ARGS=()
if [ "$APPLY_CHAT_TEMPLATE" -eq 1 ]; then
    EXTRA_EVAL_ARGS+=(--apply_chat_template)
fi
if [ "$LIMIT" -gt 0 ]; then
    EXTRA_EVAL_ARGS+=(--limit "$LIMIT")
fi

echo "=== SGLang lm-eval local run ==="
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "Model path: ${MODEL_PATH}"
echo "Parallelism: TP=${TP_SIZE}, DP=${DP_SIZE}, EP=${EP_SIZE}"
echo "Model args: ${MODEL_ARGS}"
echo "Tasks: ${TASK}"
echo "Batch size: ${BATCH_SIZE}"
echo "Limit: ${LIMIT}"
echo "Results: ${RESULTS_DIR}"
echo "================================="

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
