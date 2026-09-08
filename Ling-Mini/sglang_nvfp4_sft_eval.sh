#!/bin/bash

#SBATCH -p batch
#SBATCH -A coreai_chef_numerics
#SBATCH -J eval-sglang-nvfp4-ling-mini-v2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=3:10:00
#SBATCH --output=/lustre/fsw/portfolios/coreai/projects/coreai_chef_numerics/users/xshang/lm-evaluation-harness/Ling-Mini/slurm_logs/ling-mini-v2-eval-sglang-nvfp4-mse_%j.out

set -o pipefail

# Support both:
#   sbatch sglang_nvfp4_sft_eval.sh 1000
#   sbatch --export=ALL,ITER_ID=1000 sglang_nvfp4_sft_eval.sh
if [ "$#" -gt 0 ]; then
    export ITER_ID="$1"
fi

export CONTAINER_IMAGE=${CONTAINER_IMAGE:-/lustre/fsw/portfolios/coreai/projects/coreai_chef_numerics/users/xshang/sqsh/PyTorch-2606-PY3-NVFP4-EVAL-SGLANG.sqsh}

echo "=== Job Information ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node List: ${SLURM_JOB_NODELIST}"
echo "Container: ${CONTAINER_IMAGE}"
echo "Iteration input: ${ITER_ID:-1000}"
echo "======================="

srun --container-image="$CONTAINER_IMAGE" \
     --container-writable \
     --container-mounts=/home/xshang:/home/xshang,/lustre/fsw/portfolios/:/lustre/fsw/portfolios \
     --container-workdir=/lustre/fsw/portfolios/coreai/projects/coreai_chef_numerics/users/xshang/lm-evaluation-harness/Ling-Mini \
     --export=ALL \
     bash <<'EOF'
#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(pwd)
HARNESS_ROOT=$(dirname "$SCRIPT_DIR")
SGLANG_SRC=${SGLANG_SRC:-/lustre/fsw/portfolios/coreai/users/xshang/Quark/sglang}

if [ ! -f "${SGLANG_SRC}/python/sglang/srt/models/bailing_moe.py" ]; then
    echo "Error: SGLang source tree not found: ${SGLANG_SRC}"
    exit 1
fi

# Use the patched Bailing/ModelOpt implementation from the source tree.
export PYTHONPATH="${SGLANG_SRC}/python:${HARNESS_ROOT}:${PYTHONPATH:-}"
export HF_HOME=${HF_HOME:-/lustre/fsw/portfolios/coreai/projects/coreai_chef_numerics/users/xshang/huggingface}
export HF_ALLOW_CODE_EVAL=1
export HF_HUB_TRUST_REMOTE_CODE=1
export TOKENIZERS_PARALLELISM=false
export NCCL_TIMEOUT=3600000
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_FR_BUFFER_SIZE=1048576
export TORCHINDUCTOR_COMPILE_THREADS=${TORCHINDUCTOR_COMPILE_THREADS:-1}
export SGLANG_FLASHINFER_NVFP4_PER_TOKEN_ACTIVATION=0

LOCAL_CACHE_ROOT="/tmp/sglang_nvfp4_${SLURM_JOB_ID}_${SLURM_NODEID}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${LOCAL_CACHE_ROOT}/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${LOCAL_CACHE_ROOT}/torchinductor}"
export SGLANG_CACHE_DIR="${SGLANG_CACHE_DIR:-${LOCAL_CACHE_ROOT}/sglang}"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$SGLANG_CACHE_DIR"

MODEL_BASE_PATH=${MODEL_BASE_PATH:-/lustre/fsw/portfolios/coreai/projects/coreai_chef_numerics/users/xshang/my-script/Ling/Ling-mini/checkpoint/Ling-mini-v2-SFT-Real-NVFP4-PTQ-ExpertsOnly-MSE-Last3BF16-hf}
ITER_ID_INPUT=${ITER_ID:-1000}

if [ -n "${MODEL_PATH:-}" ]; then
    MODEL_PATH=${MODEL_PATH%/}
    ITER_DIR=$(basename "$MODEL_PATH")
else
    ITER_NUMBER=${ITER_ID_INPUT#iter_}
    if [[ ! "$ITER_NUMBER" =~ ^[0-9]+$ ]]; then
        echo "Error: ITER_ID must be numeric or use iter_<number>; got '${ITER_ID_INPUT}'."
        exit 1
    fi
    printf -v ITER_DIR "iter_%07d" "$((10#$ITER_NUMBER))"
    MODEL_PATH="${MODEL_BASE_PATH}/${ITER_DIR}"
fi

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

TASK=${TASK:-arc_easy,arc_challenge,winogrande,agieval,mmlu,humaneval,lambada_standard}
LIMIT=${LIMIT:-0}
RESULTS_DIR=${RESULTS_DIR:-${SCRIPT_DIR}/results/sglang_nvfp4_qkvonly_${ITER_DIR}}
APPLY_CHAT_TEMPLATE=${APPLY_CHAT_TEMPLATE:-0}

if [ "$EP_SIZE" -lt 1 ] || [ $((TP_SIZE % EP_SIZE)) -ne 0 ]; then
    echo "Error: EP_SIZE must be >= 1 and divide TP_SIZE (TP_SIZE=${TP_SIZE}, EP_SIZE=${EP_SIZE})."
    exit 1
fi

if [ ! -f "${MODEL_PATH}/config.json" ]; then
    echo "Error: HuggingFace config not found: ${MODEL_PATH}/config.json"
    exit 1
fi

if [ ! -f "${MODEL_PATH}/hf_quant_config.json" ]; then
    echo "Error: ModelOpt quantization config not found: ${MODEL_PATH}/hf_quant_config.json"
    exit 1
fi

python - "$MODEL_PATH" "$QUANTIZATION" <<'PY'
import inspect
import json
import pathlib
import sys

import sglang
from sglang.srt.models import bailing_moe

model_path = pathlib.Path(sys.argv[1])
quantization = sys.argv[2]
with (model_path / "hf_quant_config.json").open() as config_file:
    quant_config = json.load(config_file)

quant_algo = quant_config.get("quantization", {}).get("quant_algo")
if quant_algo != "NVFP4":
    raise RuntimeError(
        f"Expected an NVFP4 checkpoint, but hf_quant_config.json reports {quant_algo!r}."
    )
if quantization != "modelopt_fp4":
    raise RuntimeError(f"Expected quantization='modelopt_fp4', got {quantization!r}")

print(f"ModelOpt quantization format: {quant_algo}")
print(f"SGLang quantization backend: {quantization}")
print(f"SGLang version: {getattr(sglang, '__version__', 'unknown')}")
print(f"SGLang package: {inspect.getfile(sglang)}")
print(f"Bailing loader: {inspect.getfile(bailing_moe)}")
PY

if ! python -c "import torch; assert torch.cuda.is_available()" >/dev/null 2>&1; then
    echo "Error: CUDA is not available in the container."
    exit 1
fi

if ! python -c "import sacrebleu, pytablewriter" >/dev/null 2>&1; then
    python -m pip install sacrebleu pytablewriter
fi

pip install evaluate
pip install sacrebleu pytablewriter

mkdir -p "$RESULTS_DIR"

MODEL_ARGS="pretrained=${MODEL_PATH},tokenizer_path=${MODEL_PATH},tp_size=${TP_SIZE},dp_size=${DP_SIZE},ep_size=${EP_SIZE},dtype=${DTYPE},quantization=${QUANTIZATION},moe_runner_backend=${MOE_RUNNER_BACKEND},load_format=auto,max_model_len=${MAX_MODEL_LEN},mem_fraction_static=${MEM_FRACTION_STATIC},add_bos_token=${ADD_BOS_TOKEN},trust_remote_code=True"

EXTRA_EVAL_ARGS=()
if [ "$APPLY_CHAT_TEMPLATE" -eq 1 ]; then
    EXTRA_EVAL_ARGS+=(--apply_chat_template)
fi
if [ "$LIMIT" -gt 0 ]; then
    EXTRA_EVAL_ARGS+=(--limit "$LIMIT")
fi

echo "=== SGLang NVFP4 lm-eval Slurm run ==="
echo "Iteration: ${ITER_DIR}"
echo "Model path: ${MODEL_PATH}"
echo "Quantization: ${QUANTIZATION}"
echo "MoE runner backend: ${MOE_RUNNER_BACKEND}"
echo "Add BOS token: ${ADD_BOS_TOKEN}"
echo "Parallelism: TP=${TP_SIZE}, DP=${DP_SIZE}, EP=${EP_SIZE}"
echo "Tasks: ${TASK}"
echo "Batch size: ${BATCH_SIZE}"
echo "Limit: ${LIMIT} (0 means full dataset)"
echo "Results: ${RESULTS_DIR}"
echo "Model args: ${MODEL_ARGS}"
echo "========================================="

# SGLang creates and manages its own GPU workers. Do not use torchrun.
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
EOF

status=$?
if [ "$status" -ne 0 ]; then
    echo "Evaluation failed with status ${status}."
else
    echo "Evaluation completed successfully."
fi
exit "$status"
