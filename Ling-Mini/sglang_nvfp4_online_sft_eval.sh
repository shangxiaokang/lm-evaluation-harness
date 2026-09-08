#!/bin/bash

#SBATCH -p batch
#SBATCH -A coreai_chef_numerics
#SBATCH -J eval-sglang-nvfp4-online-ling-mini-v2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=3:10:00
#SBATCH --output=/lustre/fsw/portfolios/coreai/projects/coreai_chef_numerics/users/xshang/lm-evaluation-harness/Ling-Mini/slurm_logs/ling-mini-v2-eval-sglang-nvfp4-online_%j.out

set -o pipefail

# Support both:
#   sbatch sglang_nvfp4_online_sft_eval.sh 1000
#   sbatch --export=ALL,ITER_ID=1000 sglang_nvfp4_online_sft_eval.sh
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
# Match SGLang's validated nvfp4_online accuracy configuration. 4-over-6
# chooses between FP4 maxima 4 and 6 per block using MSE and uses 256 as the
# tensor-scale E4M3 bound so blocks selecting 4 retain sufficient headroom.
export FLASHINFER_NVFP4_4OVER6=${FLASHINFER_NVFP4_4OVER6:-1}
export FLASHINFER_NVFP4_4OVER6_ERR_MODE=${FLASHINFER_NVFP4_4OVER6_ERR_MODE:-MSE}
export FLASHINFER_NVFP4_4OVER6_ERR_USE_FAST_MATH=${FLASHINFER_NVFP4_4OVER6_ERR_USE_FAST_MATH:-1}
export FLASHINFER_NVFP4_4OVER6_E4M3_USE_256=${FLASHINFER_NVFP4_4OVER6_E4M3_USE_256:-1}
# Full MMLU/AGIEval runs request prompt logprobs for more than 100k inputs.
# Keep the vocabulary projection chunks small enough that one scheduler
# forward does not trip the hard watchdog.
export SGLANG_ENABLE_LOGPROB_CHUNK=${SGLANG_ENABLE_LOGPROB_CHUNK:-1}
export SGLANG_LOGPROB_CHUNK_SIZE=${SGLANG_LOGPROB_CHUNK_SIZE:-256}

LOCAL_CACHE_ROOT="/tmp/sglang_nvfp4_${SLURM_JOB_ID}_${SLURM_NODEID}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${LOCAL_CACHE_ROOT}/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${LOCAL_CACHE_ROOT}/torchinductor}"
export SGLANG_CACHE_DIR="${SGLANG_CACHE_DIR:-${LOCAL_CACHE_ROOT}/sglang}"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$SGLANG_CACHE_DIR"

MODEL_BASE_PATH=${MODEL_BASE_PATH:-/lustre/fsw/portfolios/coreai/projects/coreai_chef_numerics/users/xshang/my-script/Ling/Ling-mini/checkpoint/Ling-mini-v2-SFT-BF16-hf}
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
WATCHDOG_TIMEOUT=${WATCHDOG_TIMEOUT:-1800}
QUANTIZATION=${QUANTIZATION:-nvfp4_online}
DTYPE=${DTYPE:-bfloat16}
MOE_RUNNER_BACKEND=${MOE_RUNNER_BACKEND:-flashinfer_trtllm}
ADD_BOS_TOKEN=${ADD_BOS_TOKEN:-False}
# Keep the final three routed-MoE layers in BF16. Set this variable to another
# comma-separated layer list when testing a different mixed-precision policy.
export SGLANG_FP4_IGNORED_LAYERS=${SGLANG_FP4_IGNORED_LAYERS:-model.layers.17,model.layers.18,model.layers.19}

TASK=${TASK:-arc_easy,arc_challenge,winogrande,agieval,mmlu,humaneval,lambada_standard}
RESULTS_DIR=${RESULTS_DIR:-${SCRIPT_DIR}/results/sglang_nvfp4_online_${ITER_DIR}}
APPLY_CHAT_TEMPLATE=${APPLY_CHAT_TEMPLATE:-0}

if [ "$EP_SIZE" -lt 1 ] || [ $((TP_SIZE % EP_SIZE)) -ne 0 ]; then
    echo "Error: EP_SIZE must be >= 1 and divide TP_SIZE (TP_SIZE=${TP_SIZE}, EP_SIZE=${EP_SIZE})."
    exit 1
fi

if [ ! -f "${MODEL_PATH}/config.json" ]; then
    echo "Error: HuggingFace config not found: ${MODEL_PATH}/config.json"
    exit 1
fi

python - "$MODEL_PATH" "$QUANTIZATION" <<'PY'
import inspect
import pathlib
import sys

import sglang
from sglang.srt.models import bailing_moe

model_path = pathlib.Path(sys.argv[1])
quantization = sys.argv[2]
if quantization != "nvfp4_online":
    raise RuntimeError(f"Expected quantization='nvfp4_online', got {quantization!r}")

print(f"Online quantization: {quantization}")
print(f"BF16 source checkpoint: {model_path}")
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

MODEL_ARGS="pretrained=${MODEL_PATH},tokenizer_path=${MODEL_PATH},tp_size=${TP_SIZE},dp_size=${DP_SIZE},ep_size=${EP_SIZE},dtype=${DTYPE},quantization=${QUANTIZATION},moe_runner_backend=${MOE_RUNNER_BACKEND},load_format=auto,max_model_len=${MAX_MODEL_LEN},mem_fraction_static=${MEM_FRACTION_STATIC},watchdog_timeout=${WATCHDOG_TIMEOUT},add_bos_token=${ADD_BOS_TOKEN},trust_remote_code=True"

EXTRA_EVAL_ARGS=()
if [ "$APPLY_CHAT_TEMPLATE" -eq 1 ]; then
    EXTRA_EVAL_ARGS+=(--apply_chat_template)
fi
LIMIT=0
if [ "$LIMIT" -gt 0 ]; then
    EXTRA_EVAL_ARGS+=(--limit "$LIMIT")
fi

echo "=== SGLang online NVFP4 lm-eval Slurm run ==="
echo "Iteration: ${ITER_DIR}"
echo "Model path: ${MODEL_PATH}"
echo "Quantization: ${QUANTIZATION}"
echo "MoE runner backend: ${MOE_RUNNER_BACKEND}"
echo "FP4 ignored layers: ${SGLANG_FP4_IGNORED_LAYERS}"
echo "NVFP4 4-over-6: ${FLASHINFER_NVFP4_4OVER6}"
echo "NVFP4 error mode: ${FLASHINFER_NVFP4_4OVER6_ERR_MODE}"
echo "Add BOS token: ${ADD_BOS_TOKEN}"
echo "Parallelism: TP=${TP_SIZE}, DP=${DP_SIZE}, EP=${EP_SIZE}"
echo "Tasks: ${TASK}"
echo "Batch size: ${BATCH_SIZE}"
echo "Logprob chunk size: ${SGLANG_LOGPROB_CHUNK_SIZE}"
echo "Watchdog timeout: ${WATCHDOG_TIMEOUT}s"
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
