#!/bin/bash

#SBATCH -p batch
#SBATCH -A coreai_chef_numerics
#SBATCH -J eval-sglang-bf16-ling-mini-v2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=3:10:00
#SBATCH --output=/lustre/fsw/portfolios/coreai/users/xshang/lm-evaluation-harness/Ling-Mini/slurm_logs/ling-mini-v2-eval-sglang-bf16_%j.out

set -o pipefail

# Support both:
#   sbatch sglang_bf16_sft_eval.sh 3000
#   sbatch --export=ALL,ITER_ID=3000 sglang_bf16_sft_eval.sh
if [ "$#" -gt 0 ]; then
    export ITER_ID="$1"
fi

MAX_RESTARTS=${MAX_RESTARTS:-10}
REQUEUE_DELAY_SECONDS=${REQUEUE_DELAY_SECONDS:-180}
REQUEUE_REQUESTED=0

requeue_job() {
    local reason="$1"
    local restart_count="${SLURM_RESTART_COUNT:-0}"
    local delay="${REQUEUE_DELAY_SECONDS}"

    if [ "$REQUEUE_REQUESTED" -eq 1 ]; then
        echo "Requeue already requested in this run; skipping duplicate request."
        return
    fi
    REQUEUE_REQUESTED=1

    echo "Requeue requested. reason=${reason}, job_id=${SLURM_JOB_ID}, restart_count=${restart_count}, max_restarts=${MAX_RESTARTS}"

    if [ "$restart_count" -ge "$MAX_RESTARTS" ]; then
        echo "Reached max restarts; not requeueing job ${SLURM_JOB_ID}."
        return
    fi

    if [ -n "${SLURM_JOB_END_TIME:-}" ]; then
        local now
        local safe_remaining
        now=$(date +%s)
        safe_remaining=$((SLURM_JOB_END_TIME - now - 20))
        if [ "$safe_remaining" -le 0 ]; then
            delay=0
        elif [ "$delay" -gt "$safe_remaining" ]; then
            delay="$safe_remaining"
        fi
    fi

    if [ "$delay" -gt 0 ]; then
        echo "Sleeping ${delay}s before requeue..."
        sleep "$delay"
    fi

    # Enable this after confirming the job is stable:
    # scontrol requeue "${SLURM_JOB_ID}"
}

trap 'requeue_job "received USR1"; exit 0' USR1
trap 'requeue_job "received TERM"; exit 0' TERM
trap 'requeue_job "received INT"; exit 0' INT

echo "=== Job Information ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node List: $SLURM_JOB_NODELIST"
echo "Number of Nodes: $SLURM_NNODES"
echo "Tasks per Node: $SLURM_NTASKS_PER_NODE"
echo "======================="

export CONTAINER_IMAGE=${CONTAINER_IMAGE:-/lustre/fsw/portfolios/coreai/projects/coreai_chef_numerics/users/xshang/sqsh/PyTorch-2606-PY3-NVFP4-EVAL-SGLANG.sqsh}

srun --container-image="$CONTAINER_IMAGE" \
     --container-writable \
     --container-mounts=/home/xshang:/home/xshang,/lustre/fsw/portfolios/:/lustre/fsw/portfolios \
     --container-workdir=/lustre/fsw/portfolios/coreai/users/xshang/lm-evaluation-harness/Ling-Mini \
     --export=ALL \
     bash <<'EOF'
#!/bin/bash
set -o pipefail

SCRIPT_DIR=$(pwd)
HARNESS_ROOT=$(dirname "$SCRIPT_DIR")
SGLANG_SRC=${SGLANG_SRC:-/lustre/fsw/portfolios/coreai/users/xshang/Quark/sglang}

if [ ! -f "${SGLANG_SRC}/python/sglang/srt/models/bailing_moe.py" ]; then
    echo "Error: SGLang source tree not found: ${SGLANG_SRC}"
    exit 1
fi

export PYTHONPATH="${SGLANG_SRC}/python:${HARNESS_ROOT}:${PYTHONPATH:-}"

export HF_HOME=/lustre/fsw/portfolios/coreai/projects/coreai_chef_numerics/users/xshang/huggingface
export HF_ALLOW_CODE_EVAL=1
export HF_HUB_TRUST_REMOTE_CODE=1
export TOKENIZERS_PARALLELISM=false
export NCCL_TIMEOUT=3600000
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_FR_BUFFER_SIZE=1048576
# Avoid Python resource_tracker races from Inductor's compile worker pool at exit.
export TORCHINDUCTOR_COMPILE_THREADS=${TORCHINDUCTOR_COMPILE_THREADS:-1}

python -c "import lm_eval; print(f'lm_eval package: {lm_eval.__file__}')"

# Keep compiler caches on node-local storage.
export TRITON_CACHE_DIR="/tmp/triton_cache_${SLURM_JOB_ID}_${SLURM_NODEID}"
export TORCHINDUCTOR_CACHE_DIR="/tmp/torchinductor_cache_${SLURM_JOB_ID}_${SLURM_NODEID}"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

MODEL_BASE_PATH=${MODEL_BASE_PATH:-/lustre/fsw/portfolios/coreai/users/xshang/my-script/Ling/Ling-mini/checkpoint/Ling-mini-v2-SFT-BF16-hf}
ITER_ID_INPUT=${ITER_ID:-1000}

# MODEL_PATH takes precedence when explicitly provided. Otherwise construct an
# iteration directory from ITER_ID=3000, 0003000, or iter_0003000.
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
TASK=${TASK:-arc_easy,arc_challenge,lambada_standard,mmlu,winogrande,agieval,humaneval}
RESULTS_DIR=${RESULTS_DIR:-./results/sglang_bf16_${ITER_DIR}}
AUTO_INSTALL_SGLANG=${AUTO_INSTALL_SGLANG:-1}
APPLY_CHAT_TEMPLATE=${APPLY_CHAT_TEMPLATE:-0}

if [ "$EP_SIZE" -lt 1 ] || [ $((TP_SIZE % EP_SIZE)) -ne 0 ]; then
    echo "Error: EP_SIZE must be >= 1 and divide TP_SIZE (TP_SIZE=${TP_SIZE}, EP_SIZE=${EP_SIZE})."
    exit 1
fi

pip install sacrebleu pytablewriter
pip install evaluate

if [ ! -f "${MODEL_PATH}/config.json" ]; then
    echo "Error: HuggingFace config not found: ${MODEL_PATH}/config.json"
    exit 1
fi

if ! python -c "import sglang" >/dev/null 2>&1; then
    if [ "$AUTO_INSTALL_SGLANG" -ne 1 ]; then
        echo "Error: sglang is not installed. Use an SGLang image or set AUTO_INSTALL_SGLANG=1."
        exit 1
    fi
    echo "SGLang is not installed; installing it into the writable container..."
    python -m pip install "sglang[all]"
fi

python - <<'PY'
import sglang

print(f"SGLang version: {getattr(sglang, '__version__', 'unknown')}")
try:
    from sglang.srt.models.bailing_moe import BailingMoeV2ForCausalLM  # noqa: F401
except ImportError as exc:
    raise RuntimeError(
        "This SGLang version does not support BailingMoeV2ForCausalLM. "
        "Please use a newer SGLang container/package."
    ) from exc
PY

mkdir -p "$RESULTS_DIR"

MODEL_ARGS="pretrained=${MODEL_PATH},tokenizer_path=${MODEL_PATH},tp_size=${TP_SIZE},dp_size=${DP_SIZE},ep_size=${EP_SIZE},dtype=bfloat16,max_model_len=${MAX_MODEL_LEN},mem_fraction_static=${MEM_FRACTION_STATIC},trust_remote_code=True"

EXTRA_EVAL_ARGS=()
if [ "$APPLY_CHAT_TEMPLATE" -eq 1 ]; then
    EXTRA_EVAL_ARGS+=(--apply_chat_template)
fi

echo "Iteration: ${ITER_DIR}"
echo "Model path: ${MODEL_PATH}"
echo "Parallelism: TP=${TP_SIZE}, DP=${DP_SIZE}, EP=${EP_SIZE}"
echo "Model args: ${MODEL_ARGS}"
echo "Tasks: ${TASK}"
echo "Batch size: ${BATCH_SIZE}"

# SGLang manages its own GPU worker processes; do not launch lm_eval with torchrun.
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
    requeue_job "evaluation exited with status ${status}"
fi
exit "$status"
