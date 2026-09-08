#!/bin/bash

#SBATCH -p batch
#SBATCH -A coreai_chef_numerics
#SBATCH -J eval-bf16-ling-mini-v2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=3:50:00
#SBATCH --output=/lustre/fsw/portfolios/coreai/users/xshang/lm-evaluation-harness/Ling-Mini/slurm_logs/ling-mini-v2-eval-bf16_%j.out
##SBATCH --signal=B:USR1@300
##SBATCH --open-mode=append

set -o pipefail

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
    else
        echo "No safe delay left before job end; requeue immediately."
    fi
    # scontrol requeue "${SLURM_JOB_ID}"
}

trap 'requeue_job "received USR1"; exit 0' USR1
trap 'requeue_job "received TERM"; exit 0' TERM
trap 'requeue_job "received INT"; exit 0' INT

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR
export MASTER_PORT=${MASTER_PORT:-29500}

echo "=== Job Information ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node List: $SLURM_JOB_NODELIST"
echo "Number of Nodes: $SLURM_NNODES"
echo "Tasks per Node: $SLURM_NTASKS_PER_NODE"
echo "Master Address: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"
echo "======================="

export CONTAINER_IMAGE=/lustre/fsw/portfolios/coreai/projects/coreai_chef_numerics/users/xshang/sqsh/AutoswitchGemm-Pytorch2512_EVAL.sqsh

srun --container-image=$CONTAINER_IMAGE \
     --container-writable \
     --container-mounts=/home/xshang:/home/xshang,/lustre/fsw/portfolios/:/lustre/fsw/portfolios \
     --container-workdir=/lustre/fsw/portfolios/coreai/users/xshang/lm-evaluation-harness/Ling-Mini \
     --export=ALL \
     bash << 'EOF'
#!/bin/bash
set -o pipefail

SCRIPT_DIR=$(pwd)
HARNESS_ROOT=$(dirname "$SCRIPT_DIR")
export PYTHONPATH="${HARNESS_ROOT}:${PYTHONPATH:-}"

export HF_HOME=/lustre/fsw/portfolios/coreai/projects/coreai_chef_numerics/users/xshang/huggingface
export MEGATRON_PATH=/lustre/fsw/portfolios/coreai/projects/coreai_chef_numerics/users/xshang/Megatron-LM
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HF_ALLOW_CODE_EVAL=1
export HF_HUB_TRUST_REMOTE_CODE=1

python -c "import lm_eval; print(f'lm_eval package: {lm_eval.__file__}')"

DTYPE=${DTYPE:-BF16}
if [ -z "${DTYPE}" ]; then
    echo "Error: DTYPE environment variable is not set. Please set DTYPE=NVFP4 or DTYPE=BF16"
    exit 1
elif [ "${DTYPE}" = "NVFP4" ]; then
    LOAD=/lustre/raplab/client/xshang/workspace/huggingface/Ling-Mini/NVFP4/
elif [ "${DTYPE}" = "BF16" ]; then
    LOAD=/lustre/fsw/portfolios/coreai/projects/coreai_chef_numerics/users/xshang/my-script/Ling/Ling-mini/checkpoint/Ling-mini-v2-SFT-BF16
else
    echo "Error: Invalid DTYPE value '${DTYPE}'. Must be NVFP4 or BF16"
    exit 1
fi

TOKENIZER_TYPE=HuggingFaceTokenizer
TOKENIZER_MODEL=moonshotai/Moonlight-16B-A3B-Instruct

EXTRA_ARGS="
    --use-checkpoint-args
    --no-use-tokenizer-model-from-checkpoint-args
    --trust-remote-code
    --untie-embeddings-and-output-weights
    --swiglu
    --use-mcore-models
    --fp4-recipe nvfp4
    --fp4-format e2m1
    --transformer-impl transformer_engine
    --disable-bias-linear
    --position-embedding-type rope
    --no-rope-fusion
    --rotary-base 10000
    --rotary-percent 0.5
    --rotary-scaling-factor 40
    --normalization RMSNorm
    --norm-epsilon 1e-6
    --group-query-attention
    --num-attention-heads 16
    --num-query-groups 4
    --attention-backend auto
    --hidden-dropout 0
    --num-layers 20
    --hidden-size 2048
    --ffn-hidden-size 5120
    --qk-layernorm
    --max-position-embeddings 4096
    --attention-dropout 0
    --num-experts 256
    --moe-layer-freq [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    --moe-ffn-hidden-size 512
    --moe-shared-expert-intermediate-size 512
    --moe-router-load-balancing-type aux_loss
    --moe-z-loss-coeff 0.0000035
    --moe-router-topk 8
    --moe-router-topk-scaling-factor 2.5
    --moe-grouped-gemm
    --moe-router-dtype fp32
    --moe-router-num-groups 8
    --moe-router-group-topk 4
    --moe-router-score-function sigmoid
    --moe-router-enable-expert-bias
    --moe-router-bias-update-rate 1e-3
    --moe-token-dispatcher-type alltoall
    --moe-shared-expert-overlap
    --moe-permute-fusion
    --moe-aux-loss-coeff 0.001
"

TP=${TP:-1}
EP=${EP:-8}
DEVICES=${DEVICES:-8}
CKPT_STEP=${STEP:-16024}
BATCH_SIZE=${BATCH:-16}
#arc_easy,arc_challenge,lambada_standard,mmlu,winogrande,agieval,humaneval
TASK=${TASK:-humaneval}
RESULTS_DIR=${RESULTS_DIR:-./results/bf16}

EXTRA_ARGS_ONELINE=$(echo $EXTRA_ARGS | tr '\n' ' ' | tr -s ' ')

read -r -d '' MODEL_ARGS_JSON <<EOJSON
{
  "devices": ${DEVICES},
  "tensor_model_parallel_size": ${TP},
  "expert_model_parallel_size": ${EP},
  "micro_batch_size": ${BATCH_SIZE},
  "load": "${LOAD}",
  "ckpt_step": "${CKPT_STEP}",
  "tokenizer_type": "${TOKENIZER_TYPE}",
  "tokenizer_model": "${TOKENIZER_MODEL}",
  "extra_args": "${EXTRA_ARGS_ONELINE}"
}
EOJSON

pip install transformers==4.57.6

torchrun --nproc_per_node=${DEVICES} --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT} \
    -m lm_eval --model megatron_lm \
    --model_args "${MODEL_ARGS_JSON}" \
    --tasks ${TASK} \
    --batch_size ${BATCH_SIZE} \
    --num_fewshot 0 \
    --log_samples \
    --confirm_run_unsafe_code \
    --output_path "${RESULTS_DIR}"
EOF

status=$?
if [ "$status" -ne 0 ]; then
    requeue_job "evaluation exited with status ${status}"
fi
exit "$status"
