#!/usr/bin/env bash

#SBATCH --partition=batch
#SBATCH --account=coreai_chef_numerics
#SBATCH --job-name=eval-qwen35-nvfp4-fp8-attn
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/lustre/fsw/portfolios/coreai/users/xshang/lm-evaluation-harness/Qwen3.5/slurm_logs/qwen35-nvfp4-fp8-attn_%j.out

set -o pipefail

export CONTAINER_IMAGE=${CONTAINER_IMAGE:-/lustre/fsw/portfolios/coreai/projects/coreai_chef_numerics/users/xshang/sqsh/PyTorch-2606-PY3-NVFP4-EVAL-SGLANG.sqsh}

echo "=== Job Information ==="
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Node:      ${SLURM_JOB_NODELIST}"
echo "Container: ${CONTAINER_IMAGE}"
echo "======================="

srun --container-image="${CONTAINER_IMAGE}" \
    --container-writable \
    --container-mounts=/home/xshang:/home/xshang,/lustre/fsw/portfolios/:/lustre/fsw/portfolios \
    --container-workdir=/lustre/fsw/portfolios/coreai/users/xshang/lm-evaluation-harness/Qwen3.5 \
    --export=ALL \
    bash <<'EOF'
#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(pwd)
HARNESS_ROOT=$(dirname "${SCRIPT_DIR}")
SGLANG_SRC=${SGLANG_SRC:-/lustre/fsw/portfolios/coreai/users/xshang/Quark/sglang}
MODEL_PATH=${MODEL_PATH:-/lustre/fsw/portfolios/coreai/projects/coreai_chef_numerics/users/xshang/Qwen/Qwen3.5/Qwen3.5-35B-A3B-HF-NVFP4-FP8-Attn}

if [[ ! -f "${SGLANG_SRC}/python/sglang/srt/models/qwen3_5.py" ]]; then
    echo "ERROR: Qwen3.5 SGLang implementation not found: ${SGLANG_SRC}" >&2
    exit 1
fi
for path in \
    "${MODEL_PATH}/config.json" \
    "${MODEL_PATH}/model.safetensors.index.json" \
    "${MODEL_PATH}/hf_quant_config.json"; do
    if [[ ! -f "$path" ]]; then
        echo "ERROR: model checkpoint is incomplete; missing ${path}" >&2
        exit 1
    fi
done

export PYTHONPATH="${SGLANG_SRC}/python:${HARNESS_ROOT}:${PYTHONPATH:-}"
export HF_HOME=${HF_HOME:-/lustre/fsw/portfolios/coreai/users/xshang/huggingface}
export HF_ALLOW_CODE_EVAL=1
export HF_HUB_TRUST_REMOTE_CODE=1
export TOKENIZERS_PARALLELISM=false
export NCCL_TIMEOUT=3600000
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_FR_BUFFER_SIZE=1048576
export TORCHINDUCTOR_COMPILE_THREADS=${TORCHINDUCTOR_COMPILE_THREADS:-1}

# Large-vocabulary loglikelihood tasks such as MMLU can otherwise keep a
# single scheduler forward active longer than the default watchdog timeout.
export SGLANG_ENABLE_LOGPROB_CHUNK=${SGLANG_ENABLE_LOGPROB_CHUNK:-1}
export SGLANG_LOGPROB_CHUNK_SIZE=${SGLANG_LOGPROB_CHUNK_SIZE:-256}

LOCAL_CACHE_ROOT="/tmp/qwen35_eval_${SLURM_JOB_ID}_${SLURM_NODEID}"
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-${LOCAL_CACHE_ROOT}/triton}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-${LOCAL_CACHE_ROOT}/torchinductor}
export SGLANG_CACHE_DIR=${SGLANG_CACHE_DIR:-${LOCAL_CACHE_ROOT}/sglang}
mkdir -p "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${SGLANG_CACHE_DIR}"

TP_SIZE=${TP_SIZE:-1}
DP_SIZE=${DP_SIZE:-1}
EP_SIZE=${EP_SIZE:-1}
BATCH_SIZE=${BATCH:-2}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.8}
WATCHDOG_TIMEOUT=${WATCHDOG_TIMEOUT:-1800}
QUANTIZATION=${QUANTIZATION:-modelopt_fp4}
DTYPE=${DTYPE:-bfloat16}
MOE_RUNNER_BACKEND=${MOE_RUNNER_BACKEND:-flashinfer_trtllm}
ADD_BOS_TOKEN=${ADD_BOS_TOKEN:-False}

TASK=${TASK:-arc_easy,arc_challenge,winogrande,agieval,mmlu,humaneval,lambada_standard,gsm8k}
LIMIT=${LIMIT:-0}
NUM_FEWSHOT=${NUM_FEWSHOT:-0}
APPLY_CHAT_TEMPLATE=${APPLY_CHAT_TEMPLATE:-0}
RESULTS_DIR=${RESULTS_DIR:-${SCRIPT_DIR}/results/qwen35_nvfp4_fp8_attn}

if (( EP_SIZE < 1 || TP_SIZE % EP_SIZE != 0 )); then
    echo "ERROR: EP_SIZE must divide TP_SIZE; TP_SIZE=${TP_SIZE}, EP_SIZE=${EP_SIZE}" >&2
    exit 1
fi

python - "${MODEL_PATH}" "${QUANTIZATION}" <<'PY'
import inspect
import json
import sys
from pathlib import Path

import sglang
from sglang.srt.models import qwen3_5

model_path = Path(sys.argv[1])
quantization = sys.argv[2]
with (model_path / "hf_quant_config.json").open(encoding="utf-8") as file:
    quant_config = json.load(file)

quantization_section = quant_config.get("quantization", {})
quantized_layers = quantization_section.get("quantized_layers", {})
layer_algorithms = {
    layer_config.get("quant_algo")
    for layer_config in quantized_layers.values()
    if isinstance(layer_config, dict)
}
if not any(algo and "NVFP4" in algo for algo in layer_algorithms):
    raise RuntimeError(f"No NVFP4 layers found in {model_path / 'hf_quant_config.json'}")
if "FP8" not in layer_algorithms:
    raise RuntimeError(f"No FP8 attention layers found in {model_path / 'hf_quant_config.json'}")
if quantization != "modelopt_fp4":
    raise RuntimeError(f"Expected quantization='modelopt_fp4', got {quantization!r}")

print(f"Checkpoint quantization: {quantization_section.get('quant_algo')}")
print(f"Layer algorithms: {sorted(str(algo) for algo in layer_algorithms)}")
print(f"SGLang version: {getattr(sglang, '__version__', 'unknown')}")
print(f"SGLang package: {inspect.getfile(sglang)}")
print(f"Qwen3.5 model: {inspect.getfile(qwen3_5)}")
PY

if ! python -c "import torch; assert torch.cuda.is_available()" >/dev/null 2>&1; then
    echo "ERROR: CUDA is unavailable in the container." >&2
    exit 1
fi
if ! python -c "import sacrebleu, pytablewriter, evaluate" >/dev/null 2>&1; then
    python -m pip install sacrebleu pytablewriter evaluate
fi

mkdir -p "${RESULTS_DIR}"

MODEL_ARGS="pretrained=${MODEL_PATH},tokenizer_path=${MODEL_PATH},tp_size=${TP_SIZE},dp_size=${DP_SIZE},ep_size=${EP_SIZE},dtype=${DTYPE},quantization=${QUANTIZATION},moe_runner_backend=${MOE_RUNNER_BACKEND},load_format=auto,max_model_len=${MAX_MODEL_LEN},mem_fraction_static=${MEM_FRACTION_STATIC},watchdog_timeout=${WATCHDOG_TIMEOUT},add_bos_token=${ADD_BOS_TOKEN},trust_remote_code=True"

EXTRA_EVAL_ARGS=()
if (( APPLY_CHAT_TEMPLATE == 1 )); then
    EXTRA_EVAL_ARGS+=(--apply_chat_template)
fi
if (( LIMIT > 0 )); then
    EXTRA_EVAL_ARGS+=(--limit "${LIMIT}")
fi

echo "=== Qwen3.5 SGLang NVFP4 + FP8 Attention evaluation ==="
echo "Model:               ${MODEL_PATH}"
echo "Quantization:        ${QUANTIZATION}"
echo "MoE backend:         ${MOE_RUNNER_BACKEND}"
echo "Parallelism:         TP=${TP_SIZE}, DP=${DP_SIZE}, EP=${EP_SIZE}"
echo "Max model length:    ${MAX_MODEL_LEN}"
echo "Batch size:          ${BATCH_SIZE}"
echo "Logprob chunk size:  ${SGLANG_LOGPROB_CHUNK_SIZE}"
echo "Watchdog timeout:    ${WATCHDOG_TIMEOUT}s"
echo "Tasks:               ${TASK}"
echo "Few-shot:            ${NUM_FEWSHOT}"
echo "Limit:               ${LIMIT} (0 means full dataset)"
echo "Apply chat template: ${APPLY_CHAT_TEMPLATE}"
echo "Results:             ${RESULTS_DIR}"
echo "Model args:          ${MODEL_ARGS}"
echo "=========================================================="

python -m lm_eval run \
    --model sglang \
    --model_args "${MODEL_ARGS}" \
    --tasks "${TASK}" \
    --batch_size "${BATCH_SIZE}" \
    --num_fewshot "${NUM_FEWSHOT}" \
    --log_samples \
    --confirm_run_unsafe_code \
    --output_path "${RESULTS_DIR}" \
    "${EXTRA_EVAL_ARGS[@]}"
EOF

status=$?
if (( status == 0 )); then
    echo "Evaluation completed successfully."
else
    echo "Evaluation failed with status ${status}." >&2
fi
exit "${status}"
