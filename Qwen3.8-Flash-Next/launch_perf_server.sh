#!/usr/bin/env bash
# Launch one Qwen3.8-Flash-Next precision mode with controlled server settings.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
# shellcheck source=perf_common.sh
source "${SCRIPT_DIR}/perf_common.sh"

MODE_ARG="${1:-${MODE:-}}"
resolve_mode "${MODE_ARG}"
validate_mode_inputs
build_server_args

# Serialized checkpoints use their saved activation scales by default. Pinning
# this to 0 prevents an inherited setting from changing the offline recipe;
# nvfp4_online still forces its own per-token activation scaling.
export SGLANG_FLASHINFER_NVFP4_PER_TOKEN_ACTIVATION=0
unset SGLANG_FP4_IGNORED_LAYERS
unset SGLANG_FLASHINFER_CUTEDSL_NVFP4_W4A16
unset SGLANG_MOE_NVFP4_DISPATCH
unset SGLANG_NVFP4_CKPT_FP8_GEMM_IN_ATTN
unset SGLANG_NVFP4_CKPT_FP8_NEXTN_MOE
unset FLASHINFER_NVFP4_4OVER6
unset FLASHINFER_NVFP4_4OVER6_E4M3_USE_256

echo "=================================================="
echo "Qwen3.8-Flash-Next SGLang performance server"
echo "=================================================="
echo "Mode:              ${MODE}"
echo "Model:             ${MODEL_PATH}"
echo "Expected quant:    ${EXPECTED_QUANTIZATION}"
echo "GPUs:              ${CUDA_VISIBLE_DEVICES}"
echo "TP / DP:           ${TP_SIZE} / ${DP_SIZE}"
echo "Dtype / KV cache:  ${DTYPE} / ${KV_CACHE_DTYPE}"
echo "Context length:    ${CONTEXT_LENGTH}"
echo "Max running reqs:  ${MAX_RUNNING_REQUESTS}"
echo "Max Mamba states:  ${MAX_MAMBA_CACHE_SIZE}"
echo "PLE offload:       ${PLE_OFFLOAD_EMBEDDING}"
echo "Radix disabled:    ${DISABLE_RADIX_CACHE}"
echo "MoE backend:       ${MOE_RUNNER_BACKEND}"
echo "Linear attention:  ${LINEAR_ATTN_PREFILL_BACKEND} / ${LINEAR_ATTN_DECODE_BACKEND}"
echo "NVFP4 per-token env: ${SGLANG_FLASHINFER_NVFP4_PER_TOKEN_ACTIVATION}"
echo "Endpoint:          http://${HOST}:${PORT}"
echo "=================================================="

printf 'Command:'
printf ' %q' "${PYTHON_BIN}" -m sglang.launch_server "${SERVER_ARGS[@]}"
printf '\n'

exec "${PYTHON_BIN}" -m sglang.launch_server "${SERVER_ARGS[@]}"
