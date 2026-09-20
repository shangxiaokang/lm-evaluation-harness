#!/usr/bin/env bash
# Launch one Qwen3.5-397B-A17B precision mode with controlled settings.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
source "${SCRIPT_DIR}/perf_common.sh"

MODE_ARG="${1:-${MODE:-}}"
resolve_mode "${MODE_ARG}"
validate_mode_inputs
build_server_args

# Keep serialized ModelOpt FP4 on its checkpoint activation scales. The
# nvfp4_online quantization config independently uses per-token online scales.
export SGLANG_FLASHINFER_NVFP4_PER_TOKEN_ACTIVATION=0
unset SGLANG_FP4_IGNORED_LAYERS
unset SGLANG_FLASHINFER_CUTEDSL_NVFP4_W4A16
unset SGLANG_MOE_NVFP4_DISPATCH
unset SGLANG_NVFP4_CKPT_FP8_GEMM_IN_ATTN
unset SGLANG_NVFP4_CKPT_FP8_NEXTN_MOE
unset FLASHINFER_NVFP4_4OVER6
unset FLASHINFER_NVFP4_4OVER6_E4M3_USE_256

echo "=================================================="
echo "Qwen3.5-397B-A17B SGLang performance server"
echo "=================================================="
echo "Mode:              ${MODE}"
echo "Model:             ${MODEL_PATH}"
echo "Expected quant:    ${EXPECTED_QUANTIZATION}"
echo "GPUs:              ${CUDA_VISIBLE_DEVICES}"
echo "Outer TP / DP:     ${TP_SIZE} / ${DP_SIZE}"
echo "Attention TP:      ${ATTENTION_TP_SIZE}"
echo "EP / MoE TP:       ${EP_SIZE} / ${MOE_TP_SIZE}"
echo "MoE DP / A2A:      ${MOE_DP_SIZE} / ${MOE_A2A_BACKEND}"
echo "DP attn / LM head: ${ENABLE_DP_ATTENTION} / ${ENABLE_DP_LM_HEAD}"
echo "Dtype / KV cache:  ${DTYPE} / ${KV_CACHE_DTYPE}"
echo "Context length:    ${CONTEXT_LENGTH}"
echo "Chunked prefill:   ${CHUNKED_PREFILL_SIZE}"
echo "Max running reqs:  ${MAX_RUNNING_REQUESTS}"
echo "Mamba:             ${MAMBA_SSM_DTYPE} / ${MAMBA_RADIX_CACHE_STRATEGY}"
echo "MoE backend:       ${MOE_RUNNER_BACKEND}"
echo "Endpoint:          http://${HOST}:${PORT}"
echo "=================================================="

printf 'Command:'
printf ' %q' "${PYTHON_BIN}" -m sglang.launch_server "${SERVER_ARGS[@]}"
printf '\n'

exec "${PYTHON_BIN}" -m sglang.launch_server "${SERVER_ARGS[@]}"

\n