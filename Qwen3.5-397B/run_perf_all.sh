#!/usr/bin/env bash
# Run Qwen3.5-397B precision benchmarks sequentially.
#
# Default: 4-GPU EP4 + DP4 NVFP4 comparison. BF16 is deliberately opt-in:
# Qwen3.5-397B BF16 needs at least 8x B200/GB200-class GPUs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
source "${SCRIPT_DIR}/perf_common.sh"

if [[ -z "${MODES+x}" ]]; then
  MODES="nvfp4_online nvfp4_offline"
  echo "BF16 is omitted from the 4-GPU default because it does not fit on B200/GB200."
  echo "For an 8-GPU fair three-mode run, set MODES='bf16 nvfp4_online nvfp4_offline' TP_SIZE=8 DP_SIZE=4 EP_SIZE=4."
fi
RUN_ONE_BATCH="${RUN_ONE_BATCH:-0}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${PERF_OUTPUT_ROOT}/${RUN_ID}"

SERVER_PID=""
GPU_MONITOR_PID=""

stop_gpu_monitor() {
  if [[ -n "${GPU_MONITOR_PID}" ]] && kill -0 "${GPU_MONITOR_PID}" 2>/dev/null; then
    kill -TERM "${GPU_MONITOR_PID}" 2>/dev/null || true
    wait "${GPU_MONITOR_PID}" 2>/dev/null || true
  fi
  GPU_MONITOR_PID=""
}

stop_server() {
  if [[ -n "${SERVER_PID}" ]]; then
    kill -TERM -- "-${SERVER_PID}" 2>/dev/null || kill -TERM "${SERVER_PID}" 2>/dev/null || true
    for _ in $(seq 1 60); do
      if ! kill -0 -- "-${SERVER_PID}" 2>/dev/null && ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        break
      fi
      sleep 1
    done
    kill -KILL -- "-${SERVER_PID}" 2>/dev/null || kill -KILL "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
  SERVER_PID=""
}

cleanup() {
  stop_gpu_monitor
  stop_server
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

command -v curl >/dev/null 2>&1 || die "curl is required"
command -v setsid >/dev/null 2>&1 || die "setsid is required"
[[ ! -e "${RUN_DIR}" ]] || die "run directory already exists; choose a new RUN_ID: ${RUN_DIR}"
mkdir -p "${RUN_DIR}"

{
  echo "timestamp=$(date --iso-8601=seconds)"
  echo "hostname=$(hostname)"
  echo "run_id=${RUN_ID}"
  echo "modes=${MODES}"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
  echo "outer_tp_size=${TP_SIZE}"
  echo "dp_size=${DP_SIZE}"
  echo "ep_size=${EP_SIZE}"
  echo "effective_attention_tp_size=${ATTENTION_TP_SIZE}"
  echo "effective_moe_tp_size=${MOE_TP_SIZE}"
  echo "moe_a2a_backend=${MOE_A2A_BACKEND}"
  echo "moe_runner_backend=${MOE_RUNNER_BACKEND}"
  echo "context_length=${CONTEXT_LENGTH}"
  echo "chunked_prefill_size=${CHUNKED_PREFILL_SIZE}"
  echo "max_running_requests=${MAX_RUNNING_REQUESTS}"
  echo "workloads=${WORKLOADS:-balanced:12288:12288 prefill:16384:1024 decode:1024:16384}"
  if command -v git >/dev/null 2>&1 && [[ -d "${SGLANG_SOURCE}/.git" ]]; then
    echo "sglang_commit=$(git -C "${SGLANG_SOURCE}" rev-parse HEAD)"
  fi
  "${PYTHON_BIN}" -c 'import sglang, torch; print("sglang=" + str(getattr(sglang, "__version__", "unknown"))); print("sglang_file=" + str(sglang.__file__)); print("torch=" + str(torch.__version__))'
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -i "${CUDA_VISIBLE_DEVICES}" --query-gpu=index,name,uuid,driver_version,memory.total --format=csv,noheader
  fi
} > "${RUN_DIR}/environment.txt" 2>&1

echo "Run directory: ${RUN_DIR}"
for mode in ${MODES}; do
  resolve_mode "${mode}"
  validate_mode_inputs
  MODE_DIR="${RUN_DIR}/${mode}"
  SERVER_LOG="${MODE_DIR}/server.log"
  GPU_LOG="${MODE_DIR}/gpu_metrics.csv"
  mkdir -p "${MODE_DIR}"

  echo
  echo "=================================================="
  echo "Starting mode: ${mode}"
  echo "=================================================="
  curl -fsS "http://${BENCH_HOST}:${PORT}/health" >/dev/null 2>&1 && die "port ${PORT} already has a healthy server"

  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "timestamp,index,memory_used_mib,gpu_utilization_pct,memory_utilization_pct,power_draw_w" > "${GPU_LOG}"
    nvidia-smi -i "${CUDA_VISIBLE_DEVICES}" --query-gpu=timestamp,index,memory.used,utilization.gpu,utilization.memory,power.draw --format=csv,noheader,nounits -l 1 >> "${GPU_LOG}" 2>&1 &
    GPU_MONITOR_PID=$!
  fi

  start_seconds="$(date +%s)"
  setsid bash "${SCRIPT_DIR}/launch_perf_server.sh" "${mode}" > "${SERVER_LOG}" 2>&1 &
  SERVER_PID=$!
  deadline=$((SECONDS + SERVER_READY_TIMEOUT))
  while ! curl -fsS "http://${BENCH_HOST}:${PORT}/health" >/dev/null 2>&1; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      echo "Server exited before becoming ready: ${mode}" >&2
      tail -n 100 "${SERVER_LOG}" >&2 || true
      exit 1
    fi
    if (( SECONDS >= deadline )); then
      echo "Timed out waiting ${SERVER_READY_TIMEOUT}s for ${mode}" >&2
      tail -n 100 "${SERVER_LOG}" >&2 || true
      exit 1
    fi
    sleep 2
  done

  ready_seconds="$(date +%s)"
  echo "$((ready_seconds - start_seconds))" > "${MODE_DIR}/time_to_ready_seconds.txt"
  curl -fsS "http://${BENCH_HOST}:${PORT}/server_info" > "${MODE_DIR}/server_info.json"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_perf.py" --validate-server-info "${MODE_DIR}/server_info.json" --mode "${mode}" --expected-quantization "${EXPECTED_QUANTIZATION}" --validate-controlled-config | tee "${MODE_DIR}/resolved_config.txt"
  echo "Server ready in $((ready_seconds - start_seconds)) seconds."

  RESULTS_DIR="${MODE_DIR}" RUN_ID="${RUN_ID}" bash "${SCRIPT_DIR}/bench_perf_serving.sh" "${mode}"
  if is_true "${RUN_ONE_BATCH}"; then
    RESULTS_DIR="${MODE_DIR}" RUN_ID="${RUN_ID}" bash "${SCRIPT_DIR}/bench_perf_one_batch.sh" "${mode}"
  fi
  stop_server
  stop_gpu_monitor
  sleep 5
  curl -fsS "http://${BENCH_HOST}:${PORT}/health" >/dev/null 2>&1 && die "server for ${mode} still answers after shutdown"
done

SUMMARY_CMD=("${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_perf.py" "${RUN_DIR}" --expected-repeats "${REPEATS:-3}" --expected-modes)
for mode in ${MODES}; do SUMMARY_CMD+=("${mode}"); done
"${SUMMARY_CMD[@]}"

echo
echo "=================================================="
echo "All requested precision modes completed"
echo "Run directory: ${RUN_DIR}"
echo "Summary CSV:   ${RUN_DIR}/summary.csv"
echo "Summary MD:    ${RUN_DIR}/summary.md"
echo "Server CSV:    ${RUN_DIR}/server_summary.csv"
echo "Server MD:     ${RUN_DIR}/server_summary.md"
echo "=================================================="

\n