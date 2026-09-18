#!/usr/bin/env bash
# Sequentially benchmark BF16, NVFP4 online, and NVFP4 offline.
#
# SGLang models EP as a sub-dimension of its outer TP process group. Therefore
# four-way EP with effective Attention TP=1 and MoE TP=1 uses an outer
# tp_size=4 process group plus four-way DP attention; tp_size=1, ep_size=4 is
# not a valid SGLang topology.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"

# Four physical GPUs: outer world=4, DP attention=4, EP=4. The resulting
# effective Attention TP and MoE TP are both 1.
TP_SIZE="${TP_SIZE:-4}"
DP_SIZE="${DP_SIZE:-4}"
EP_SIZE="${EP_SIZE:-4}"
MOE_DP_SIZE="${MOE_DP_SIZE:-1}"
ENABLE_DP_ATTENTION="${ENABLE_DP_ATTENTION:-1}"
ENABLE_DP_LM_HEAD="${ENABLE_DP_LM_HEAD:-1}"
MOE_A2A_BACKEND="${MOE_A2A_BACKEND:-none}"

# shellcheck source=perf_common.sh
source "${SCRIPT_DIR}/perf_common.sh"

MODES="${MODES:-bf16 nvfp4_online nvfp4_offline}"
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
    if kill -0 -- "-${SERVER_PID}" 2>/dev/null || \
       kill -0 "${SERVER_PID}" 2>/dev/null; then
      kill -TERM -- "-${SERVER_PID}" 2>/dev/null || \
        kill -TERM "${SERVER_PID}" 2>/dev/null || true
    fi
    for _ in $(seq 1 60); do
      if ! kill -0 -- "-${SERVER_PID}" 2>/dev/null && \
         ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        break
      fi
      sleep 1
    done
    if kill -0 -- "-${SERVER_PID}" 2>/dev/null || \
       kill -0 "${SERVER_PID}" 2>/dev/null; then
      kill -KILL -- "-${SERVER_PID}" 2>/dev/null || \
        kill -KILL "${SERVER_PID}" 2>/dev/null || true
    fi
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
[[ -x "${PYTHON_BIN}" ]] || die "Python is not executable: ${PYTHON_BIN}"

[[ ! -e "${RUN_DIR}" ]] || \
  die "run directory already exists; choose a new RUN_ID: ${RUN_DIR}"
mkdir -p "${RUN_DIR}"

{
  echo "timestamp=$(date --iso-8601=seconds)"
  echo "hostname=$(hostname)"
  echo "run_id=${RUN_ID}"
  echo "modes=${MODES}"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
  echo "sglang_outer_tp_size=${TP_SIZE}"
  echo "dp_size=${DP_SIZE}"
  echo "ep_size=${EP_SIZE}"
  echo "moe_dp_size=${MOE_DP_SIZE}"
  echo "enable_dp_attention=${ENABLE_DP_ATTENTION}"
  echo "enable_dp_lm_head=${ENABLE_DP_LM_HEAD}"
  echo "moe_a2a_backend=${MOE_A2A_BACKEND}"
  echo "effective_attention_tp_size=${ATTENTION_TP_SIZE}"
  echo "effective_moe_tp_size=${MOE_TP_SIZE}"
  echo "dtype=${DTYPE}"
  echo "kv_cache_dtype=${KV_CACHE_DTYPE}"
  echo "ple_offload_embedding=${PLE_OFFLOAD_EMBEDDING}"
  echo "disable_radix_cache=${DISABLE_RADIX_CACHE}"
  echo "max_running_requests=${MAX_RUNNING_REQUESTS}"
  echo "max_mamba_cache_size=${MAX_MAMBA_CACHE_SIZE}"
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

  if curl -fsS "http://${BENCH_HOST}:${PORT}/health" >/dev/null 2>&1; then
    die "port ${PORT} already has a healthy server; stop it before benchmarking"
  fi

  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "timestamp,index,memory_used_mib,gpu_utilization_pct,memory_utilization_pct,power_draw_w" > "${GPU_LOG}"
    nvidia-smi -i "${CUDA_VISIBLE_DEVICES}" \
      --query-gpu=timestamp,index,memory.used,utilization.gpu,utilization.memory,power.draw \
      --format=csv,noheader,nounits \
      -l 1 >> "${GPU_LOG}" 2>&1 &
    GPU_MONITOR_PID=$!
  fi

  start_seconds="$(date +%s)"
  setsid bash "${SCRIPT_DIR}/launch_perf_server.sh" "${mode}" \
    > "${SERVER_LOG}" 2>&1 &
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
  curl -fsS "http://${BENCH_HOST}:${PORT}/server_info" \
    > "${MODE_DIR}/server_info.json"

  "${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_perf.py" \
    --validate-server-info "${MODE_DIR}/server_info.json" \
    --mode "${mode}" \
    --expected-quantization "${EXPECTED_QUANTIZATION}" \
    --validate-controlled-config | tee "${MODE_DIR}/resolved_config.txt"

  echo "Server ready in $((ready_seconds - start_seconds)) seconds."

  RESULTS_DIR="${MODE_DIR}" RUN_ID="${RUN_ID}" \
    bash "${SCRIPT_DIR}/bench_perf_serving.sh" "${mode}"

  if is_true "${RUN_ONE_BATCH}"; then
    RESULTS_DIR="${MODE_DIR}" RUN_ID="${RUN_ID}" \
      bash "${SCRIPT_DIR}/bench_perf_one_batch.sh" "${mode}"
  fi

  stop_server
  stop_gpu_monitor

  # Give CUDA/NCCL processes a moment to release their contexts before the
  # next precision mode is started.
  sleep 5
  if curl -fsS "http://${BENCH_HOST}:${PORT}/health" >/dev/null 2>&1; then
    die "server for ${mode} still answers after process-group shutdown"
  fi
done

SUMMARY_CMD=(
  "${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_perf.py" "${RUN_DIR}"
  --expected-repeats "${REPEATS:-3}"
  --expected-modes
)
for mode in ${MODES}; do
  SUMMARY_CMD+=("${mode}")
done
"${SUMMARY_CMD[@]}"

echo
echo "=================================================="
echo "All precision modes completed"
echo "Run directory: ${RUN_DIR}"
echo "Summary CSV:   ${RUN_DIR}/summary.csv"
echo "Summary MD:    ${RUN_DIR}/summary.md"
echo "Server CSV:    ${RUN_DIR}/server_summary.csv"
echo "Server MD:     ${RUN_DIR}/server_summary.md"
echo "=================================================="
