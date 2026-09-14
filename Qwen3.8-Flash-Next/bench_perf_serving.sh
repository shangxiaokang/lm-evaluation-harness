#!/usr/bin/env bash
# Benchmark an already-running SGLang server with controlled request shapes.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
# shellcheck source=perf_common.sh
source "${SCRIPT_DIR}/perf_common.sh"

MODE_ARG="${1:-${MODE:-}}"
resolve_mode "${MODE_ARG}"

WORKLOADS="${WORKLOADS:-balanced:1024:256 prefill:4096:32 decode:128:1024}"
CONCURRENCIES="${CONCURRENCIES:-1 4 16 32}"
REPEATS="${REPEATS:-3}"
REQUEST_MULTIPLIER="${REQUEST_MULTIPLIER:-10}"
MIN_PROMPTS="${MIN_PROMPTS:-64}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-auto}"
REQUEST_RATE="${REQUEST_RATE:-inf}"
SEED="${SEED:-42}"
OUTPUT_DETAILS="${OUTPUT_DETAILS:-0}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RESULTS_DIR="${RESULTS_DIR:-${PERF_OUTPUT_ROOT}/${RUN_ID}/${MODE}}"
RESULT_FILE="${RESULT_FILE:-${RESULTS_DIR}/serving.jsonl}"
BENCH_LOG="${BENCH_LOG:-${RESULTS_DIR}/benchmark.log}"
BASE_URL="http://${BENCH_HOST}:${PORT}"

mkdir -p "${RESULTS_DIR}"

[[ -x "${PYTHON_BIN}" ]] || die "Python is not executable: ${PYTHON_BIN}"
command -v curl >/dev/null 2>&1 || die "curl is required"
curl -fsS "${BASE_URL}/health" >/dev/null || \
  die "SGLang server is not ready at ${BASE_URL}"

if [[ -s "${RESULT_FILE}" ]]; then
  die "result file already exists and is non-empty: ${RESULT_FILE}"
fi

SERVER_INFO_FILE="${RESULTS_DIR}/serving_server_info.json"
RESOLVED_CONFIG_FILE="${RESULTS_DIR}/serving_resolved_config.txt"
[[ ! -e "${SERVER_INFO_FILE}" ]] || \
  die "server-info file already exists: ${SERVER_INFO_FILE}"
curl -fsS "${BASE_URL}/server_info" > "${SERVER_INFO_FILE}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_perf.py" \
  --validate-server-info "${SERVER_INFO_FILE}" \
  --mode "${MODE}" \
  --expected-quantization "${EXPECTED_QUANTIZATION}" \
  --validate-controlled-config | tee "${RESOLVED_CONFIG_FILE}"

DETAIL_ARGS=()
if is_true "${OUTPUT_DETAILS}"; then
  DETAIL_ARGS+=(--output-details)
fi

echo "==================================================" | tee -a "${BENCH_LOG}"
echo "SGLang serving benchmark: ${MODE}" | tee -a "${BENCH_LOG}"
echo "Workloads:     ${WORKLOADS}" | tee -a "${BENCH_LOG}"
echo "Concurrency:   ${CONCURRENCIES}" | tee -a "${BENCH_LOG}"
echo "Repeats:       ${REPEATS}" | tee -a "${BENCH_LOG}"
echo "Result file:   ${RESULT_FILE}" | tee -a "${BENCH_LOG}"
echo "==================================================" | tee -a "${BENCH_LOG}"

for workload_spec in ${WORKLOADS}; do
  IFS=':' read -r workload_name input_len output_len <<< "${workload_spec}"
  [[ -n "${workload_name}" && "${input_len}" =~ ^[0-9]+$ && "${output_len}" =~ ^[0-9]+$ ]] || \
    die "invalid workload '${workload_spec}'; expected name:input_len:output_len"
  (( input_len + output_len <= CONTEXT_LENGTH )) || \
    die "${workload_name}: input ${input_len} + output ${output_len} exceeds CONTEXT_LENGTH=${CONTEXT_LENGTH}"

  for concurrency in ${CONCURRENCIES}; do
    [[ "${concurrency}" =~ ^[1-9][0-9]*$ ]] || \
      die "invalid concurrency: ${concurrency}"

    num_prompts=$((concurrency * REQUEST_MULTIPLIER))
    if (( num_prompts < MIN_PROMPTS )); then
      num_prompts="${MIN_PROMPTS}"
    fi

    if [[ "${WARMUP_REQUESTS}" == "auto" ]]; then
      warmup_requests="${concurrency}"
    else
      [[ "${WARMUP_REQUESTS}" =~ ^[1-9][0-9]*$ ]] || \
        die "WARMUP_REQUESTS must be a positive integer or auto"
      warmup_requests="${WARMUP_REQUESTS}"
    fi

    for ((repeat = 1; repeat <= REPEATS; repeat++)); do
      tag="mode=${MODE}|workload=${workload_name}|isl=${input_len}|osl=${output_len}|c=${concurrency}|n=${num_prompts}|repeat=${repeat}"
      BENCH_CMD=(
        "${PYTHON_BIN}" -m sglang.benchmark.serving
        --backend sglang
        --host "${BENCH_HOST}"
        --port "${PORT}"
        --model "${TOKENIZER_PATH}"
        --served-model-name "${SERVED_MODEL_NAME}"
        --tokenizer "${TOKENIZER_PATH}"
        --dataset-name random-ids
        --tokenize-prompt
        --random-input-len "${input_len}"
        --random-output-len "${output_len}"
        --random-range-ratio 1
        --num-prompts "${num_prompts}"
        --request-rate "${REQUEST_RATE}"
        --max-concurrency "${concurrency}"
        --warmup-requests "${warmup_requests}"
        --temperature 0
        --flush-cache
        --seed "${SEED}"
        --disable-tqdm
        --tag "${tag}"
        --output-file "${RESULT_FILE}"
        "${DETAIL_ARGS[@]}"
      )

      echo | tee -a "${BENCH_LOG}"
      echo "[${MODE}] workload=${workload_name}, ISL=${input_len}, OSL=${output_len}, concurrency=${concurrency}, repeat=${repeat}/${REPEATS}" | tee -a "${BENCH_LOG}"
      printf 'Command:' | tee -a "${BENCH_LOG}"
      printf ' %q' "${BENCH_CMD[@]}" | tee -a "${BENCH_LOG}"
      printf '\n' | tee -a "${BENCH_LOG}"

      "${BENCH_CMD[@]}" 2>&1 | tee -a "${BENCH_LOG}"
    done
  done
done

echo "Completed ${MODE}. Results: ${RESULT_FILE}" | tee -a "${BENCH_LOG}"
