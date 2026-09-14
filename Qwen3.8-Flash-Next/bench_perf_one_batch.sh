#!/usr/bin/env bash
# Fixed-batch benchmark against an already-running SGLang server.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
# shellcheck source=perf_common.sh
source "${SCRIPT_DIR}/perf_common.sh"

MODE_ARG="${1:-${MODE:-}}"
resolve_mode "${MODE_ARG}"

BATCH_SIZES="${BATCH_SIZES:-1 4 16 32}"
INPUT_LENS="${INPUT_LENS:-1024}"
OUTPUT_LENS="${OUTPUT_LENS:-256}"
SEED="${SEED:-42}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RESULTS_DIR="${RESULTS_DIR:-${PERF_OUTPUT_ROOT}/${RUN_ID}/${MODE}}"
RESULT_FILE="${ONE_BATCH_RESULT_FILE:-${RESULTS_DIR}/one_batch.jsonl}"
BASE_URL="http://${BENCH_HOST}:${PORT}"

mkdir -p "${RESULTS_DIR}"
command -v curl >/dev/null 2>&1 || die "curl is required"
curl -fsS "${BASE_URL}/health" >/dev/null || \
  die "SGLang server is not ready at ${BASE_URL}"

if [[ -s "${RESULT_FILE}" ]]; then
  die "result file already exists and is non-empty: ${RESULT_FILE}"
fi

SERVER_INFO_FILE="${RESULTS_DIR}/one_batch_server_info.json"
RESOLVED_CONFIG_FILE="${RESULTS_DIR}/one_batch_resolved_config.txt"
[[ ! -e "${SERVER_INFO_FILE}" ]] || \
  die "server-info file already exists: ${SERVER_INFO_FILE}"
curl -fsS "${BASE_URL}/server_info" > "${SERVER_INFO_FILE}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_perf.py" \
  --validate-server-info "${SERVER_INFO_FILE}" \
  --mode "${MODE}" \
  --expected-quantization "${EXPECTED_QUANTIZATION}" \
  --validate-controlled-config | tee "${RESOLVED_CONFIG_FILE}"

read -r -a batch_args <<< "${BATCH_SIZES}"
read -r -a input_args <<< "${INPUT_LENS}"
read -r -a output_args <<< "${OUTPUT_LENS}"

"${PYTHON_BIN}" -m sglang.benchmark.one_batch_server \
  --model None \
  --base-url "${BASE_URL}" \
  --tp-size "${TP_SIZE}" \
  --local-tokenizer-path "${TOKENIZER_PATH}" \
  --dataset-name random-ids \
  --batch-size "${batch_args[@]}" \
  --input-len "${input_args[@]}" \
  --output-len "${output_args[@]}" \
  --cache-hit-rate 0 \
  --seed "${SEED}" \
  --run-name "${MODE}" \
  --result-filename "${RESULT_FILE}" \
  --show-report \
  --no-append-to-github-summary

echo "Completed fixed-batch benchmark. Results: ${RESULT_FILE}"
