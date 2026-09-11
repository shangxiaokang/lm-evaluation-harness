#!/bin/bash
# =============================================================================
# Source-install lm-eval + SGLang backend
# =============================================================================
# lm-eval has no [sglang] extra. Install the harness from this tree, then
# install SGLang separately (https://docs.sglang.io/get_started/install.html).
#
# E.g. on the node (inside the container):
#   bash Qwen3.6-35B/install_sglang.sh
#   SKIP_SGLANG=1 bash Qwen3.6-35B/install_sglang.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
HARNESS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${PYTHON:-python3}"
SKIP_SGLANG="${SKIP_SGLANG:-0}"

echo "=============================================="
echo "Install lm-eval from source + SGLang backend"
echo "=============================================="
echo "Harness: ${HARNESS_DIR}"
echo "Python:  ${PYTHON} ($("${PYTHON}" --version 2>&1))"
echo "=============================================="

cd "${HARNESS_DIR}"
"${PYTHON}" -m pip install --upgrade pip
"${PYTHON}" -m pip install -e ".[hf]"

# NGC/enroot Python is PEP 668 "externally managed". pip already works here
# (used above for lm_eval). uv --system does not, and pip has --pre not --prerelease.
export PIP_BREAK_SYSTEM_PACKAGES="${PIP_BREAK_SYSTEM_PACKAGES:-1}"

if [[ "${SKIP_SGLANG}" != "1" ]]; then
  if "${PYTHON}" -c "import sglang" >/dev/null 2>&1; then
    echo "[INFO] sglang already importable: $("${PYTHON}" -c 'import sglang; print(getattr(sglang, "__version__", "unknown"))')"
  else
    echo "[INFO] Installing sglang with pip --pre (CUDA 13 default; Qwen3.6 needs >=0.5.10)"
    "${PYTHON}" -m pip install --pre sglang
  fi
fi

echo "=============================================="
echo "Verify"
echo "=============================================="
"${PYTHON}" - <<'PY'
import importlib.util
import lm_eval

print(f"lm_eval: {lm_eval.__file__}")
assert "EleutherAI/lm-evaluation-harness" in lm_eval.__file__.replace("\\", "/"), (
    f"lm_eval is not the source tree: {lm_eval.__file__}"
)

if importlib.util.find_spec("sglang") is None:
    raise SystemExit("sglang is not installed")

import sglang
from lm_eval.api.registry import get_model

print(f"sglang: {getattr(sglang, '__version__', 'unknown')}")
cls = get_model("sglang")
print(f"lm-eval backend: {cls}")
print("OK: source lm-eval + sglang backend")
PY

echo "=============================================="
echo "Done. E.g. run eval:"
echo "  bash ${SCRIPT_DIR}/run_sglang.sh                 # NVFP4 online (BF16/FP8)"
echo "  bash ${SCRIPT_DIR}/run_sglang_nvfp4_offline.sh   # NVFP4 offline (serialized)"
echo "  bash ${SCRIPT_DIR}/run_sglang_hf.sh              # Hub BF16, no NVFP4"
echo "=============================================="
