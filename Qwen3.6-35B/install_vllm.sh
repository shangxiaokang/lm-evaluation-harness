#!/bin/bash
# =============================================================================
# Source-install lm-eval + vLLM backend
# =============================================================================
# Official extra: pip install -e ".[vllm]"  (vllm>=0.18)
#
# E.g. on the node (inside the container):
#   bash Qwen3.6-35B/install_vllm.sh
#   SKIP_VLLM=1 bash Qwen3.6-35B/install_vllm.sh
#   INSTALL_PRE=1 bash Qwen3.6-35B/install_vllm.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
HARNESS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${PYTHON:-python3}"
SKIP_VLLM="${SKIP_VLLM:-0}"
INSTALL_PRE="${INSTALL_PRE:-0}"

echo "=============================================="
echo "Install lm-eval from source + vLLM backend"
echo "=============================================="
echo "Harness: ${HARNESS_DIR}"
echo "Python:  ${PYTHON} ($("${PYTHON}" --version 2>&1))"
echo "=============================================="

cd "${HARNESS_DIR}"

# NGC/enroot Python is PEP 668 "externally managed". Use pip, not uv --system.
export PIP_BREAK_SYSTEM_PACKAGES="${PIP_BREAK_SYSTEM_PACKAGES:-1}"

"${PYTHON}" -m pip install --upgrade pip
"${PYTHON}" -m pip install -e ".[hf]"

if [[ "${SKIP_VLLM}" != "1" ]]; then
  if "${PYTHON}" -c "import vllm" >/dev/null 2>&1; then
    echo "[INFO] vllm already importable: $("${PYTHON}" -c 'import vllm; print(getattr(vllm, "__version__", "unknown"))')"
  else
    echo "[INFO] Installing lm_eval[vllm] (vllm>=0.18)"
    if [[ "${INSTALL_PRE}" == "1" ]]; then
      "${PYTHON}" -m pip install --pre -e ".[vllm]"
    else
      "${PYTHON}" -m pip install -e ".[vllm]"
    fi
  fi
  # Recent vLLM imports openai.types.responses.NamespaceTool (needs openai>=2.25.0).
  echo "[INFO] Ensuring openai>=2.25.0 for vLLM tool parsers"
  "${PYTHON}" -m pip install -U "openai>=2.25.0"

  # NGC image flash_attn is often ABI-mismatched with the running torch.
  # vLLM only suppresses ModuleNotFoundError (not ImportError) when loading
  # flash_attn RoPE, so a broken .so crashes EngineCore. Prefer the PYTHONPATH
  # stub in run_vllm.sh; uninstall here too if the import actually fails.
  if ! "${PYTHON}" -c "from flash_attn.ops.triton.rotary import apply_rotary" >/dev/null 2>&1; then
    echo "[INFO] flash_attn import failed (ABI mismatch). Uninstalling so vLLM can fall back."
    "${PYTHON}" -m pip uninstall -y flash-attn || true
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

if importlib.util.find_spec("vllm") is None:
    raise SystemExit("vllm is not installed")

import vllm
from lm_eval.api.registry import get_model

print(f"vllm: {getattr(vllm, '__version__', 'unknown')}")
import openai
from openai.types.responses import NamespaceTool  # noqa: F401

print(f"openai: {getattr(openai, '__version__', 'unknown')}")
cls = get_model("vllm")
print(f"lm-eval backend: {cls}")
print("OK: source lm-eval + vllm backend")
PY

echo "=============================================="
echo "Done. E.g. run eval:"
echo "  bash ${SCRIPT_DIR}/run_vllm.sh"
echo "=============================================="
