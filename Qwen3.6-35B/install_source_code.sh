#!/bin/bash
# =============================================================================
# Source-install lm-eval + local SGLang backend
# =============================================================================
# lm-eval has no [sglang] extra. Install the harness from this tree, then
# install SGLang in editable mode from a local source checkout.
#
# E.g. on the node (inside the container):
#   bash Qwen3.6-35B/install_sglang.sh
#   SGLANG_SOURCE_DIR=/path/to/sglang bash Qwen3.6-35B/install_sglang.sh
#   SKIP_SGLANG=1 bash Qwen3.6-35B/install_sglang.sh  # already source-installed
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
HARNESS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${PYTHON:-python3}"
SKIP_SGLANG="${SKIP_SGLANG:-0}"
SGLANG_SOURCE_DIR="${SGLANG_SOURCE_DIR:-${HARNESS_DIR}/../sglang}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"
# lm-eval uses the Python Engine. The optional Rust HTTP/gRPC/tree extensions
# require Cargo and are not needed here; set this to all when they are desired.
SGLANG_BUILD_RUST_EXTS="${SGLANG_BUILD_RUST_EXTS:-none}"

if ! command -v "${PYTHON}" >/dev/null 2>&1; then
  echo "Error: ${PYTHON} was not found." >&2
  echo "Set PYTHON=/path/to/python (Python 3.10 or newer)." >&2
  exit 1
fi

PYTHON_EXE="$("${PYTHON}" -c 'import sys; print(sys.executable)')"
if ! "${PYTHON_EXE}" -c 'import sys; raise SystemExit(sys.version_info < (3, 10))'; then
  echo "Error: SGLang requires Python 3.10 or newer." >&2
  exit 1
fi

if [[ ! -d "${SGLANG_SOURCE_DIR}" ]]; then
  echo "Error: SGLang source directory does not exist: ${SGLANG_SOURCE_DIR}" >&2
  echo "Set SGLANG_SOURCE_DIR=/path/to/sglang." >&2
  exit 1
fi
SGLANG_SOURCE_DIR="$(cd "${SGLANG_SOURCE_DIR}" && pwd)"
SGLANG_PYTHON_DIR="${SGLANG_SOURCE_DIR}/python"
if [[ ! -f "${SGLANG_PYTHON_DIR}/pyproject.toml" || \
      ! -f "${SGLANG_PYTHON_DIR}/sglang/__init__.py" ]]; then
  echo "Error: not an SGLang source checkout: ${SGLANG_SOURCE_DIR}" >&2
  echo "Expected python/pyproject.toml and python/sglang/__init__.py." >&2
  exit 1
fi

SGLANG_GIT_INFO="unknown"
if command -v git >/dev/null 2>&1 && \
    git -C "${SGLANG_SOURCE_DIR}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  SGLANG_GIT_BRANCH="$(git -C "${SGLANG_SOURCE_DIR}" rev-parse --abbrev-ref HEAD)"
  SGLANG_GIT_COMMIT="$(git -C "${SGLANG_SOURCE_DIR}" rev-parse --short=12 HEAD)"
  SGLANG_GIT_INFO="${SGLANG_GIT_BRANCH}@${SGLANG_GIT_COMMIT}"
fi

echo "=============================================="
echo "Install lm-eval + SGLang from local source"
echo "=============================================="
echo "Harness: ${HARNESS_DIR}"
echo "Python:  ${PYTHON_EXE} ($("${PYTHON_EXE}" --version 2>&1))"
echo "SGLang:  ${SGLANG_SOURCE_DIR} (${SGLANG_GIT_INFO})"
echo "PyTorch: ${PYTORCH_INDEX_URL}"
echo "Rust ext: ${SGLANG_BUILD_RUST_EXTS}"
echo "=============================================="

cd "${HARNESS_DIR}"

# NGC/enroot Python can be PEP 668 "externally managed".
export PIP_BREAK_SYSTEM_PACKAGES="${PIP_BREAK_SYSTEM_PACKAGES:-1}"

"${PYTHON_EXE}" -m pip install --upgrade pip uv
"${PYTHON_EXE}" -m pip install -e ".[hf]"

if [[ "${SKIP_SGLANG}" != "1" ]]; then
  if "${PYTHON_EXE}" -c "import sglang" >/dev/null 2>&1; then
    echo "[INFO] Replacing/upgrading installed SGLang: $("${PYTHON_EXE}" -c 'import importlib.metadata; print(importlib.metadata.version("sglang"))')"
  fi
  echo "[INFO] Installing SGLang editable source: ${SGLANG_PYTHON_DIR}"
  export SGLANG_BUILD_RUST_EXTS
  "${PYTHON_EXE}" -m uv pip install \
    --python "${PYTHON_EXE}" \
    --break-system-packages \
    --upgrade \
    --prerelease=allow \
    --index-strategy unsafe-best-match \
    --extra-index-url "${PYTORCH_INDEX_URL}" \
    --editable "${SGLANG_PYTHON_DIR}"
else
  echo "[INFO] SKIP_SGLANG=1; verifying the existing SGLang source install"
fi

echo "=============================================="
echo "Verify"
echo "=============================================="
EXPECTED_HARNESS_DIR="${HARNESS_DIR}" \
EXPECTED_SGLANG_PACKAGE_DIR="${SGLANG_PYTHON_DIR}/sglang" \
"${PYTHON_EXE}" - <<'PY'
import importlib.util
import os
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import lm_eval

print(f"lm_eval: {lm_eval.__file__}")
actual_lm_eval = Path(lm_eval.__file__).resolve().parent
expected_lm_eval = (Path(os.environ["EXPECTED_HARNESS_DIR"]) / "lm_eval").resolve()
if actual_lm_eval != expected_lm_eval:
    raise SystemExit(
        f"lm_eval is not the source tree: {actual_lm_eval}; "
        f"expected {expected_lm_eval}"
    )
if importlib.util.find_spec("sglang") is None:
    raise SystemExit("sglang is not installed")

import sglang
import torch
from lm_eval.api.registry import get_model

sglang_version = version("sglang")
print(f"sglang: {sglang_version} ({sglang.__file__})")
actual_sglang = Path(sglang.__file__).resolve().parent
expected_sglang = Path(os.environ["EXPECTED_SGLANG_PACKAGE_DIR"]).resolve()
if actual_sglang != expected_sglang:
    raise SystemExit(
        f"sglang is not loaded from the source tree: {actual_sglang}; "
        f"expected {expected_sglang}"
    )

for distribution in ("sglang-kernel", "sgl-deep-gemm", "flashinfer-python"):
    try:
        print(f"{distribution}: {version(distribution)}")
    except PackageNotFoundError:
        pass

print(f"torch: {torch.__version__} (CUDA {torch.version.cuda})")
if torch.version.cuda is None or torch.version.cuda.split(".", 1)[0] != "13":
    raise SystemExit(
        f"Expected a CUDA 13 PyTorch build, but torch reports {torch.version.cuda}"
    )
cls = get_model("sglang")
print(f"lm-eval backend: {cls}")
print("OK: source lm-eval + editable SGLang source backend")
PY

echo "=============================================="
echo "Done. E.g. run eval:"
echo "  bash ${SCRIPT_DIR}/run_sglang.sh                 # NVFP4 online (BF16/FP8)"
echo "  bash ${SCRIPT_DIR}/run_sglang_nvfp4_offline.sh   # NVFP4 offline (serialized)"
echo "  bash ${SCRIPT_DIR}/run_sglang_hf.sh              # Hub BF16, no NVFP4"
echo "=============================================="
