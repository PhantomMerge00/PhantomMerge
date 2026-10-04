#!/usr/bin/env bash
# Source from replication scripts:  source reproduce/env.sh
set -euo pipefail

export PHANTOM_MERGE_ROOT="${PHANTOM_MERGE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

export PYTHONPATH="${PHANTOM_MERGE_ROOT}/ccer:${PHANTOM_MERGE_ROOT}/third_party/jacobian-lens:${PYTHONPATH:-}"

export HF_HOME="${HF_HOME:-${PHANTOM_MERGE_ROOT}/runtime/.cache/huggingface}"

export CCER_MODEL_MANIFEST_ID="${CCER_MODEL_MANIFEST_ID:-qwen3-32b_shopping_v1}"

export CCER_VLLM_BASE_URL="${CCER_VLLM_BASE_URL:-http://127.0.0.1:8003/v1}"
export CCER_MITIGATION_VLLM_URL="${CCER_MITIGATION_VLLM_URL:-http://127.0.0.1:8012/v1}"

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"

_ccer_echo() { echo "[ccer-replication] $*"; }

_ccer_require_dir() {
  local d="$1"
  [[ -d "$d" ]] || { echo "ERROR: missing directory: $d" >&2; exit 1; }
}

_ccer_check_core() {
  _ccer_require_dir "${PHANTOM_MERGE_ROOT}/ccer"
  _ccer_require_dir "${PHANTOM_MERGE_ROOT}/scripts"
  _ccer_echo "ROOT=${PHANTOM_MERGE_ROOT}"
  _ccer_echo "PYTHONPATH ok; HF_HOME=${HF_HOME}"
  _ccer_echo "MODEL_MANIFEST_ID=${CCER_MODEL_MANIFEST_ID}"
}
