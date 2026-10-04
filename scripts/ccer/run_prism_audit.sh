#!/usr/bin/env bash
set -euo pipefail
ROOT="${PHANTOM_MERGE_ROOT}"
source "${ROOT}/scripts/_lib.sh"
activate_wz26b
export HF_HOME="${ROOT}/runtime/.cache/huggingface"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
cd "${ROOT}"
export PYTHONPATH="${ROOT}/active_code:${PYTHONPATH:-}"
python ccer/pipelines/run_prism_audit.py --device-map cpu --topk 50 "$@"
