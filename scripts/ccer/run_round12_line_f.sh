#!/usr/bin/env bash
# Line F: shell-anchor expanded evidence mask + wrong_owner_donor control.
set -euo pipefail
ROOT="${PHANTOM_MERGE_ROOT}"
export PYTHONPATH="${ROOT}/active_code"
PY="${PY:-${PHANTOM_MERGE_HOME}/wz26b/bin/python3}"
if [[ ! -x "${PY}" ]]; then
  PY="$(command -v python3)"
fi
if [[ ! -x "${PY}" ]]; then
  echo "[line_f] ERROR: python not found" >&2
  exit 1
fi

# Prefer idle whole-GPU cards (1 or 3); override: CUDA_VISIBLE_DEVICES=3 STAGE=shell-wrong-donor bash ...
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

STAGE="${STAGE:-all}"
SHARED_GPU="${SHARED_GPU:-0}"  # 0 = whole GPU (recommended); 1 = shared/offload (slow)
LOGDIR="${ROOT}/results/p3/round12_line_f"
mkdir -p "${LOGDIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="${LOGDIR}/line_f_${STAGE}_${STAMP}.log"

GPU_ARGS=()
if [[ "${SHARED_GPU}" == "1" ]]; then
  GPU_ARGS+=(--shared-gpu)
else
  GPU_ARGS+=(--no-shared-gpu)
fi

{
  echo "[line_f] PY=${PY}"
  echo "[line_f] STAGE=${STAGE}"
  echo "[line_f] SHARED_GPU=${SHARED_GPU}"
  echo "[line_f] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF} $(date -Is)"
  "${PY}" -u "${ROOT}/ccer/pipelines/run_p3_round12_line_f.py" \
    --stage "${STAGE}" \
    --pool mechanism_research \
    --device-map cuda:0 \
    "${GPU_ARGS[@]}" \
    --probe-max-tokens 384 \
    --layer 32 \
    --rank 16 \
    --alpha 1.0 \
    ${RESUME:+--resume}
  echo "[line_f] done $(date -Is)"
} >> "${LOG}" 2>&1

echo "[line_f] finished stage=${STAGE} log=${LOG}"
