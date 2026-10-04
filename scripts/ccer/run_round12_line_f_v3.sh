#!/usr/bin/env bash
# Line F v3 mechanism extreme: extended metrics + decode hook fix + multi-position sweep.
set -euo pipefail
ROOT="${ROOT:-${PHANTOM_MERGE_ROOT}}"
PY="${PY:-${PHANTOM_MERGE_HOME}/wz26b/bin/python3}"
LOGDIR="${ROOT}/results/p3/round12_line_f"
STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"

export PYTHONPATH="${ROOT}/active_code"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

STAGE="${1:-position-sweep-v3}"
GPU="${CUDA_VISIBLE_DEVICES:-1}"

echo "[line_f v3] STAGE=${STAGE} GPU=${GPU} ${STAMP}" | tee "${LOGDIR}/line_f_v3_${STAGE}_${STAMP}.log"

CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" -u "${ROOT}/ccer/pipelines/run_p3_round12_line_f.py" \
  --stage "${STAGE}" \
  --protocol v3 \
  --device-map cuda:0 \
  --no-shared-gpu \
  --alphas "0.5,1.0,1.5" \
  2>&1 | tee -a "${LOGDIR}/line_f_v3_${STAGE}_${STAMP}.log"

echo "[line_f v3] done → ${LOGDIR}/shell_wrong_owner_v3_position_sweep.json"
