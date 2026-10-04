#!/usr/bin/env bash
# ROUND11: shell-only wrong_owner_donor control (strong mask held constant).
set -euo pipefail
ROOT="${PHANTOM_MERGE_ROOT}"
export PYTHONPATH="${ROOT}/active_code"
PY="${PY:-${PHANTOM_MERGE_HOME}/wz26b/bin/python3}"

if [[ ! -x "${PY}" ]]; then
  echo "[round11] ERROR: python not found: ${PY}" >&2
  exit 1
fi

# CUDA index != nvidia-smi index on this host.
# Empty 97GB PRO 6000 = nvidia-smi GPU 4 = CUDA_VISIBLE_DEVICES=3.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"

LOGDIR="${ROOT}/results/p3/round11"
mkdir -p "${LOGDIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="${LOGDIR}/shell_wrong_owner_${STAMP}.log"

{
  echo "[round11] PY=${PY}"
  echo "[round11] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} $(date -Is)"
  "${PY}" -u "${ROOT}/ccer/pipelines/run_p3_round9.py" \
    --stage shell-wrong-donor \
    --track CEM \
    --device-map cuda:0 \
    --probe-max-tokens 384 \
    --layer 32 \
    --rank 16 \
    --alpha 1.0
  echo "[round11] done $(date -Is)"
} >> "${LOG}" 2>&1

echo "[round11] finished log=${LOG}"
