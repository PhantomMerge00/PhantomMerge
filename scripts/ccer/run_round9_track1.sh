#!/usr/bin/env bash
# ROUND10 CORE ONLY: strong evidence-mask + commitment patch (n=22 native pairs).
set -euo pipefail
ROOT="${PHANTOM_MERGE_ROOT}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${ROOT}/active_code"
PY="${PY:-python3}"
LOGDIR="${ROOT}/results/p3/round9"
mkdir -p "${LOGDIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="${LOGDIR}/strong_mask_${STAMP}.log"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  if "${PY}" "${SCRIPT_DIR}/pick_gpu_vram.py" --min-gb 65 >/tmp/round10_gpu_pick.sh 2>/dev/null; then
    eval "$(cat /tmp/round10_gpu_pick.sh)"
  else
    export CUDA_VISIBLE_DEVICES=4,1,3
    echo "[round10] pick_gpu fallback → CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >&2
  fi
fi

echo "[round10] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} STRONG mask only $(date -Is)" | tee "${LOG}"

"${PY}" -u "${ROOT}/ccer/pipelines/run_p3_round9.py" \
  --stage evidence-mask \
  --track CEM \
  --device-map auto \
  --probe-max-tokens 384 \
  --layer 32 \
  --rank 16 \
  --alpha 1.0 \
  2>&1 | tee -a "${LOG}"

echo "[round10] done $(date -Is)" | tee -a "${LOG}"
