#!/usr/bin/env bash
# Line D v3: symmetric activation refresh + four-position sweep (one position per process).
set -euo pipefail
ROOT="${PHANTOM_MERGE_ROOT}"
cd "$ROOT"
export PYTHONPATH="${ROOT}/active_code"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
PYTHON_BIN="${PYTHON_BIN:-${PHANTOM_MERGE_HOME}/wz26b/bin/python3}"
LOG_DIR="${ROOT}/data/logs/ops/$(date +%Y%m%d)"
mkdir -p "$LOG_DIR"

unset CUDA_VISIBLE_DEVICES
while true; do
  GPU_EXPORT="$(python3 scripts/ccer/pick_gpu_vram.py --min-gb 65 2>/dev/null | tail -1 || true)"
  if [[ "${GPU_EXPORT}" == export\ CUDA_VISIBLE_DEVICES=* ]]; then
    eval "${GPU_EXPORT}"
    break
  fi
  echo "[line_d_sweep] waiting for >=65GB free GPU (multi-GPU OK)..."
  sleep 120
done
echo "[line_d_sweep] ${GPU_EXPORT}"

REFRESH_LOG="${LOG_DIR}/line_d_activation_refresh_$(date +%H%M%S).log"
echo "[line_d_sweep] === activation-refresh (symmetric v3, force) === log → ${REFRESH_LOG}"
"${PYTHON_BIN}" -u ccer/pipelines/run_p3_line_d.py \
  --stage activation-refresh \
  --force-refresh \
  --device-map auto \
  2>&1 | tee "${REFRESH_LOG}"

SWEEP_JSON="${ROOT}/results/p3/line_d/position_sweep_controls.json"
rm -f "${SWEEP_JSON}"
echo "[line_d_sweep] removed stale ${SWEEP_JSON}"

for POS in prompt_end commitment claim_onset pre_value; do
  LOG="${LOG_DIR}/line_d_sweep_${POS}_$(date +%H%M%S).log"
  echo "[line_d_sweep] === ${POS} === log → ${LOG}"
  "${PYTHON_BIN}" -u ccer/pipelines/run_p3_line_d.py \
    --stage position-one \
    --position "${POS}" \
    --device-map auto \
    --probe-max-tokens 384 \
    --layer 32 \
    --claim-layer 49 \
    --rank 16 \
    2>&1 | tee "${LOG}"
done

"${PYTHON_BIN}" -u ccer/pipelines/run_p3_line_d.py --stage report
echo "[line_d_sweep] done"
