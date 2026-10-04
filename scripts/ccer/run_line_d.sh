#!/usr/bin/env bash
# Line D: cross-position causal contrast (internal design note)
set -euo pipefail
ROOT="${PHANTOM_MERGE_ROOT}"
cd "$ROOT"
export PYTHONPATH="${ROOT}/active_code"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
PYTHON_BIN="${PYTHON_BIN:-${PHANTOM_MERGE_HOME}/wz26b/bin/python3}"
LOG_DIR="${ROOT}/data/logs/ops/$(date +%Y%m%d)"
mkdir -p "$LOG_DIR"
LOG="${LOG_DIR}/line_d_$(date +%H%M%S).log"
echo "[line_d] log → $LOG"
while true; do
  GPU_EXPORT="$(python3 scripts/ccer/pick_gpu_vram.py --min-gb 65 2>/dev/null | tail -1)"
  if [[ "${GPU_EXPORT}" == export\ CUDA_VISIBLE_DEVICES=* ]]; then
    eval "${GPU_EXPORT}"
    break
  fi
  echo "[line_d] waiting for >=65GB free GPU..." | tee -a "$LOG"
  sleep 120
done
echo "[line_d] ${GPU_EXPORT}" | tee -a "$LOG"
"${PYTHON_BIN}" -u ccer/pipelines/run_p3_line_d.py \
  --stage activation-refresh \
  --track CEM \
  --device-map auto \
  2>&1 | tee -a "$LOG"
bash scripts/ccer/run_line_d_sweep_sequential.sh 2>&1 | tee -a "$LOG"
