#!/usr/bin/env bash
# Task I: four-position simultaneous joint interchange patch
set -euo pipefail
ROOT="${PHANTOM_MERGE_ROOT}"
cd "$ROOT"
export PYTHONPATH="${ROOT}/active_code"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
PYTHON_BIN="${PYTHON_BIN:-${PHANTOM_MERGE_HOME}/wz26b/bin/python3}"
LOG_DIR="${ROOT}/data/logs/ops/$(date +%Y%m%d)"
mkdir -p "$LOG_DIR"
LOG="${LOG_DIR}/line_i_joint_$(date +%H%M%S).log"
echo "[line_i] log → $LOG"

STAGE="${1:-all}"
SHARD_INDEX="${SHARD_INDEX:-0}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SKIP_SMOKE="${SKIP_SMOKE:-0}"

while true; do
  GPU_EXPORT="$(python3 scripts/ccer/pick_gpu_vram.py --min-gb 65 2>/dev/null | tail -1)"
  if [[ "${GPU_EXPORT}" == export\ CUDA_VISIBLE_DEVICES=* ]]; then
    eval "${GPU_EXPORT}"
    break
  fi
  echo "[line_i] waiting for >=65GB free GPU..." | tee -a "$LOG"
  sleep 120
done
echo "[line_i] ${GPU_EXPORT}" | tee -a "$LOG"

if [[ "${SKIP_SMOKE}" != "1" && ("${STAGE}" == "all" || "${STAGE}" == "smoke") ]]; then
  "${PYTHON_BIN}" -u scripts/ccer/smoke_joint_patch.py \
    --device-map auto \
    --probe-max-tokens 96 \
    2>&1 | tee -a "$LOG"
fi

RUN_STAGE="${STAGE}"
if [[ "${STAGE}" == "all" ]]; then
  RUN_STAGE="run"
fi
if [[ "${STAGE}" == "smoke" ]]; then
  echo "[line_i] smoke-only done" | tee -a "$LOG"
  exit 0
fi

"${PYTHON_BIN}" -u ccer/pipelines/run_p3_line_i.py \
  --stage "${RUN_STAGE}" \
  --device-map auto \
  --probe-max-tokens 384 \
  --rank 16 \
  --shard-index "${SHARD_INDEX}" \
  --num-shards "${NUM_SHARDS}" \
  2>&1 | tee -a "$LOG"

if [[ "${RUN_STAGE}" == "run" || "${STAGE}" == "all" ]]; then
  "${PYTHON_BIN}" -u ccer/pipelines/run_p3_line_i.py \
    --stage report \
    2>&1 | tee -a "$LOG"
fi

echo "[line_i] done → ${LOG}" | tee -a "$LOG"
