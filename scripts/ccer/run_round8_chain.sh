#!/usr/bin/env bash
# ROUND8: dual-metric n=18 + activation refresh + position/DAS/path pilots
set -euo pipefail

ROOT="${PHANTOM_MERGE_ROOT}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${ROOT}/active_code"

eval "$(python3 "${SCRIPT_DIR}/pick_gpu_vram.py" --min-gb 65)"

LOG="${ROOT}/results/p3/round8/run.log"
mkdir -p "$(dirname "$LOG")"

echo "[round8] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} starting $(date -Is)" | tee -a "$LOG"

python3 -u "${ROOT}/ccer/pipelines/run_p3_round8.py" \
  --stage all \
  --track CEM \
  --device-map auto \
  --pair-limit 18 \
  --probe-max-tokens 384 \
  --layer 32 \
  --rank 16 \
  --path-patch-limit 4 \
  2>&1 | tee -a "$LOG"

echo "[round8] done $(date -Is)" | tee -a "$LOG"
