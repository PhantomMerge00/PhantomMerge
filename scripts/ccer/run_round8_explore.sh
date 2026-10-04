#!/usr/bin/env bash
# ROUND8 exploration: position sweep + DAS + path patch (single model load, multi-GPU).
set -euo pipefail
ROOT="${PHANTOM_MERGE_ROOT}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${ROOT}/active_code"

# Auto-pick multi-GPU combo (sum >=65GB). Override: CUDA_VISIBLE_DEVICES=4,0,2
eval "$(python3 "${SCRIPT_DIR}/pick_gpu_vram.py" --min-gb 65)"

LOG="${ROOT}/results/p3/round8/explore_run.log"
mkdir -p "$(dirname "$LOG")"

echo "[round8-explore] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} starting $(date -Is)" | tee -a "$LOG"

python3 -u "${ROOT}/ccer/pipelines/run_p3_round8.py" \
  --stage explore \
  --track CEM \
  --device-map auto \
  --pair-limit 18 \
  --probe-max-tokens 384 \
  --layer 32 \
  --rank 16 \
  --path-patch-limit 4 \
  2>&1 | tee -a "$LOG"

echo "[round8-explore] done $(date -Is)" | tee -a "$LOG"
