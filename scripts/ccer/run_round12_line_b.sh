#!/usr/bin/env bash
# Line B: DAS full-pool training + rank sweep (mechanism_research pool).
set -euo pipefail
cd ${PHANTOM_MERGE_ROOT}
export PYTHONPATH=${PHANTOM_MERGE_ROOT}
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
PY="${PY:-python3}"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="results/p3/round12_line_b/line_b_${STAMP}.log"
mkdir -p results/p3/round12_line_b

echo "[line_b] starting stage=all log=$LOG"
nohup "$PY" -u ccer/pipelines/run_p3_round12_line_b.py \
  --stage all \
  --pool mechanism_research \
  --ranks 16,32,64 \
  --das-steps 500 \
  > "$LOG" 2>&1 &
echo "[line_b] pid=$! log=$LOG"
