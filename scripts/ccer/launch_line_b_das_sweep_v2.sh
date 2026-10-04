#!/usr/bin/env bash
# Line B v2 sweep: Line A ROI (L49/claim_onset) + methodology fixes (see line_b_protocol.py).
# Uses separate checkpoint — does NOT resume v1 L32/commitment sweep.
set -euo pipefail
cd ${PHANTOM_MERGE_ROOT}
export PYTHONPATH="${PWD}/active_code"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

OUT="results/p3/round12_line_b"
PIPE="ccer/pipelines/run_p3_round12_line_b.py"
mkdir -p "$OUT"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="$OUT/line_b_das-sweep_v2_${STAMP}.log"
PIDFILE="$OUT/line_b_das_sweep_v2.pid"
CKPT="$OUT/das_sweep_checkpoint_v2.json"

echo "[line_b v2] CUDA=$CUDA_VISIBLE_DEVICES ROI=L49/claim_onset ckpt=$CKPT log=$LOG"

# Phase 1: merge L49 dense layers into existing activation cache (required for v2 ROI).
EXTRACT_LOG="$OUT/line_b_extract_roi_layers_v2_${STAMP}.log"
echo "[line_b v2] phase 1: extract-activations -> $EXTRACT_LOG"
python3 -u "$PIPE" \
  --stage extract-activations \
  --pool mechanism_research \
  --device-map auto \
  --no-shared-gpu \
  >> "$EXTRACT_LOG" 2>&1

echo "[line_b v2] phase 2: das-sweep"
setsid python3 -u "$PIPE" \
  --stage das-sweep \
  --pool mechanism_research \
  --ranks "${RANKS:-16,32,64}" \
  --das-steps 500 \
  --probe-max-tokens 96 \
  --device-map auto \
  --no-shared-gpu \
  --checkpoint "$CKPT" \
  >> "$LOG" 2>&1 &
echo $! > "$PIDFILE"
echo "started v2 pid=$(cat $PIDFILE) log=$LOG"
