#!/usr/bin/env bash
# Line B v3 DAS sweep on GPU4: incremental checkpoint + progress logs.
set -euo pipefail
cd ${PHANTOM_MERGE_ROOT}
source ${PHANTOM_MERGE_HOME}/wz26b/bin/activate
export PYTHONPATH="${PWD}/active_code"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

OUT="results/p3/round12_line_b"
PIPE="ccer/pipelines/run_p3_round12_line_b.py"
CKPT="$OUT/das_sweep_checkpoint_v3.json"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="$OUT/line_b_v3_sweep_gpu4_${STAMP}.log"
PIDFILE="$OUT/line_b_v3_sweep_gpu4.pid"

mkdir -p "$OUT"

echo "[line_b v3 sweep] GPU=$CUDA_VISIBLE_DEVICES ckpt=$CKPT log=$LOG"
echo "[line_b v3 sweep] resume: re-run this script; checkpoint is per-pair incremental"

setsid env \
  CUDA_DEVICE_ORDER=PCI_BUS_ID \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}" \
  python3 -u "$PIPE" \
    --stage das-sweep \
    --pool mechanism_research \
    --ranks "${RANKS:-16,32,64}" \
    --das-steps "${DAS_STEPS:-500}" \
    --probe-max-tokens "${PROBE_MAX_TOKENS:-384}" \
    --device-map cuda:0 \
    --no-shared-gpu \
    --checkpoint "$CKPT" \
    --sweep-out "$OUT/das_vs_pca_rank_sweep_v3.json" \
  >> "$LOG" 2>&1 &

echo $! > "$PIDFILE"
echo "started pid=$(cat "$PIDFILE")"
echo "monitor: tail -f $LOG"
echo "checkpoint: watch -n30 'ls -lh $CKPT 2>/dev/null; jq .completed_rank_rows[].rank $CKPT 2>/dev/null'"
