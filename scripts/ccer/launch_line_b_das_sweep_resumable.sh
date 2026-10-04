#!/usr/bin/env bash
# Line B das-sweep: pair-level checkpoint + dual Ada GPUs (CUDA 0,1 ~70GiB).
set -euo pipefail
cd ${PHANTOM_MERGE_ROOT}
export PYTHONPATH="${PWD}/active_code"
# nvidia-smi GPU4 (PRO 6000 ~97GiB) → CUDA_VISIBLE_DEVICES=3 on this host
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

OUT="results/p3/round12_line_b"
PIPE="ccer/pipelines/run_p3_round12_line_b.py"
mkdir -p "$OUT"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="$OUT/line_b_das-sweep_resumable_${STAMP}.log"
PIDFILE="$OUT/line_b_eval_all.pid"
CKPT="$OUT/das_sweep_checkpoint_v2.json"

if [[ -f "$PIDFILE" ]]; then
  old=$(cat "$PIDFILE")
  if kill -0 "$old" 2>/dev/null; then
    echo "[line_b] stopping pid=$old"
    kill "$old" 2>/dev/null || true
    sleep 2
  fi
fi

echo "[line_b] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES checkpoint=$CKPT log=$LOG"
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
echo "started pid=$(cat "$PIDFILE") log=$LOG"
