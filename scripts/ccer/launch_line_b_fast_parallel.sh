#!/usr/bin/env bash
# B 线 das-sweep：GPU2 Ada ~39GiB 空闲（Line E/D 占 GPU3/4 时用此脚本）
# 整 ranks 16,32,64 顺序跑；shared_gpu reserve=5 尽量多层在 GPU
set -euo pipefail
cd ${PHANTOM_MERGE_ROOT}
export PYTHONPATH=${PHANTOM_MERGE_ROOT}
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export GPU_RESERVE_GIB="${GPU_RESERVE_GIB:-5}"

OUT="results/p3/round12_line_b"
PIPE="ccer/pipelines/run_p3_round12_line_b.py"
mkdir -p "$OUT"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="$OUT/line_b_das-sweep_fast_${STAMP}.log"
PIDFILE="$OUT/line_b_eval_all.pid"

if [[ -f "$PIDFILE" ]]; then
  old=$(cat "$PIDFILE")
  if kill -0 "$old" 2>/dev/null; then
    echo "[line_b] stopping pid=$old"
    kill "$old" 2>/dev/null || true
    sleep 2
  fi
fi

setsid python3 -u "$PIPE" \
  --stage das-sweep \
  --pool mechanism_research \
  --ranks 16,32,64 \
  --das-steps 500 \
  --probe-max-tokens 96 \
  --device-map auto \
  --shared-gpu \
  --gpu-reserve-gib "$GPU_RESERVE_GIB" \
  >> "$LOG" 2>&1 &
echo $! > "$PIDFILE"
echo "started pid=$(cat $PIDFILE) GPU=$CUDA_VISIBLE_DEVICES log=$LOG"
