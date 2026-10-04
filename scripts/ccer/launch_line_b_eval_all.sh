#!/usr/bin/env bash
# B 线后台启动器
#   默认 das-sweep（核心交付：DAS vs PCA + wrong_owner_donor，probe_max_tokens=96）
#   STAGE=eval-all 才跑 QA（慢，~2天；与主结论无关，可另开）
set -euo pipefail
cd ${PHANTOM_MERGE_ROOT}
export PYTHONPATH=${PHANTOM_MERGE_ROOT}
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

STAGE="${STAGE:-das-sweep}"
OUT="results/p3/round12_line_b"
mkdir -p "$OUT"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="$OUT/line_b_${STAGE}_${STAMP}.log"
PIDFILE="$OUT/line_b_eval_all.pid"

if [[ -f "$PIDFILE" ]]; then
  old=$(cat "$PIDFILE")
  if kill -0 "$old" 2>/dev/null; then
    echo "already running pid=$old log=$(ls -t $OUT/line_b_*.log | head -1)"
    exit 0
  fi
fi

# shared-gpu + reserve=6: ~57GiB weights on GPU (L0-57), L58-63 CPU spill.
# Do NOT use --no-shared-gpu while hugh (~29GiB) shares nvidia-smi GPU1.
setsid python3 -u ccer/pipelines/run_p3_round12_line_b.py \
  --stage "$STAGE" \
  --pool mechanism_research \
  --ranks 16,32,64 \
  --das-steps 500 \
  --probe-max-tokens 96 \
  --device-map auto \
  --shared-gpu \
  --gpu-reserve-gib "${GPU_RESERVE_GIB:-6}" \
  >> "$LOG" 2>&1 &
echo $! > "$PIDFILE"
echo "started pid=$(cat "$PIDFILE") stage=$STAGE log=$LOG CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
