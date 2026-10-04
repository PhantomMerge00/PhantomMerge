#!/usr/bin/env bash
# Line B v3: seven root-cause closures + extreme enhancement sweep.
# Protocol: line_b_v3_root_cause_closed (L49/claim_onset, use_cache=False, quality filter, layer scan)
set -euo pipefail
cd ${PHANTOM_MERGE_ROOT}
source ${PHANTOM_MERGE_HOME}/wz26b/bin/activate
export PYTHONPATH="${PWD}/active_code"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

OUT="results/p3/round12_line_b"
PIPE="ccer/pipelines/run_p3_round12_line_b.py"
mkdir -p "$OUT"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="$OUT/line_b_v3_extreme_${STAMP}.log"
PIDFILE="$OUT/line_b_v3_extreme.pid"
CKPT="$OUT/das_sweep_checkpoint_v3.json"

echo "[line_b v3] CUDA=$CUDA_VISIBLE_DEVICES log=$LOG ckpt=$CKPT"

# Phase 0: CPU preflight (writes preflight_v3.json)
echo "[line_b v3] phase 0: preflight"
python3 -u "$PIPE" --stage preflight --pool mechanism_research || {
  echo "[line_b v3] preflight blocked — will run extract then retry preflight"
}

# Phase 1: symmetric Line D v3 refresh + ROI layer merge (claim_onset gate)
EXTRACT_LOG="$OUT/line_b_extract_v3_symmetric_${STAMP}.log"
echo "[line_b v3] phase 1: extract-activations (symmetric v3) -> $EXTRACT_LOG"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python3 -u "$PIPE" \
  --stage extract-activations \
  --pool mechanism_research \
  --device-map cuda:0 \
  --no-shared-gpu \
  --min-eligible-pairs 0 \
  >> "$EXTRACT_LOG" 2>&1

# Phase 1b: re-preflight after extract (must pass before sweep)
echo "[line_b v3] phase 1b: preflight (post-extract)"
python3 -u "$PIPE" --stage preflight --pool mechanism_research

# Phase 2: DAS vs PCA rank sweep (v3 checkpoint, auto layer scan, quality-filtered eval)
echo "[line_b v3] phase 2: das-sweep -> $LOG"
setsid env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python3 -u "$PIPE" \
  --stage das-sweep \
  --pool mechanism_research \
  --ranks "${RANKS:-16,32,64}" \
  --das-steps 500 \
  --probe-max-tokens 96 \
  --device-map cuda:0 \
  --no-shared-gpu \
  --checkpoint "$CKPT" \
  --sweep-out "$OUT/das_vs_pca_rank_sweep_v3.json" \
  >> "$LOG" 2>&1 &
echo $! > "$PIDFILE"
echo "started v3 pid=$(cat "$PIDFILE") log=$LOG"
echo "monitor: tail -f $LOG"
