#!/usr/bin/env bash
# Line E v4 ultimate: refresh activations → probe → full holdout eval
set -euo pipefail
ROOT="${PHANTOM_MERGE_ROOT}"
cd "$ROOT"
export PYTHONPATH=active_code
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Pick GPU with most free VRAM (default 0). Override: CUDA_VISIBLE_DEVICES=2 bash ...
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  BEST_GPU="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | sort -t, -k2 -nr | head -1 | cut -d, -f1 | tr -d ' ')"
  export CUDA_VISIBLE_DEVICES="${BEST_GPU}"
fi
echo "[line_e v4] Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

LOGDIR="${ROOT}/results/p3/line_e"
mkdir -p "$LOGDIR"
STAMP="$(date +%Y%m%d_%H%M%S)"

# A100/80GB+ empty GPU: load directly on GPU (shared-gpu = CPU offload, no GPU for 10+ min).
FREE_MIB="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${CUDA_VISIBLE_DEVICES%%,*}" | tr -d ' ')"
if [[ "${FREE_MIB}" -ge 70000 ]]; then
  DEV_MAP="cuda:0"
  SHARED_FLAG="--no-shared-gpu"
  echo "[line_e v4] GPU free=${FREE_MIB}MiB → device-map=cuda:0 (no CPU offload)"
else
  DEV_MAP="auto"
  SHARED_FLAG="--shared-gpu"
  echo "[line_e v4] GPU free=${FREE_MIB}MiB → device-map=auto --shared-gpu"
fi

echo "[line_e v4] Step 1: refresh missing bilateral claim_onset activations"
python3 -u ccer/pipelines/run_line_e_refresh_activations.py \
  --device-map "${DEV_MAP}" ${SHARED_FLAG} \
  2>&1 | tee "${LOGDIR}/activation_refresh_${STAMP}.log"

echo "[line_e v4] Step 2: 2-case ultimate probe"
python3 -u ccer/pipelines/run_line_e_probe_v3.py \
  --device-map "${DEV_MAP}" ${SHARED_FLAG} \
  2>&1 | tee "${LOGDIR}/probe_v4_${STAMP}.log"

echo "[line_e v4] Step 3: full holdout eval (--steering-v3 = v4 ultimate protocol)"
python3 -u ccer/pipelines/run_p3_line_e.py \
  --steering-v3 \
  --pool mechanism_research \
  --device-map "${DEV_MAP}" \
  2>&1 | tee "${LOGDIR}/steering_v4_${STAMP}.log"

echo "[line_e v4] Done. Summary: ${LOGDIR}/steering_summary_v3.json"
