#!/usr/bin/env bash
# ROUND5 dual-GPU: shard0@GPU1 + shard1@GPU4, then merge.
set -euo pipefail
ROOT="${PHANTOM_MERGE_ROOT}"
cd "${ROOT}"
export PYTHONPATH="${ROOT}/active_code"
PY="${PY:-${PHANTOM_MERGE_HOME}/wz26b/bin/python3}"
LOGDIR="${ROOT}/data/logs/ops/$(date +%Y%m%d)"
mkdir -p "${LOGDIR}"
STAMP="$(date +%H%M%S)"
GPU0="${ROUND5_GPU0:-1}"
GPU1="${ROUND5_GPU1:-4}"
NUM_SHARDS=2
ARGS=(--track CEM --layer 32 --position commitment --ranks 16,32,44
  --probe-max-tokens 96 --num-shards "${NUM_SHARDS}")

pkill -f "run_p3_rank_sweep" 2>/dev/null || true
sleep 2

_log() { echo "[$(date -Iseconds)] $*"; }

_log "shard0 GPU${GPU0} (pairs even + positions commitment,claim_onset)"
setsid env CUDA_VISIBLE_DEVICES="${GPU0}" PYTHONPATH="${ROOT}/active_code" \
  "${PY}" -u ccer/pipelines/run_p3_rank_sweep.py \
  "${ARGS[@]}" --shard-index 0 \
  > "${LOGDIR}/round5_s0_${STAMP}.log" 2>&1 < /dev/null &
PID0=$!
echo "${PID0}" > "${LOGDIR}/round5_s0.pid"

_log "stagger 45s before shard1 GPU${GPU1}"
sleep 45

_log "shard1 GPU${GPU1} (pairs odd + positions pre_value,prompt_end)"
setsid env CUDA_VISIBLE_DEVICES="${GPU1}" PYTHONPATH="${ROOT}/active_code" \
  "${PY}" -u ccer/pipelines/run_p3_rank_sweep.py \
  "${ARGS[@]}" --shard-index 1 \
  > "${LOGDIR}/round5_s1_${STAMP}.log" 2>&1 < /dev/null &
PID1=$!
echo "${PID1}" > "${LOGDIR}/round5_s1.pid"

_log "waiting shards pid0=${PID0} pid1=${PID1}"
wait "${PID0}"; EC0=$?
wait "${PID1}"; EC1=$?
_log "shards done ec0=${EC0} ec1=${EC1}"

"${PY}" -u ccer/pipelines/run_p3_rank_sweep.py --merge-shards --num-shards "${NUM_SHARDS}" \
  2>&1 | tee "${LOGDIR}/round5_merge_${STAMP}.log"
_log "ROUND5 complete STAMP=${STAMP}"
