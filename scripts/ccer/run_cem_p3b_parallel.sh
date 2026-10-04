#!/usr/bin/env bash
# CEM P3b: kill stuck jobs + optional GPU4 vLLM, then 3-way shard on free GPUs.
set -uo pipefail
ROOT="${PHANTOM_MERGE_ROOT}"
cd "${ROOT}"
# shellcheck source=/dev/null
source "${ROOT}/scripts/_lib.sh"
activate_wz26b
export PYTHONPATH="${ROOT}/active_code"

NUM_SHARDS="${NUM_SHARDS:-3}"
CONTROLS="${CONTROLS:-no_intervention,self_donor,target_interchange}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-384}"
KILL_GPU4_VLLM="${KILL_GPU4_VLLM:-1}"
DAY="$(date +%Y%m%d)"
LOGDIR="${ROOT}/data/logs/ops/${DAY}"
mkdir -p "${LOGDIR}"
STAMP="$(date +%H%M%S)"

echo "[p3b-parallel] stopping stuck run_p3b..."
pkill -f "python3 ccer/pipelines/run_p3b.py" 2>/dev/null || true
sleep 2

if [[ "${KILL_GPU4_VLLM}" == "1" ]]; then
  echo "[p3b-parallel] stopping GPU4 vLLM (port 8003)..."
  pkill -f "vllm.*8003" 2>/dev/null || true
  sleep 10
fi

# GPU assignment: A100 + two RTX PRO 6000 with headroom
GPUS=(3 1 4)
PIDS=()
for i in "${!GPUS[@]}"; do
  if [[ "${i}" -ge "${NUM_SHARDS}" ]]; then
    break
  fi
  gpu="${GPUS[$i]}"
  log="${LOGDIR}/ccer_p3b_cem_s${i}_gpu${gpu}_${STAMP}.log"
  echo "[p3b-parallel] shard ${i}/${NUM_SHARDS} -> GPU ${gpu} log=${log}"
  CUDA_VISIBLE_DEVICES="${gpu}" nohup python3 -u ccer/pipelines/run_p3b.py \
    --device-map auto \
    --track CEM \
    --shard-index "${i}" \
    --num-shards "${NUM_SHARDS}" \
    --controls "${CONTROLS}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    >"${log}" 2>&1 &
  PIDS+=("$!")
done

echo "[p3b-parallel] launched PIDs: ${PIDS[*]}"
wait "${PIDS[@]}"
echo "[p3b-parallel] shards done; merging..."
python3 -u ccer/pipelines/run_p3b.py \
  --track CEM \
  --merge-shards \
  --num-shards "${NUM_SHARDS}"
echo "[p3b-parallel] finished $(date -Iseconds)"
