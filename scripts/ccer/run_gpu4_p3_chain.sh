#!/usr/bin/env bash
# P3 chain: stop vLLM → P3-0 → P3a → P3b on GPU4 (HF exclusive).
set -uo pipefail
export ROOT="${PHANTOM_MERGE_ROOT}"
export PHANTOM_MERGE_ROOT="${ROOT}"
export PHANTOM_LOG_ROOT="${ROOT}/data/logs/ops"
cd "${ROOT}"
# shellcheck source=/dev/null
source "${ROOT}/scripts/_lib.sh"
activate_wz26b
export PYTHONPATH="${ROOT}/active_code"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"

# Default: full 56 CAP pairs. Export P3_LIMIT=N before invoke to slim.
: "${P3_LIMIT:=}"
P3_ALPHA_SWEEP="${P3_ALPHA_SWEEP:-0}"
RESTART_VLLM="${RESTART_VLLM:-1}"

DAY="$(date +%Y%m%d)"
LOGDIR="${ROOT}/data/logs/ops/${DAY}"
mkdir -p "${LOGDIR}"
STAMP="$(date +%H%M%S)"
P30_LOG="${LOGDIR}/ccer_p3_0_${STAMP}.log"
P3A_LOG="${LOGDIR}/ccer_p3a_${STAMP}.log"
P3B_LOG="${LOGDIR}/ccer_p3b_${STAMP}.log"
META="${LOGDIR}/ccer_gpu4_p3_chain_${STAMP}.meta"

{
  echo "host=$(hostname)"
  echo "started=$(date -Iseconds)"
  echo "pid=$$"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
  echo "p3_limit=${P3_LIMIT}"
  echo "p3_alpha_sweep=${P3_ALPHA_SWEEP}"
} > "${META}"

stop_vllm_if_running() {
  if pgrep -f "vllm.*8003" >/dev/null 2>&1; then
    echo "Stopping vLLM on port 8003 for HF exclusive P3..." | tee -a "${META}"
    pkill -f "vllm.*8003" || true
    sleep 15
  fi
}

stop_vllm_if_running

P30_ARGS=(--device-map auto)
if [[ -n "${P3_LIMIT}" ]]; then
  P30_ARGS+=(--limit "${P3_LIMIT}")
fi

echo "=== P3-0 teacher-forced + activation extraction ===" | tee -a "${P30_LOG}"
if ! python3 -u "${ROOT}/ccer/pipelines/run_p3_0.py" "${P30_ARGS[@]}" 2>&1 | tee -a "${P30_LOG}"; then
  echo "P3_0_FAILED exit=$?" | tee -a "${META}"
  exit 1
fi

echo "=== P3a binding ROI / owner PCA ===" | tee -a "${P3A_LOG}"
if ! python3 -u "${ROOT}/ccer/pipelines/run_p3a.py" 2>&1 | tee -a "${P3A_LOG}"; then
  echo "P3A_FAILED exit=$?" | tee -a "${META}"
  exit 1
fi

P3B_ARGS=(--device-map auto)
if [[ -n "${P3_LIMIT}" ]]; then
  P3B_ARGS+=(--limit "${P3_LIMIT}")
fi
if [[ "${P3_ALPHA_SWEEP}" == "1" ]]; then
  P3B_ARGS+=(--alpha-sweep)
fi

echo "=== P3b interchange + IIA ===" | tee -a "${P3B_LOG}"
if ! python3 -u "${ROOT}/ccer/pipelines/run_p3b.py" "${P3B_ARGS[@]}" 2>&1 | tee -a "${P3B_LOG}"; then
  echo "P3B_FAILED exit=$?" | tee -a "${META}"
  exit 1
fi

if [[ "${RESTART_VLLM}" == "1" ]]; then
  echo "Restarting vLLM (background)..." | tee -a "${META}"
  nohup bash "${ROOT}/scripts/vllm/shopping_agent_qwen32b_gpu0.sh" >> "${LOGDIR}/vllm_restart_${STAMP}.log" 2>&1 &
fi

echo "finished=$(date -Iseconds)" >> "${META}"
echo "=== P3 ALL DONE $(date -Iseconds) ===" | tee -a "${META}"
