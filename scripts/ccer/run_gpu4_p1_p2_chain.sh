#!/usr/bin/env bash
# P1 (incremental/resume) → Tier-1 go summary → slim P2 only on GO.
set -uo pipefail
export ROOT="${PHANTOM_MERGE_ROOT}"
export PHANTOM_MERGE_ROOT="${ROOT}"
export PHANTOM_LOG_ROOT="${ROOT}/data/logs/ops"
cd "${ROOT}"
# shellcheck source=/dev/null
source "${ROOT}/scripts/_lib.sh"
activate_wz26b
export PYTHONPATH="${ROOT}/active_code"

P1_RESUME="${P1_RESUME:-1}"
P1_FRESH="${P1_FRESH:-1}"
P2_LIMIT="${P2_LIMIT:-16}"

DAY="$(date +%Y%m%d)"
LOGDIR="${ROOT}/data/logs/ops/${DAY}"
mkdir -p "${LOGDIR}"
STAMP="$(date +%H%M%S)"
P1_LOG="${LOGDIR}/ccer_p1_dev_full_${STAMP}.log"
P2_LOG="${LOGDIR}/ccer_p2_dev_slim_${STAMP}.log"
META="${LOGDIR}/ccer_gpu4_chain_${STAMP}.meta"

{
  echo "host=$(hostname)"
  echo "started=$(date -Iseconds)"
  echo "pid=$$"
  echo "vllm_port=8003"
  echo "p1_resume=${P1_RESUME}"
  echo "p1_fresh=${P1_FRESH}"
  echo "p2_limit=${P2_LIMIT}"
} > "${META}"

preflight_vllm 8003 "Shopping Qwen3-32B" | tee -a "${META}"

P1_ARGS=(--resume)
if [[ "${P1_FRESH}" == "1" ]]; then
  P1_ARGS=(--fresh --no-resume)
fi

echo "=== P1 full dev cohort (CEM+CAP, incremental checkpoint) ===" | tee -a "${P1_LOG}"
if ! python3 -u "${ROOT}/ccer/pipelines/run_p1_dev.py" "${P1_ARGS[@]}" 2>&1 | tee -a "${P1_LOG}"; then
  echo "P1_FAILED exit=$?" | tee -a "${META}"
  exit 1
fi

echo "=== P1 Tier-1 go/no-go summary ===" | tee -a "${META}"
GO_DECISION="$(
  python3 -u "${ROOT}/ccer/pipelines/p1_go_summary.py" --query decision 2>&1 | tee -a "${META}" | tail -1
)"
echo "P1_GO_DECISION=${GO_DECISION}" | tee -a "${META}"

if [[ "${GO_DECISION}" != "GO" ]]; then
  echo "=== P2 SKIPPED (${GO_DECISION}) ===" | tee -a "${META}"
  echo "finished=$(date -Iseconds)" >> "${META}"
  echo "=== CHAIN DONE (P1 only, no P2) $(date -Iseconds) ===" | tee -a "${META}"
  exit 0
fi

echo "=== P2 slim dev (limit=${P2_LIMIT}, GO confirmed) ===" | tee -a "${P2_LOG}"
if ! python3 -u "${ROOT}/ccer/pipelines/run_p2_dev.py" --limit "${P2_LIMIT}" 2>&1 | tee -a "${P2_LOG}"; then
  echo "P2_FAILED exit=$?" | tee -a "${META}"
  exit 1
fi

echo "finished=$(date -Iseconds)" >> "${META}"
echo "=== ALL DONE $(date -Iseconds) ===" | tee -a "${META}"
