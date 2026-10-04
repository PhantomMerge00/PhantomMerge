#!/usr/bin/env bash
# Wait for base CAP P3b, then run ROUND2 GPU chain.
set -uo pipefail
ROOT="${PHANTOM_MERGE_ROOT}"
P3B_PID="${1:-1079248}"
LOG="${ROOT}/data/logs/ops/20260913/round2_wait_p3b.log"
echo "waiting for P3b pid=${P3B_PID} $(date -Is)" >> "${LOG}"
while kill -0 "${P3B_PID}" 2>/dev/null; do
  wc -l "${ROOT}/results/p3/interchange_rows.jsonl" >> "${LOG}" 2>/dev/null || true
  sleep 60
done
echo "P3b exited, starting round2 chain $(date -Is)" >> "${LOG}"
bash "${ROOT}/scripts/ccer/run_gpu4_round2_chain.sh" >> "${LOG}" 2>&1
echo "round2 chain done $(date -Is)" >> "${LOG}"
