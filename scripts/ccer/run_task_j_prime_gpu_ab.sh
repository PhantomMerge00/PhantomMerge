#!/usr/bin/env bash
# Task J′ GPU A/B: re-run Task H sweep with live anchor enhancement, then compare patch_tier.
#
# Prerequisites:
#   1. Baseline Task H sweep finished in results/p3/task_h/
#   2. GPU 0+2 free (same as baseline H sweep)
#
# Usage:
#   bash scripts/ccer/run_task_j_prime_gpu_ab.sh          # sweep + compare
#   bash scripts/ccer/run_task_j_prime_gpu_ab.sh compare  # compare only
set -euo pipefail
cd ${PHANTOM_MERGE_ROOT}
source ~/wz26b/bin/activate
export PYTHONPATH=active_code
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,2}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

STAGE="${1:-all}"
BASELINE_DIR="results/p3/task_h"
J_PRIME_DIR="results/p3/task_h_j_prime"
AUDIT_DIR="results/p3/round12_line_b/task_j_prime"

mkdir -p "${J_PRIME_DIR}" "${AUDIT_DIR}"

if [[ "${STAGE}" == "all" || "${STAGE}" == "sweep" ]]; then
  BASELINE_N="$(wc -l < "${BASELINE_DIR}/claim_onset_pair_details.jsonl" 2>/dev/null || echo 0)"
  if [[ "${BASELINE_N}" -lt 57 ]]; then
    echo "[j_prime_gpu_ab] WARN: baseline pair_details=${BASELINE_N} (<57). Wait for H sweep to finish."
    echo "[j_prime_gpu_ab] tail -f ${BASELINE_DIR}/../full_sweep_384.log"
    exit 1
  fi

  echo "[j_prime_gpu_ab] snapshot baseline tier stats..."
  cp -f "${BASELINE_DIR}/claim_onset_pair_details.jsonl" \
    "${AUDIT_DIR}/baseline_pair_details_snapshot.jsonl"

  LOG="${AUDIT_DIR}/j_prime_h_sweep_$(date +%Y%m%d_%H%M%S).log"
  echo "[j_prime_gpu_ab] starting J′ H sweep -> ${J_PRIME_DIR} (CCER_J_PRIME_LIVE_ANCHOR=1)"
  export CCER_J_PRIME_LIVE_ANCHOR=1
  export CCER_TASK_H_DIR="${J_PRIME_DIR}"
  FRESH=1 bash scripts/ccer/run_task_h_claim_onset.sh sweep \
    2>&1 | tee "${LOG}"

  CCER_J_PRIME_LIVE_ANCHOR=1 CCER_TASK_H_DIR="${J_PRIME_DIR}" \
    python3 -u ccer/pipelines/run_p3_task_h.py --stage report
fi

if [[ "${STAGE}" == "all" || "${STAGE}" == "compare" ]]; then
  python3 -u scripts/ccer/compare_j_prime_patch_tier.py \
    --baseline-dir "${BASELINE_DIR}" \
    --j-prime-dir "${J_PRIME_DIR}"
fi
