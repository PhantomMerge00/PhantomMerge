#!/usr/bin/env bash
# Task H: claim_onset @ L49 claim attribution sweep (dual-GPU: physical 0+2)
set -euo pipefail
cd ${PHANTOM_MERGE_ROOT}
source ~/wz26b/bin/activate
export PYTHONPATH=active_code
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# Physical GPU 0 + 2 (~80GiB combined) for Qwen3-32B device_map=auto
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,2}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

STAGE="${1:-all}"
PAIR_LIMIT="${PAIR_LIMIT:-}"
FRESH="${FRESH:-}"
TASK_H_OUT="${CCER_TASK_H_DIR:-results/p3/task_h}"
mkdir -p "${TASK_H_OUT}"

EXTRA=()
if [[ -n "${PAIR_LIMIT}" ]]; then
  EXTRA+=(--pair-limit "${PAIR_LIMIT}")
fi
if [[ "${FRESH}" == "1" ]]; then
  EXTRA+=(--fresh)
fi

LOG="${TASK_H_OUT}/run_${STAGE}_$(date +%Y%m%d_%H%M%S).log"
if [[ "${STAGE}" == "sweep" || "${STAGE}" == "all" ]]; then
  ln -sfn "$(basename "${LOG}")" "${TASK_H_OUT}/full_sweep_384.log"
fi

python3 -u ccer/pipelines/run_p3_task_h.py \
  --stage "${STAGE}" \
  --device-map auto \
  --probe-max-tokens 384 \
  "${EXTRA[@]}" \
  2>&1 | tee "${LOG}"
