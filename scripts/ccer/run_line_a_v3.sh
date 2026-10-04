#!/usr/bin/env bash
# Line A v3: expand clean from 2k gold + probe with CEM/CAP/AH breakdown.
set -euo pipefail

ROOT="${PHANTOM_MERGE_ROOT}"
ACTIVE="${ROOT}/active_code"
OUT="${ROOT}/results/line_a/v3"
LOG="${OUT}/line_a_v3.log"
MODE="${LINE_A_V3_MODE:-shard}"

mkdir -p "${OUT}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1
cd "${ACTIVE}"

python -u -m ccer.pipelines.run_line_a_probe_v3 --force 2>&1 | tee "${OUT}/cohort_preview.log" || true

case "${MODE}" in
  shard)
    NUM_SHARDS="${LINE_A_NUM_SHARDS:-2}"
    mapfile -t GPUS < <(echo "${LINE_A_SHARD_GPUS:-2,4}" | tr ',' ' ')
    for ((i = 0; i < NUM_SHARDS; i++)); do
      gpu="${GPUS[$i]}"
      setsid env CUDA_VISIBLE_DEVICES="${gpu}" python -u -m ccer.pipelines.run_line_a_extract_v3_clean \
        --shared-gpu --gpu-reserve-gib "${LINE_A_GPU_RESERVE_GIB:-18}" \
        --shard-id "${i}" --num-shards "${NUM_SHARDS}" \
        >> "${LOG}.extract.s${i}" 2>&1 &
      echo $! >> "${OUT}/extract_pids.txt"
    done
    echo "[line_a_v3] clean extraction workers launched; monitor ${LOG}.extract.s*"
    ;;
  single)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
    python -u -m ccer.pipelines.run_line_a_extract_v3_clean --shared-gpu --gpu-reserve-gib 18 >> "${LOG}.extract" 2>&1
    python -u -m ccer.pipelines.run_line_a_probe_v3 2>&1 | tee -a "${LOG}.probe"
    ;;
esac
