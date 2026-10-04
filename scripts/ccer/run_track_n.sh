#!/usr/bin/env bash
set -euo pipefail
ROOT="${PHANTOM_MERGE_ROOT}"
source "${ROOT}/scripts/_lib.sh"
activate_wz26b
export HF_HOME="${ROOT}/runtime/.cache/huggingface"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
mkdir -p "${ROOT}/results/jspace/track_n"
cd "${ROOT}"
python ccer/pipelines/run_track_n_jspace_swap.py \
  --device-map "cuda:0" \
  "$@"
