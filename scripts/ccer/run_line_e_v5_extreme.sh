#!/usr/bin/env bash
# Line E v5 extreme in-generation mitigation (B interchange + probe steer + anchor logit bias)
set -euo pipefail

ROOT="${PHANTOM_MERGE_ROOT}"
VENV="${PHANTOM_MERGE_HOME}/wz26b/bin/activate"
GPU="${CUDA_VISIBLE_DEVICES:-3}"

source "$VENV"
export PYTHONPATH="${ROOT}/active_code"
export CUDA_VISIBLE_DEVICES="$GPU"

cd "$ROOT"

python -m ccer.pipelines.run_p3_line_e \
  --steering-v5 \
  --use-cache \
  --device-map cuda:0 \
  --pool mechanism_research \
  --holdout-frac 0.35 \
  --seed 42 \
  "$@"
