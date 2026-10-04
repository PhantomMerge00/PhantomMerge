#!/usr/bin/env bash
# Re-run dual-metric n=18 with fixed generation scope + 384 tokens (needs ~64GB GPU).
set -euo pipefail
ROOT="${PHANTOM_MERGE_ROOT}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${ROOT}/active_code"

# Auto-pick GPU(s) with >=65GB free total (multi-GPU shard OK), or verify user-set CUDA_VISIBLE_DEVICES.
# Do NOT hardcode GPU index — cuda index != safe default on shared nodes.
eval "$(python3 "${SCRIPT_DIR}/pick_gpu_vram.py" --min-gb 65)"

echo "[round8-dual] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} starting $(date -Is)"

python3 -u "${ROOT}/ccer/pipelines/run_p3_round8.py" \
  --stage dual-metric \
  --track CEM \
  --device-map auto \
  --pair-limit 18 \
  --probe-max-tokens 384 \
  --layer 32 \
  --rank 16

PYTHONPATH="${ROOT}/active_code" python3 "${ROOT}/ccer/pipelines/audit_answer_metrics.py" \
  --input "${ROOT}/results/p3/round8/dual_metric_n18.json" \
  --output "${ROOT}/results/p3/round8/dual_metric_n18_reaudit.json"

echo "[round8-dual] done $(date -Is) — see results/p3/round8/"
