#!/usr/bin/env bash
# Run AFTER main das-sweep completes (das_vs_pca_rank_sweep.json).
# Does NOT touch the in-flight main sweep checkpoint.
set -euo pipefail
cd ${PHANTOM_MERGE_ROOT}
export PYTHONPATH="${PWD}/active_code"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

OUT="results/p3/round12_line_b"
PIPE="ccer/pipelines/run_p3_round12_line_b.py"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="${OUT}/line_b_post_sweep_${STAMP}.log"
mkdir -p "$OUT"

if [[ ! -f "${OUT}/das_vs_pca_rank_sweep.json" ]]; then
  echo "[line_b] waiting for main sweep: ${OUT}/das_vs_pca_rank_sweep.json"
  echo "  tail -f ${OUT}/line_b_das-sweep_resumable_*.log"
  exit 1
fi

echo "[line_b] post-sweep CUDA=${CUDA_VISIBLE_DEVICES} log=${LOG}"

{
  echo "=== 1/6 extract missing activations ==="
  python3 -u "$PIPE" --stage extract-activations --pool mechanism_research --no-shared-gpu

  echo "=== 2/6 pair audit (expanded pool) ==="
  python3 -u "$PIPE" --stage pair-audit --pool mechanism_research

  echo "=== 3/6 QA evidence mask ==="
  python3 -u "$PIPE" --stage qa-evidence-mask --pool mechanism_research --no-shared-gpu

  echo "=== 4/6 QA wrong_owner shell ==="
  python3 -u "$PIPE" --stage qa-wrong-donor --pool mechanism_research --no-shared-gpu

  echo "=== 5/6 supplemental controls + holdout sweep ==="
  python3 -u "$PIPE" --stage supplemental-controls --pool mechanism_research --no-shared-gpu
  python3 -u "$PIPE" --stage das-sweep-holdout --pool mechanism_research --no-shared-gpu

  echo "=== 6/6 refresh report ==="
  python3 -u "$PIPE" --stage report --pool mechanism_research

  echo "[line_b] post-sweep complete"
} 2>&1 | tee "$LOG"
