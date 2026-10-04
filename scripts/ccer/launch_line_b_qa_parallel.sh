#!/usr/bin/env bash
# protocol note step 2: QA on expanded pool — safe to run parallel with main das-sweep.
set -euo pipefail
cd ${PHANTOM_MERGE_ROOT}
export PYTHONPATH="${PWD}/active_code"
# Default A100 (nvidia index 3); avoid GPU0 Ada when crowded. Main sweep uses PRO6000 separately.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

OUT="results/p3/round12_line_b"
PIPE="ccer/pipelines/run_p3_round12_line_b.py"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="${OUT}/line_b_qa_parallel_${STAMP}.log"
PIDFILE="${OUT}/line_b_qa.pid"
mkdir -p "$OUT"

echo "[line_b] QA parallel launch CUDA=${CUDA_VISIBLE_DEVICES} log=${LOG}"

setsid bash -c "
  python3 -u ${PIPE} --stage extract-activations --pool mechanism_research --no-shared-gpu && \
  python3 -u ${PIPE} --stage qa-evidence-mask --pool mechanism_research --no-shared-gpu && \
  python3 -u ${PIPE} --stage qa-wrong-donor --pool mechanism_research --no-shared-gpu
" >> "$LOG" 2>&1 &

echo $! > "$PIDFILE"
echo "started QA pid=$(cat $PIDFILE) log=$LOG"
