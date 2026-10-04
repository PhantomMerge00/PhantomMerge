#!/usr/bin/env bash
# Line B 分阶段执行
#
# GPU 映射（nvidia-smi index ≠ CUDA_VISIBLE_DEVICES）:
#   nvidia-smi GPU1 (PRO 6000 95GB) → CUDA_VISIBLE_DEVICES=2
#
# 阶段:
#   0  pool + train-diagnostics (纯 CPU，已完成可跳过)
#   1  extract-activations      (通常可跳过：复用 Line A 激活)
#   2  pair-audit              (CPU)
#   3  qa-evidence-mask        (GPU)
#   4  qa-wrong-donor          (GPU)
#   5  das-sweep               (GPU，最终对比表)
#   6  report                  (从 JSON 重生成 md)
#   all  eval-all 单次加载模型连续跑 2→5（跳过 extract 若已有 90+ pairs）
set -euo pipefail
cd ${PHANTOM_MERGE_ROOT}
export PYTHONPATH=${PHANTOM_MERGE_ROOT}
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PY="${PY:-python3}"
PIPE="ccer/pipelines/run_p3_round12_line_b.py"
OUT="results/p3/round12_line_b"
mkdir -p "$OUT"

# nvidia-smi GPU1 = PRO 6000 Blackwell (~65GiB 空闲)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

STAGE="${1:-}"
if [[ -z "$STAGE" ]]; then
  cat <<'EOF'
用法: run_round12_line_b_staged.sh <stage>

  0   pool + train-diagnostics (CPU)
  1   extract-activations (GPU，91/92 已有则自动跳过)
  2   pair-audit (CPU)
  3   qa-evidence-mask (GPU)
  4   qa-wrong-donor (GPU)
  5   das-sweep (GPU)
  6   report
  all eval-all: 单次模型加载连续跑 pair-audit + QA + sweep（推荐）

环境变量:
  CUDA_VISIBLE_DEVICES  默认 2（对应 nvidia-smi GPU1 PRO 6000）
  PY                  python 路径
EOF
  exit 1
fi

STAMP=$(date +%Y%m%d_%H%M%S)
COMMON_ARGS=(--pool mechanism_research --ranks 16,32,64 --das-steps 500 --device-map auto --shared-gpu)

run_stage() {
  local name="$1"
  local log="$OUT/stage_${name}_${STAMP}.log"
  echo "[line_b] === stage=$name GPUs=$CUDA_VISIBLE_DEVICES log=$log ==="
  "$PY" -u "$PIPE" --stage "$name" "${COMMON_ARGS[@]}" 2>&1 | tee "$log"
}

case "$STAGE" in
  0)
    run_stage pool
    run_stage train-diagnostics
    ;;
  1) run_stage extract-activations ;;
  2) run_stage pair-audit ;;
  3) run_stage qa-evidence-mask ;;
  4) run_stage qa-wrong-donor ;;
  5) run_stage das-sweep ;;
  6) run_stage report ;;
  all|eval-all)
    run_stage eval-all
    ;;
  *)
    echo "未知 stage: $STAGE" >&2
    exit 1
    ;;
esac

echo "[line_b] done stage=$STAGE"
