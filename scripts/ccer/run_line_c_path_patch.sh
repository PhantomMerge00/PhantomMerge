#!/usr/bin/env bash
# Line C: Attention vs MLP path patching — multi-GPU share + optional site staging.
#
# 多卡分担（与线路 B 相同思路）:
#   eval "$(python3 scripts/ccer/pick_gpu_vram.py --min-gb 35)"
#   bash scripts/ccer/run_line_c_path_patch.sh
#
# 分阶段（attention / mlp 分开跑，降低单次时长、便于续跑）:
#   LINE_C_MODE=staged bash scripts/ccer/run_line_c_path_patch.sh
#
# pair 分片（缩短单次运行，不减少模型显存占用；多 shard 顺序执行）:
#   NUM_SHARDS=2 bash scripts/ccer/run_line_c_path_patch.sh
set -euo pipefail
ROOT="${PHANTOM_MERGE_ROOT}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${ROOT}/active_code"
PY="${PY:-${PHANTOM_MERGE_HOME}/wz26b/bin/python3}"

if [[ ! -x "${PY}" ]]; then
  PY="python3"
fi

# Auto-pick multi-GPU if user未指定；35GB 阈值允许 GPU1+GPU4 (~39GB) + CPU spill
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  eval "$("${PY}" "${SCRIPT_DIR}/pick_gpu_vram.py" --min-gb "${LINE_C_MIN_VRAM_GB:-35}")"
else
  eval "$("${PY}" "${SCRIPT_DIR}/pick_gpu_vram.py" --min-gb "${LINE_C_MIN_VRAM_GB:-35}")" || true
fi

LOGDIR="${ROOT}/results/p3/line_c"
mkdir -p "${LOGDIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="${LOGDIR}/path_patch_${STAMP}.log"
NUM_SHARDS="${NUM_SHARDS:-1}"
LINE_C_MODE="${LINE_C_MODE:-both}"  # both | staged

PIPE_ARGS=(
  --track CEM
  --device-map auto
  --probe-max-tokens 384
  --layer 32
  --rank 16
  --alpha 1.0
)

run_shard() {
  local site="$1"
  local shard="$2"
  local extra=()
  if [[ "${site}" != "both" ]]; then
    extra+=(--inject-site "${site}")
  fi
  "${PY}" -u "${ROOT}/ccer/pipelines/run_p3_line_c.py" \
    --stage run \
    "${PIPE_ARGS[@]}" \
    "${extra[@]}" \
    --shard-index "${shard}" \
    --num-shards "${NUM_SHARDS}"
}

{
  echo "[line_c] PY=${PY}"
  echo "[line_c] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} NUM_SHARDS=${NUM_SHARDS} MODE=${LINE_C_MODE} $(date -Is)"

  "${PY}" -u "${ROOT}/ccer/pipelines/run_p3_line_c.py" \
    --stage pair-audit --layer 32

  if [[ "${LINE_C_MODE}" == "staged" ]]; then
    for site in attention mlp; do
      for SHARD in $(seq 0 $((NUM_SHARDS - 1))); do
        echo "[line_c] staged site=${site} shard=${SHARD}/${NUM_SHARDS}..."
        run_shard "${site}" "${SHARD}"
      done
      if [[ "${NUM_SHARDS}" -gt 1 ]]; then
        "${PY}" -u "${ROOT}/ccer/pipelines/run_p3_line_c.py" \
          --stage merge-shards --num-shards "${NUM_SHARDS}" --inject-site "${site}"
      else
        cp -f "${LOGDIR}/path_patch_${site}_s0.json" "${LOGDIR}/path_patch_${site}_merged.json"
      fi
    done
    "${PY}" -u "${ROOT}/ccer/pipelines/run_p3_line_c.py" --stage merge-sites
  else
    for SHARD in $(seq 0 $((NUM_SHARDS - 1))); do
      echo "[line_c] shard ${SHARD}/${NUM_SHARDS}..."
      run_shard both "${SHARD}"
    done
    if [[ "${NUM_SHARDS}" -gt 1 ]]; then
      "${PY}" -u "${ROOT}/ccer/pipelines/run_p3_line_c.py" \
        --stage merge-shards --num-shards "${NUM_SHARDS}"
    else
      cp -f "${LOGDIR}/path_patch_s0.json" "${LOGDIR}/path_patch_merged.json" 2>/dev/null || true
      "${PY}" -u "${ROOT}/ccer/pipelines/run_p3_line_c.py" --stage report
    fi
  fi

  echo "[line_c] done $(date -Is)"
} >> "${LOG}" 2>&1

echo "[line_c] finished log=${LOG}"
