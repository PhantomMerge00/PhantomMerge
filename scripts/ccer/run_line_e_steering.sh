#!/usr/bin/env bash
# Line E: CAA contrastive steering mitigation (internal design note §E).
set -uo pipefail
export ROOT="${PHANTOM_MERGE_ROOT}"
export PHANTOM_MERGE_ROOT="${ROOT}"
export PHANTOM_LOG_ROOT="${ROOT}/data/logs/ops"
cd "${ROOT}"
# shellcheck source=/dev/null
source "${ROOT}/scripts/_lib.sh"
activate_wz26b
export PYTHONPATH="${ROOT}/active_code"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

: "${LINE_E_DRY_RUN:=0}"
: "${LINE_E_POOL:=mechanism_research}"
: "${LINE_E_LIMIT:=}"
: "${LINE_E_LAYER:=}"
: "${LINE_E_POSITION:=}"
: "${LINE_E_NUM_SHARDS:=1}"
: "${LINE_E_SHARD_INDEX:=0}"
: "${LINE_E_MERGE:=0}"

DAY="$(date +%Y%m%d)"
LOGDIR="${ROOT}/data/logs/ops/${DAY}"
mkdir -p "${LOGDIR}"
STAMP="$(date +%H%M%S)"
LOG="${LOGDIR}/ccer_line_e_${STAMP}.log"

PYTHON_BIN="${PYTHON_BIN:-${PHANTOM_MERGE_HOME}/wz26b/bin/python3}"
ARGS=("${PYTHON_BIN}" -u "${ROOT}/ccer/pipelines/run_p3_line_e.py" --pool "${LINE_E_POOL}")
if [[ "${LINE_E_DRY_RUN}" == "1" ]]; then
  ARGS+=(--dry-run)
fi
if [[ -n "${LINE_E_LAYER}" ]]; then
  ARGS+=(--layer "${LINE_E_LAYER}")
fi
if [[ -n "${LINE_E_POSITION}" ]]; then
  ARGS+=(--position "${LINE_E_POSITION}")
fi
if [[ -n "${LINE_E_LIMIT}" ]]; then
  ARGS+=(--limit "${LINE_E_LIMIT}")
fi
ARGS+=(--shard-index "${LINE_E_SHARD_INDEX}" --num-shards "${LINE_E_NUM_SHARDS}")
if [[ "${LINE_E_MERGE}" == "1" ]]; then
  ARGS+=(--merge-shards)
fi

echo "[line_e] ${ARGS[*]}" | tee "${LOG}"
"${ARGS[@]}" 2>&1 | tee -a "${LOG}"
