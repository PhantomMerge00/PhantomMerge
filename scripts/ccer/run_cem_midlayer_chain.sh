#!/usr/bin/env bash
# CEM mid-layer: P3a (IIA pick, fast) → core P3b (3 shards) → path patch if diff>0
# GPUs: 0/1/2 idle-friendly; override via P3A_GPU / P3B_GPU0..2
set -uo pipefail
ROOT="${PHANTOM_MERGE_ROOT}"
cd "${ROOT}"
export PYTHONPATH="${ROOT}/active_code"
PY="${PY:-${PHANTOM_MERGE_HOME}/wz26b/bin/python3}"
LOGDIR="${ROOT}/data/logs/ops/$(date +%Y%m%d)"
mkdir -p "${LOGDIR}"
STAMP="$(date +%H%M%S)"
STATUS="${LOGDIR}/cem_midlayer_${STAMP}.status"

log() { echo "[$(date -Iseconds)] $*" | tee -a "${STATUS}"; }

# CUDA_VISIBLE_DEVICES = nvidia-smi GPU index (no remapping)
P3A_GPU="${P3A_GPU:-1}"
P3B_GPU0="${P3B_GPU0:-1}"
P3B_GPU1="${P3B_GPU1:-4}"
P3B_GPU2="${P3B_GPU2:-1}"
P3B_NUM_SHARDS="${P3B_NUM_SHARDS:-2}"
SKIP_P3A="${SKIP_P3A:-0}"

if [[ "${SKIP_P3A}" != "1" ]]; then
  log "=== P3a CEM IIA-pick (L28-54, fast screen) GPU${P3A_GPU} ==="
  CUDA_VISIBLE_DEVICES="${P3A_GPU}" "${PY}" -u ccer/pipelines/run_p3a.py \
    --track CEM --iia-pick --device-map auto \
    --probe-pairs 4 --probe-max-tokens 96 --fast-screen \
    --layer-lo 28 --layer-hi 54 \
    2>&1 | tee "${LOGDIR}/p3a_iia_${STAMP}.log"
else
  log "=== SKIP P3a (SKIP_P3A=1); using existing binding_roi.json ==="
fi

rm -f "${ROOT}/results/p3/interchange_rows_cem_s"*.jsonl

log "=== P3b core (${P3B_NUM_SHARDS} shards parallel GPUs ${P3B_GPU0},${P3B_GPU1}[,${P3B_GPU2}]) ==="
_p3b_shard() {
  local idx="$1" gpu="$2"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" -u ccer/pipelines/run_p3b.py \
    --device-map auto --track CEM --no-resume --shard-index "${idx}" --num-shards "${P3B_NUM_SHARDS}" \
    --controls no_intervention,self_donor,target_interchange --max-new-tokens 384 \
    >> "${LOGDIR}/p3b_s${idx}_${STAMP}.log" 2>&1
}
if [[ "${P3B_NUM_SHARDS}" -le 1 ]]; then
  _p3b_shard 0 "${P3B_GPU0}"
else
  _p3b_shard 0 "${P3B_GPU0}" &
  pid0=$!
  log "shard0 pid=${pid0} gpu=${P3B_GPU0}; stagger 45s before shard1 (avoid dual model-load OOM)"
  sleep 45
  _p3b_shard 1 "${P3B_GPU1}" &
  pid1=$!
  log "shard1 pid=${pid1} gpu=${P3B_GPU1}"
  wait "${pid0}" "${pid1}"
  if [[ "${P3B_NUM_SHARDS}" -ge 3 ]]; then
    _p3b_shard 2 "${P3B_GPU2}"
  fi
fi
"${PY}" -u ccer/pipelines/run_p3b.py --track CEM --merge-shards --num-shards "${P3B_NUM_SHARDS}" \
  2>&1 | tee -a "${LOGDIR}/p3b_merge_${STAMP}.log"

DIFF_RATE="$("${PY}" - <<'PY'
import json
from pathlib import Path
p = Path("results/p3/iia_summary.json")
if not p.is_file():
    print(0)
    raise SystemExit
s = json.loads(p.read_text())
print(s.get("target_interchange_output_diff_rate") or 0)
PY
)"
log "target_interchange_output_diff_rate=${DIFF_RATE}"

if "${PY}" -c "import sys; sys.exit(0 if float('${DIFF_RATE}')>0 else 1)"; then
  log "=== Path patch (attention + MLP) ==="
  for site in attention mlp; do
    _path_shard() {
      local idx="$1" gpu="$2"
      CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" -u ccer/pipelines/run_p3b.py \
        --device-map auto --track CEM --no-resume --shard-index "${idx}" --num-shards "${P3B_NUM_SHARDS}" \
        --inject-site "$site" --controls no_intervention,target_interchange --max-new-tokens 384 \
        >> "${LOGDIR}/p3_path_${site}_s${idx}_${STAMP}.log" 2>&1
    }
    _path_shard 0 "${P3B_GPU0}" &
    _path_shard 1 "${P3B_GPU1}" &
    wait
    if [[ "${P3B_NUM_SHARDS}" -ge 3 ]]; then
      _path_shard 2 "${P3B_GPU2}"
    fi
  done
  "${PY}" -u ccer/pipelines/run_p3_path_patch.py --track CEM --num-shards "${P3B_NUM_SHARDS}" \
    2>&1 | tee -a "${LOGDIR}/p3_path_merge_${STAMP}.log"
  log "PATH_PATCH=done"
else
  log "SKIP path patch: target_interchange still identical to no_intervention"
fi
log "DONE"
