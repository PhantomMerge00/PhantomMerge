#!/usr/bin/env bash
# Wait for a GPU job to finish, then launch Line F on first whole-GPU with >=62GiB free.
set -euo pipefail
ROOT="${PHANTOM_MERGE_ROOT}"
PY="${PY:-${PHANTOM_MERGE_HOME}/wz26b/bin/python3}"
WAIT_PID="${WAIT_PID:-2015895}"
POLL_SEC="${POLL_SEC:-30}"
MIN_FREE_MIB="${MIN_FREE_MIB:-65000}"
# Prefer nvidia-smi GPU order: 1 Blackwell, 3 A100, 4 Blackwell
PREFER_GPUS="${PREFER_GPUS:-1,3,4}"

export PYTHONPATH="${ROOT}/active_code"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

LOGDIR="${ROOT}/results/p3/round12_line_f"
mkdir -p "${LOGDIR}"
LAUNCH_LOG="${LOGDIR}/line_f_autolaunch_$(date +%Y%m%d_%H%M%S).log"

log() { echo "[$(date -Is)] $*" | tee -a "${LAUNCH_LOG}" >&2; }

log "waiting for PID ${WAIT_PID} to exit (poll ${POLL_SEC}s)"
while kill -0 "${WAIT_PID}" 2>/dev/null; do
  ps -p "${WAIT_PID}" -o etime=,pcpu= 2>/dev/null | awk -v p="${WAIT_PID}" '{print "[wait] pid=" p " elapsed=" $1 " cpu=" $2 "%"}' | tee -a "${LAUNCH_LOG}"
  sleep "${POLL_SEC}"
done
log "PID ${WAIT_PID} finished"

pick_gpu() {
  local gpu free
  IFS=',' read -ra ORDER <<< "${PREFER_GPUS}"
  for gpu in "${ORDER[@]}"; do
    free="$(nvidia-smi -i "${gpu}" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')"
    if [[ -n "${free}" && "${free}" -ge "${MIN_FREE_MIB}" ]]; then
      echo "${gpu}"
      return 0
    fi
    log "GPU${gpu} free=${free}MiB < ${MIN_FREE_MIB}MiB, skip" >&2
  done
  return 1
}

GPU=""
for attempt in $(seq 1 120); do
  if GPU="$(pick_gpu)"; then
    log "selected nvidia-smi GPU${GPU} (free >= ${MIN_FREE_MIB} MiB)"
    break
  fi
  log "no suitable GPU yet, retry ${attempt}/120 in ${POLL_SEC}s"
  sleep "${POLL_SEC}"
done

if [[ -z "${GPU}" ]]; then
  log "ERROR: no GPU with >= ${MIN_FREE_MIB} MiB free after waiting"
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${GPU}"
RUN_LOG="${LOGDIR}/line_f_final_$(date +%Y%m%d_%H%M%S).log"
log "launching Line F on CUDA_VISIBLE_DEVICES=${GPU} device-map=cuda:0"
log "run log → ${RUN_LOG}"

cd "${ROOT}"
"${PY}" -u ccer/pipelines/run_p3_round12_line_f.py \
  --stage shell-wrong-donor \
  --pool mechanism_research \
  --device-map cuda:0 \
  --no-shared-gpu \
  --probe-max-tokens 384 \
  --layer 32 \
  --rank 16 \
  --alpha 1.0 \
  2>&1 | tee "${RUN_LOG}"

log "Line F finished with exit=${PIPESTATUS[0]}"
