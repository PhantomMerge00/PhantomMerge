#!/usr/bin/env bash
# Line C: 2-shard parallel on idle GPUs (reuse Line B pool, no duplicate activations).
set -euo pipefail
ROOT="${PHANTOM_MERGE_ROOT}"
export PYTHONPATH="${ROOT}/active_code"
PY="${PY:-${PHANTOM_MERGE_HOME}/wz26b/bin/python3}"
PIPE="${ROOT}/ccer/pipelines/run_p3_line_c.py"
OUT="${ROOT}/results/p3/line_c"
mkdir -p "${OUT}"
STAMP="$(date +%Y%m%d_%H%M%S)"
NUM_SHARDS="${NUM_SHARDS:-2}"

# Verified mapping (2026-09-16): CUDA 2=PRO6000 ~94GiB, CUDA 4=A100 ~79GiB
GPUS=(2 4)
COMMON=(--track CEM --device-map cuda:0 --probe-max-tokens 384 --layer 32 --rank 16 --alpha 1.0
  --pool mechanism_research --skip-pair-audit --num-shards "${NUM_SHARDS}")

echo "[line_c] parallel launch NUM_SHARDS=${NUM_SHARDS} stamp=${STAMP}"
"${PY}" -c "import os,subprocess; [print('probe',i,subprocess.run(['${PY}','-c','import torch;print(torch.cuda.get_device_name(0),round(torch.cuda.mem_get_info()[0]/1024**3,1))'],env={**os.environ,'CUDA_VISIBLE_DEVICES':str(i)},capture_output=True,text=True).stdout.strip()) for i in [2,4]]"

for i in $(seq 0 $((NUM_SHARDS - 1))); do
  gpu="${GPUS[$i]:-$i}"
  log="${OUT}/shard${i}_gpu${gpu}_${STAMP}.log"
  echo "[line_c] shard ${i} → CUDA_VISIBLE_DEVICES=${gpu} log=${log}"
  setsid env CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${ROOT}/active_code" \
    "${PY}" -u "${PIPE}" --stage run \
    "${COMMON[@]}" --shard-index "${i}" >> "${log}" 2>&1 &
  echo $! > "${OUT}/shard${i}.pid"
done

echo "[line_c] waiting for shards..."
fail=0
for i in $(seq 0 $((NUM_SHARDS - 1))); do
  pid="$(cat "${OUT}/shard${i}.pid")"
  if ! wait "${pid}"; then
    echo "[line_c] shard ${i} failed (pid=${pid})" >&2
    fail=1
  fi
done

"${PY}" -u "${PIPE}" --stage merge-shards --num-shards "${NUM_SHARDS}"
echo "[line_c] done stamp=${STAMP} fail=${fail}"
