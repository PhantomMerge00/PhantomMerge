#!/usr/bin/env bash
# ROUND2 follow-up: position_permutation activations, P3a v2, extended P3b, P1 CEM rerun, AH, commitment.
set -uo pipefail
export ROOT="${PHANTOM_MERGE_ROOT}"
export PHANTOM_MERGE_ROOT="${ROOT}"
export PHANTOM_LOG_ROOT="${ROOT}/data/logs/ops"
cd "${ROOT}"
# shellcheck source=/dev/null
source "${ROOT}/scripts/_lib.sh"
activate_wz26b
export PYTHONPATH="${ROOT}/active_code"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"

DAY="$(date +%Y%m%d)"
LOGDIR="${ROOT}/data/logs/ops/${DAY}"
mkdir -p "${LOGDIR}"
STAMP="$(date +%H%M%S)"

stop_vllm_if_running() {
  if pgrep -f "vllm.*8003" >/dev/null 2>&1; then
    echo "Stopping vLLM for HF jobs..."
    pkill -f "vllm.*8003" || true
    sleep 15
  fi
}

start_vllm_if_needed() {
  if curl -sf "http://127.0.0.1:8003/v1/models" >/dev/null 2>&1; then
    echo "vLLM already up on :8003"
    return 0
  fi
  export QWEN32_MODEL_DIR="${PHANTOM_MERGE_ROOT}/runtime/.cache/huggingface/hub/Qwen3-32B"
  export CUDA_VISIBLE_DEVICES=1
  export SHOPPING_QWEN_PORT=8003
  echo "Starting vLLM on GPU1 :8003 for P1/AH..."
  nohup bash "${ROOT}/scripts/vllm/shopping_agent_qwen32b_gpu0.sh" >> "${LOGDIR}/vllm_round2_${STAMP}.log" 2>&1 &
  for _ in $(seq 1 24); do
    sleep 10
    curl -sf "http://127.0.0.1:8003/v1/models" >/dev/null 2>&1 && return 0
  done
  echo "WARN: vLLM did not become ready" >&2
}

echo "=== ROUND2: P3-0 position_permutation + dense layers (CAP) ==="
stop_vllm_if_running
python3 -u "${ROOT}/ccer/pipelines/run_p3_0.py" --device-map auto --dense-layers --track CAP \
  2>&1 | tee "${LOGDIR}/ccer_p3_0_round2_${STAMP}.log"

echo "=== ROUND2: P3a v2 ==="
if ! python3 -u "${ROOT}/ccer/pipelines/run_p3a.py" --track CAP \
  2>&1 | tee "${LOGDIR}/ccer_p3a_round2_${STAMP}.log"; then
  echo "P3A_FAILED exit=$?" | tee -a "${LOGDIR}/ccer_p3a_round2_${STAMP}.log"
fi

echo "=== ROUND2: P3b extended (resume, all positions, layer sweep) ==="
python3 -u "${ROOT}/ccer/pipelines/run_p3b.py" --device-map auto --all-positions --layer-sweep \
  2>&1 | tee "${LOGDIR}/ccer_p3b_round2_${STAMP}.log"

echo "=== ROUND2: P1 CEM rival refresh ==="
start_vllm_if_needed
python3 -u "${ROOT}/ccer/pipelines/run_p1_dev.py" --cohort CEM --refresh-conditions rival_value_swap \
  2>&1 | tee "${LOGDIR}/ccer_p1_cem_round2_${STAMP}.log"
python3 -u "${ROOT}/ccer/pipelines/p1_go_summary.py" \
  2>&1 | tee "${LOGDIR}/ccer_p1_go_round2_${STAMP}.log"

echo "=== ROUND2: AH specificity ==="
python3 -u "${ROOT}/ccer/pipelines/run_ah_specificity.py" \
  2>&1 | tee "${LOGDIR}/ccer_ah_round2_${STAMP}.log"

echo "=== ROUND2: CEM P3-0/a/b ==="
stop_vllm_if_running
python3 -u "${ROOT}/ccer/pipelines/run_p3_0.py" --device-map auto --dense-layers --track CEM \
  2>&1 | tee "${LOGDIR}/ccer_p3_0_cem_${STAMP}.log"
python3 -u "${ROOT}/ccer/pipelines/run_p3a.py" --track CEM \
  2>&1 | tee "${LOGDIR}/ccer_p3a_cem_${STAMP}.log"
python3 -u "${ROOT}/ccer/pipelines/run_p3b.py" --device-map auto --track CEM --limit 32 \
  2>&1 | tee "${LOGDIR}/ccer_p3b_cem_${STAMP}.log"

echo "=== ROUND2: Commitment restoration ==="
python3 -u "${ROOT}/ccer/pipelines/run_p3_commitment.py" --device-map auto \
  2>&1 | tee "${LOGDIR}/ccer_p3_commitment_${STAMP}.log"

echo "=== ROUND2: P3a v2 retry (post-DAS fix) ==="
python3 -u "${ROOT}/ccer/pipelines/run_p3a.py" --track CAP \
  2>&1 | tee -a "${LOGDIR}/ccer_p3a_round2_retry_${STAMP}.log" || true
python3 -u "${ROOT}/ccer/pipelines/run_p3a.py" --track CEM \
  2>&1 | tee -a "${LOGDIR}/ccer_p3a_cem_retry_${STAMP}.log" || true

echo "=== ROUND2 DONE $(date -Iseconds) ==="
