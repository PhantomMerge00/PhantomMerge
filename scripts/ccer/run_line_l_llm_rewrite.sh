#!/usr/bin/env bash
# Line L: LLM fallback rewrite — vLLM on GPU2 (8B) or GPU0+2 (32B TP2 if enough free VRAM).
set -euo pipefail
cd ${PHANTOM_MERGE_ROOT}
source ~/wz26b/bin/activate
export PYTHONPATH=active_code
export CUDA_DEVICE_ORDER=PCI_BUS_ID

LOG_DIR="results/line_l/logs"
mkdir -p "${LOG_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"

# Profiles:
#   8b  — Qwen3-8B on physical GPU2 only (~16GiB weights; fits with ~9GiB already in use)
#   32b — Qwen3-32B TP=2 on physical GPU0+2 (needs ~0.75 gpu_util when each card has ~9GiB reserved)
PROFILE="${LINE_L_VLLM_PROFILE:-8b}"
PORT="${LINE_L_VLLM_PORT:-8012}"
BASE_URL="http://127.0.0.1:${PORT}/v1"
export LINE_L_VLLM_BASE_URL="${BASE_URL}"

HF_HOME="${HF_HOME:-${PHANTOM_MERGE_ROOT}/runtime/.cache/huggingface}"
QWEN32="${QWEN32_MODEL_DIR:-${HF_HOME}/hub/Qwen3-32B}"
QWEN8="${QWEN8_MODEL_DIR:-${HF_HOME}/hub/Qwen3-8B}"

wait_vllm() {
  for _ in $(seq 1 180); do
    if curl -sf "${BASE_URL}/models" >/dev/null 2>&1; then
      echo "[line_l_llm] vLLM ready at ${BASE_URL}"
      return 0
    fi
    sleep 5
  done
  return 1
}

start_vllm() {
  local log="$1"
  shift
  echo "[line_l_llm] starting: $*"
  echo "[line_l_llm] log: ${log}"
  nohup "$@" >"${log}" 2>&1 &
}

if curl -sf "${BASE_URL}/models" >/dev/null 2>&1; then
  echo "[line_l_llm] vLLM already up at ${BASE_URL}"
else
  if [[ "${PROFILE}" == "32b" ]]; then
    PHYSICAL_GPUS="${LINE_L_VLLM_GPUS:-0,2}"
    GPU_UTIL="${LINE_L_GPU_UTIL:-0.75}"
    MAX_LEN="${LINE_L_MAX_MODEL_LEN:-4096}"
    LOG="${LOG_DIR}/vllm_line_l_32b_tp2_${PORT}_${TS}.log"
    echo "[line_l_llm] profile=32b TP=2 GPUs=${PHYSICAL_GPUS} util=${GPU_UTIL} max_len=${MAX_LEN}"
    echo "[line_l_llm] NOTE: failed at 0.88 util — cards had only ~38–39GiB free each."
    start_vllm "${LOG}" \
      env CUDA_VISIBLE_DEVICES="${PHYSICAL_GPUS}" \
      vllm serve "${QWEN32}" \
      --served-model-name Qwen3-32B \
      --dtype bfloat16 \
      --tensor-parallel-size 2 \
      --max-model-len "${MAX_LEN}" \
      --gpu-memory-utilization "${GPU_UTIL}" \
      --port "${PORT}" \
      --default-chat-template-kwargs '{"enable_thinking": false}' \
      --enforce-eager
  else
    PHYSICAL_GPU="${LINE_L_VLLM_GPUS:-2}"
    GPU_UTIL="${LINE_L_GPU_UTIL:-0.82}"
    MAX_LEN="${LINE_L_MAX_MODEL_LEN:-8192}"
    LOG="${LOG_DIR}/vllm_line_l_8b_gpu${PHYSICAL_GPU}_${PORT}_${TS}.log"
    echo "[line_l_llm] profile=8b single-GPU=${PHYSICAL_GPU} util=${GPU_UTIL} (recommended while GPU0/2 have ~9GiB in use)"
    start_vllm "${LOG}" \
      env CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}" \
      vllm serve "${QWEN8}" \
      --served-model-name Qwen3-8B \
      --dtype bfloat16 \
      --max-model-len "${MAX_LEN}" \
      --gpu-memory-utilization "${GPU_UTIL}" \
      --port "${PORT}" \
      --default-chat-template-kwargs '{"enable_thinking": false}' \
      --enforce-eager
  fi
  if ! wait_vllm; then
    echo "[line_l_llm] FATAL: vLLM failed; see latest log in ${LOG_DIR}" >&2
    ls -t "${LOG_DIR}"/vllm_line_l_*"${PORT}"*.log 2>/dev/null | head -1 | xargs tail -30 >&2 || true
    exit 2
  fi
fi

echo "[line_l_llm] running rewrite_all_flagged --rewrite-fallback llm"
python3 -m ccer.pipelines.run_line_l_claim_filter \
  --rewrite-fallback llm \
  --n-boot 500 \
  2>&1 | tee "${LOG_DIR}/line_l_llm_rewrite_${TS}.log"
