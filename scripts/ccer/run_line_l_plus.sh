#!/usr/bin/env bash
# Line L+ full experiment: CPU baselines + optional Llama-3.1-8B for RARR/CoVe.
set -euo pipefail
cd ${PHANTOM_MERGE_ROOT}
source ~/wz26b/bin/activate
export PYTHONPATH=active_code
export CUDA_DEVICE_ORDER=PCI_BUS_ID

LOG_DIR="results/line_l_plus/logs"
mkdir -p "${LOG_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"

METHODS="${LINE_L_PLUS_METHODS:-track_k,eval_e1,eval_e2,eval_e3,b3_copy,b1_rarr,b2_cove,ours_v3,wrong_anchor_extractive}"
PORT="${LINE_L_PLUS_VLLM_PORT:-8013}"
BASE_URL="http://127.0.0.1:${PORT}/v1"
export LINE_L_PLUS_VLLM_BASE_URL="${BASE_URL}"
export LINE_L_PLUS_LLM_MODEL="Llama-3.1-8B-Instruct"

HF_HOME="${HF_HOME:-${PHANTOM_MERGE_ROOT}/runtime/.cache/huggingface}"
LLAMA8="${LLAMA8_MODEL_DIR:-${HF_HOME}/hub/Llama-3.1-8B-Instruct}"

NO_LLM=""
if [[ "${LINE_L_PLUS_NO_LLM:-0}" == "1" ]]; then
  NO_LLM="--no-llm"
fi

need_llm=0
if [[ "${METHODS}" == *"b1_rarr"* ]] || [[ "${METHODS}" == *"b2_cove"* ]]; then
  need_llm=1
fi

wait_vllm() {
  for _ in $(seq 1 180); do
    if curl -sf "${BASE_URL}/models" >/dev/null 2>&1; then
      echo "[line_l_plus] vLLM ready at ${BASE_URL}"
      return 0
    fi
    sleep 5
  done
  return 1
}

if [[ "${need_llm}" == "1" ]] && [[ "${LINE_L_PLUS_NO_LLM:-0}" != "1" ]]; then
  # Fallback to existing Qwen3-8B on 8012 if Llama port not up
  if curl -sf "http://127.0.0.1:8012/v1/models" >/dev/null 2>&1; then
    export LINE_L_PLUS_VLLM_BASE_URL="http://127.0.0.1:8012/v1"
    export LINE_L_PLUS_LLM_MODEL="Qwen3-8B"
    BASE_URL="${LINE_L_PLUS_VLLM_BASE_URL}"
    need_llm=0
    echo "[line_l_plus] using existing Qwen3-8B at ${BASE_URL}"
  fi
fi

if [[ "${need_llm}" == "1" ]] && [[ "${LINE_L_PLUS_NO_LLM:-0}" != "1" ]]; then
  if ! curl -sf "${BASE_URL}/models" >/dev/null 2>&1; then
    LOG="${LOG_DIR}/vllm_llama8_${PORT}_${TS}.log"
    echo "[line_l_plus] starting Llama-3.1-8B on GPU2 port=${PORT}"
    nohup env CUDA_VISIBLE_DEVICES=2 \
      vllm serve "${LLAMA8}" \
      --served-model-name Llama-3.1-8B-Instruct \
      --dtype bfloat16 \
      --max-model-len 4096 \
      --gpu-memory-utilization 0.78 \
      --port "${PORT}" \
      >"${LOG}" 2>&1 &
    wait_vllm || { echo "vLLM failed"; exit 1; }
  fi
fi

python -m ccer.pipelines.run_line_l_plus_experiment \
  --methods "${METHODS}" \
  --n-boot 500 \
  ${NO_LLM}

echo "[line_l_plus] done. See results/line_l_plus/METHOD_COMPARISON.md"
