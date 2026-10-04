#!/usr/bin/env bash
# Minimal vLLM for CCER: 8B @8012 (mitigation / FActScore) | 32B @8003 (FacTool PM)
# Mitigation 默认 GPU0（空闲 Ada 49G）；FActScore 批跑可与缓解共用 8012。
set -euo pipefail
ROOT="${PHANTOM_MERGE_ROOT:-${PHANTOM_MERGE_ROOT}}"
HF="${HF_HOME:-${ROOT}/runtime/.cache/huggingface}"
Q8="${HF}/hub/Qwen3-8B"
Q32="${HF}/hub/Qwen3-32B"

usage() {
  echo "Usage: source ~/wz26b/bin/activate && bash $0 {8b|32b|check}"
  echo "  8b  — GPU0 (default), port 8012, served name Qwen3-8B"
  echo "  32b — GPU3 (A100), port 8003, served name Qwen3-32B"
  echo "  check — curl /v1/models on both ports"
  exit 1
}

[[ $# -eq 1 ]] || usage
source ~/wz26b/bin/activate
export CUDA_DEVICE_ORDER=PCI_BUS_ID

ready_8b() {
  [[ -f "${Q8}/config.json" ]] && compgen -G "${Q8}/model-*.safetensors" >/dev/null
}

ready_32b() {
  [[ -f "${Q32}/config.json" ]] && compgen -G "${Q32}/model-*.safetensors" >/dev/null
}

case "$1" in
  8b)
    if ! ready_8b; then
      echo "FATAL: 缓解实验同款权重不完整: ${Q8}" >&2
      echo "  需要 model-*.safetensors（约 16GiB）。 mitigation 登记为 Qwen3-8B@8012；" >&2
      echo "  若目录只有 config/tokenizer，请补全: hf download Qwen/Qwen3-8B --local-dir ${Q8}" >&2
      exit 2
    fi
    exec env CUDA_VISIBLE_DEVICES="${LINE_L_VLLM_GPUS:-0}" vllm serve "${Q8}" \
      --served-model-name Qwen3-8B --port 8012 --dtype bfloat16 \
      --max-model-len 8192 --gpu-memory-utilization 0.82 --enforce-eager \
      --default-chat-template-kwargs '{"enable_thinking": false}'
    ;;
  32b)
    if ! ready_32b; then
      echo "FATAL: missing weights under ${Q32}" >&2
      exit 2
    fi
    exec env CUDA_VISIBLE_DEVICES="${TAU2_GPU3:-3}" vllm serve "${Q32}" \
      --served-model-name Qwen3-32B --port 8003 --dtype bfloat16 \
      --max-model-len 16384 --gpu-memory-utilization 0.95 --max-num-seqs 16 \
      --enforce-eager --default-chat-template-kwargs '{"enable_thinking": false}' \
      --enable-auto-tool-choice --tool-call-parser hermes --generation-config vllm
    ;;
  check)
    curl -sf "http://127.0.0.1:8012/v1/models" && echo " — 8012 OK" || echo " — 8012 down"
    curl -sf "http://127.0.0.1:8003/v1/models" && echo " — 8003 OK" || echo " — 8003 down"
    ;;
  *) usage ;;
esac
