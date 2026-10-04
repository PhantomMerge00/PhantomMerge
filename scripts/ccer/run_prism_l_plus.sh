#!/usr/bin/env bash
# PRISM-L+ unified experiment: forensic + full method matrix @ frozen test cohort.
set -euo pipefail
cd ${PHANTOM_MERGE_ROOT}
source ~/wz26b/bin/activate
export PYTHONPATH=ccer:third_party/jacobian-lens

METHODS="${PRISM_L_PLUS_METHODS:-track_k,b3_copy,b1_rarr_official,b2_cove_official,prism_l_plus_a,prism_l_plus_b,prism_audit_only}"
PORT="${PRISM_L_PLUS_VLLM_PORT:-8012}"
export LINE_L_PLUS_VLLM_BASE_URL="http://127.0.0.1:${PORT}/v1"
export LINE_L_PLUS_LLM_MODEL="${PRISM_L_PLUS_LLM_MODEL:-Qwen3-8B}"

NO_LLM=""
if [[ "${PRISM_L_PLUS_NO_LLM:-0}" == "1" ]]; then
  NO_LLM="--no-llm"
fi

echo "[prism_l_plus] forensic diff"
python -m ccer.pipelines.run_forensic_diff

if curl -sf "${LINE_L_PLUS_VLLM_BASE_URL}/models" >/dev/null 2>&1; then
  echo "[prism_l_plus] vLLM ready at ${LINE_L_PLUS_VLLM_BASE_URL}"
else
  echo "[prism_l_plus] WARNING: vLLM not at ${LINE_L_PLUS_VLLM_BASE_URL}; LLM methods may delete"
fi

python -m ccer.pipelines.run_prism_l_plus_experiment \
  --methods "${METHODS}" \
  --n-boot 500 \
  ${NO_LLM}

echo "[prism_l_plus] done. See results/prism_l_plus/METHOD_COMPARISON.md"
