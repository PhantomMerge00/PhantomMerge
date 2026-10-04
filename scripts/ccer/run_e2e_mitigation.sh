#!/usr/bin/env bash
# E2E main experiment: native detection per system; probe-gated matrix in prism_l_plus appendix.
set -euo pipefail
cd ${PHANTOM_MERGE_ROOT}
source ~/wz26b/bin/activate
export PYTHONPATH=active_code

METHODS="${E2E_METHODS:-track_k_e2e,b3_copy_e2e,b1_rarr_e2e,b2_cove_e2e,prism_l_plus_a_e2e,prism_l_plus_b_e2e}"
PORT="${E2E_VLLM_PORT:-8012}"
export LINE_L_PLUS_VLLM_BASE_URL="http://127.0.0.1:${PORT}/v1"
export LINE_L_PLUS_LLM_MODEL="${E2E_LLM_MODEL:-Qwen3-8B}"

NO_LLM=""
if [[ "${E2E_NO_LLM:-0}" == "1" ]]; then
  NO_LLM="--no-llm"
fi

if curl -sf "${LINE_L_PLUS_VLLM_BASE_URL}/models" >/dev/null 2>&1; then
  echo "[e2e] vLLM ready at ${LINE_L_PLUS_VLLM_BASE_URL}"
else
  echo "[e2e] WARNING: vLLM not ready; LLM methods may fail"
fi

python -m ccer.pipelines.run_e2e_mitigation_experiment \
  --methods "${METHODS}" \
  --n-boot 500 \
  ${NO_LLM}

echo "[e2e] done. Main: results/e2e_mitigation/METHOD_COMPARISON_E2E.md"
echo "[e2e] Appendix: results/prism_l_plus/APPENDIX_PROBE_GATED.md"
