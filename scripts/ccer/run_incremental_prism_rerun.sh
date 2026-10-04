#!/usr/bin/env bash
# Incremental: re-run PRISM gate arms only (top-k=50 audit), merge into existing summaries.
set -euo pipefail
cd ${PHANTOM_MERGE_ROOT}
source ~/wz26b/bin/activate
export PYTHONPATH=active_code
export LINE_L_PLUS_VLLM_BASE_URL="${LINE_L_PLUS_VLLM_BASE_URL:-http://127.0.0.1:8012/v1}"
export LINE_L_PLUS_LLM_MODEL="${LINE_L_PLUS_LLM_MODEL:-Qwen3-8B}"

NO_LLM=""
if [[ "${INCR_NO_LLM:-0}" == "1" ]]; then
  NO_LLM="--no-llm"
fi

python -m ccer.pipelines.run_incremental_prism_rerun --n-boot 500 ${NO_LLM}
echo "[incremental] done."
