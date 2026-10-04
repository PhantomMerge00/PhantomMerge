#!/usr/bin/env bash
# Full Shopping diagnostic → detection → mitigation chain (matches Qwen paper).
set -euo pipefail
source "$(dirname "$0")/env.sh"
_ccer_check_core

ROOT="${PHANTOM_MERGE_ROOT}"
OPS="${ROOT}/scripts"
cd "${ROOT}"

SKIP_LINE_A="${SKIP_LINE_A:-0}"
SKIP_JLENS="${SKIP_JLENS:-0}"
SKIP_E2E_LLM="${SKIP_E2E_LLM:-1}"

_ccer_echo "Phase B: Shopping pipeline (model=${CCER_MODEL_MANIFEST_ID})"

if [[ "${SKIP_LINE_A}" != "1" ]]; then
  _ccer_echo "B1 Line A v3 extract + probe"
  bash "${OPS}/ccer/run_line_a_v3.sh"
  # Wait for shard workers if launched in background
  if [[ -f "${ROOT}/results/line_a/v3/extract_pids.txt" ]]; then
    _ccer_echo "Waiting for Line A extract PIDs..."
    while read -r pid; do
      wait "${pid}" 2>/dev/null || true
    done < "${ROOT}/results/line_a/v3/extract_pids.txt"
    python -u -m ccer.pipelines.run_line_a_probe_v3
  fi
fi

if [[ "${SKIP_JLENS}" != "1" ]]; then
  _ccer_echo "B2 J-lens fit"
  bash "${OPS}/ccer/run_fit_jacobian_lens.sh"
fi

_ccer_echo "B3 Line K probe scores"
bash "${OPS}/ccer/run_line_k_claim_filter.sh"

_ccer_echo "B4 BindSurprise / PRISM audit (topk=50)"
bash "${OPS}/ccer/run_prism_audit.sh" --topk 50

_ccer_echo "B5 Detection baselines"
python -m ccer.pipelines.run_detection_baselines

_ccer_echo "B6 Line L claim filter / rewrite"
bash "${OPS}/ccer/run_line_l_claim_filter.sh"

_ccer_echo "B7 PRISM-L+ experiment"
if [[ "${SKIP_E2E_LLM}" == "1" ]]; then
  export PRISM_L_PLUS_NO_LLM=1
fi
bash "${OPS}/ccer/run_prism_l_plus.sh"

_ccer_echo "B7b E2E mitigation"
if [[ "${SKIP_E2E_LLM}" == "1" ]]; then
  export E2E_NO_LLM=1
fi
bash "${OPS}/ccer/run_e2e_mitigation.sh"

_ccer_echo "B8 AGR anchor mass cache + validation"
python -m ccer.pipelines.run_agr_anchor_mass_cache
python -m ccer.pipelines.run_agr_validation

_ccer_echo "B9 Detector comparison (AGR vs BindSurprise)"
python -m ccer.pipelines.run_detector_comparison --skip-e2e

_ccer_echo "Phase D: paper tables"
bash "$(dirname "$0")/run_paper_tables.sh"

_ccer_echo "Shopping pipeline complete."
