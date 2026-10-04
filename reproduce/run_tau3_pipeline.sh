#!/usr/bin/env bash
# τ³ cross-domain pipeline for one domain (telecom | airline | retail).
# Usage: bash reproduce/run_tau3_pipeline.sh telecom
set -euo pipefail
source "$(dirname "$0")/env.sh"
_ccer_check_core

DOMAIN="${1:-telecom}"
if [[ ! "${DOMAIN}" =~ ^(telecom|airline|retail)$ ]]; then
  echo "Usage: $0 {telecom|airline|retail}" >&2
  exit 1
fi

cd "${PHANTOM_ACTIVE_CODE}"

_ccer_echo "Phase C: τ³ domain=${DOMAIN}"

_ccer_echo "C1 normalize"
python -m ccer.pipelines.run_tau3_telecom_normalize --domain "${DOMAIN}"

_ccer_echo "C2 extract activations"
python -m ccer.pipelines.run_tau3_telecom_extract_activations --domain "${DOMAIN}"

_ccer_echo "C3 slot readout (topk=50)"
python -m ccer.pipelines.run_tau3_telecom_slot_readout --domain "${DOMAIN}" --topk 50

_ccer_echo "C4 export shopping frozen probe (if missing)"
python -c "from ccer.mechanism.tau3.frozen_probe import export_shopping_frozen_probe; export_shopping_frozen_probe()"

_ccer_echo "C5 cross-domain eval"
python -m ccer.pipelines.run_tau3_agr_slot_cross_domain --domain "${DOMAIN}"

_ccer_echo "C6 red-flag audit"
python -m ccer.audit.tau3_agr_red_flag_audit
python -m ccer.audit.tau3_bind_surprise_reval --domain "${DOMAIN}" || true

_ccer_echo "τ³ ${DOMAIN} complete. Run run_paper_tables.sh after all domains."
