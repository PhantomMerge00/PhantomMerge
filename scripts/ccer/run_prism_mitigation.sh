#!/usr/bin/env bash
set -euo pipefail
ROOT="${PHANTOM_MERGE_ROOT}"
source "${ROOT}/scripts/_lib.sh"
activate_wz26b
cd "${ROOT}"
export PYTHONPATH="${ROOT}/active_code:${PYTHONPATH:-}"
python ccer/pipelines/run_prism_mitigation.py "$@"
