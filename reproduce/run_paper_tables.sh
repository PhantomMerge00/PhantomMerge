#!/usr/bin/env bash
# Aggregate RQ1–RQ3 paper tables (CPU; requires prior pipeline outputs).
set -euo pipefail
source "$(dirname "$0")/env.sh"
_ccer_check_core

OPS="${PHANTOM_MERGE_ROOT}/scripts"
cd "${OPS}"

_ccer_echo "=== RQ1 ==="
python run_rq1_main_table.py

_ccer_echo "=== RQ2 ==="
python build_construct_validity_audit.py
python build_rq2_main_table.py
python build_rq2_p2_tables.py

_ccer_echo "=== RQ3 ==="
python run_anchor_baselines_line_a_test.py
python build_repair_vs_delete.py

_ccer_echo "Done. See results/rq{1,2,3}/"
