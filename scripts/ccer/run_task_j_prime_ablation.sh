#!/usr/bin/env bash
# Task J′: CPU A/B patch-tier ablation (J vs J′ live anchor)
set -euo pipefail
cd ${PHANTOM_MERGE_ROOT}
source ~/wz26b/bin/activate
export PYTHONPATH=active_code

N_PAIRS="${N_PAIRS:-19}"
python3 -u scripts/ccer/run_task_j_prime_ablation.py \
  --n-pairs "${N_PAIRS}" \
  2>&1 | tee "results/p3/round12_line_b/task_j_prime/run_ablation_$(date +%Y%m%d_%H%M%S).log"
