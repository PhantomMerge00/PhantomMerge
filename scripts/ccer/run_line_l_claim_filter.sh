#!/usr/bin/env bash
# Track L: anchor-evidence-grounded extractive rewrite (CPU only).
set -euo pipefail
ROOT="${PHANTOM_MERGE_ROOT}"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

# Ensure frozen Track K scores exist
if [[ ! -f "$ROOT/results/line_k/probe_scores.jsonl" ]]; then
  python3 -m ccer.pipelines.run_line_k_claim_filter --n-boot 500 --seed 42
fi

python3 -m ccer.pipelines.run_line_l_claim_filter --n-boot 500 --seed 42
