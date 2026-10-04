#!/usr/bin/env bash
# Track K: probe-guided claim filtering (Line A v3 probe, CPU only).
set -euo pipefail
ROOT="${PHANTOM_MERGE_ROOT}"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
python3 -m ccer.pipelines.run_line_k_claim_filter --n-boot 2000 --seed 42
