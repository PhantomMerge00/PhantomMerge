#!/usr/bin/env bash
# Plan B parallel track 2 — zero GPU, offline evidence packaging.
set -euo pipefail

ROOT="${PHANTOM_MERGE_ROOT}"
cd "$ROOT"

echo "[plan_b] Running Plan B evidence packaging..."
python3 ccer/pipelines/run_plan_b.py

echo "[plan_b] Regenerating CAP-only IIA summary (merge-shards, no GPU)..."
python3 ccer/pipelines/run_p3b.py --merge-shards --track CAP

echo "[plan_b] Copying CAP IIA summary to plan_b directory..."
mkdir -p results/plan_b
cp -f results/p3/iia_summary.json results/plan_b/cap_iia_summary.json

echo "[plan_b] Done. Expert brief:"
ls -1 results/reports/plan_b_expert_brief.md
