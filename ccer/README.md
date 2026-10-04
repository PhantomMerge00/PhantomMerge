# CCER P0–P2 Implementation

Commitment-Conditioned Evidence Routing pipeline for Phantom Merge.

## Layout

- `schema/` — JSON Schema + Python validators (§3)
- `normalize/` — Shopping + FHIR trajectory normalization
- `audit/` — data_audit, model_manifest, split_manifest
- `replay/` — vLLM replay, fixed_history harness, pilot selection
- `counterfactual/` — 8 evidence-layer operators + CEM/CAP bundles
- `repair/` — B0–B4 / M0 baselines
- `eval/` — P2 metrics
- `pipelines/` — run_p0, run_p1_dev, run_p2_dev

## Run

```bash
export PYTHONPATH=${PHANTOM_MERGE_ROOT}

# P0: normalize + audit + split + replay pilot
python3 ccer/pipelines/run_p0.py

# P1: dev cohort counterfactuals (requires vLLM on :8003)
python3 ccer/pipelines/run_p1_dev.py

# P1 offline: operator validation only (no generation)
python3 ccer/pipelines/run_p1_dev.py --offline

# P2: repair comparison (requires vLLM)
python3 ccer/pipelines/run_p2_dev.py --limit 10
```

## Artifacts

Outputs under `results/` and reports under `results/reports/`.
