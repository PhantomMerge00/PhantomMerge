# Model profiles (Qwen / Llama / Mixtral)

Paper numbers on this machine use **`qwen3-32b_shopping_v1`**. On another device with Llama/Mixtral rollouts, add a profile JSON and point pipelines at your rollout + weights.

## Steps for a new backbone

1. Place rollouts under `data/rollouts/shopping/<profile_name>/rollout.jsonl`
2. Place τ³ gold exports under `data/rollouts/tau3_exports/<domain>_<profile>_pm_gold_export/`
3. Copy `qwen3-32b_shopping_v1.json` → `<your_profile>.json`, edit `model_id`, `weights_path`, `vllm.base_url`
4. Merge into `results/model_manifest.json` manifests dict
5. Export: `export CCER_MODEL_MANIFEST_ID=<your_profile>`
6. Re-run Phase B/C pipelines (activations + J-lens are **model-specific** — must recompute)

## Frozen across models (do NOT copy from Qwen)

- Gold labels (`data/gold/.../merged_gold_v1/`) — same tasks/claims
- Split manifests (`results/split_manifest.json`, `agr/split_manifest_agr.json`) — same trajectory IDs
- Adjudication keys (`shopping_incremental_adjudication.jsonl`) — same instance_audit_key

## Model-specific (must recompute per backbone)

- `results/line_a/**/instance_activations/`
- `results/jspace/J_l_*_L49.pt`
- `results/prism/prism_audit.jsonl` (BindSurprise scores)
- `results/line_k/probe_scores.jsonl`
- `results/tau3/**/instance_activations/`
- `results/tau3/shopping_frozen_probe.joblib` (τ³ uses shopping-trained probe)
