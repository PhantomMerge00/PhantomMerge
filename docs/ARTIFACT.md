# Artifact guide

Single reference for methods, metrics, data, provenance, and paper-to-code mapping. PDF: [paper.pdf](paper.pdf). Submission reference numbers (not computed outputs): [reference_values.json](reference_values.json).

## 1. Repository layout

```text
ccer/            Detection, AGR/FAC mechanisms, pipelines
scripts/         Experiment and table drivers
configs/         Models, datasets, method registry
results/         Frozen scores, splits, tables (checksums in results/MANIFEST.json)
reproduce/       Env, data-sync, shell scripts; Level-1 CLI (phantom-merge)
third_party/     J-lens, RARR
assets/          Framework figure (framework.png for README; framework.pdf vector)
tests/
```

Set `export PHANTOM_MERGE_ROOT="$(pwd)"` to the clone root before running Python or shell scripts.

## 2. Reproduction

**Level 1 (CPU, no weights).**

```bash
pip install -e '.[reproduce]'
phantom-merge reproduce --bundle results --out outputs/metrics
phantom-merge verify --bundle results --results outputs/metrics
```

Reads `results/rq2/table2_prf_audit/per_claim_scores_test274.jsonl` (274 claims, 156 PM, 118 CB).

**Level 2–3.** Gold, rollouts, activations, and GPUs: see `reproduce/DATA_SYNC_MANIFEST.json` and `reproduce/REPLICATION.md`.

## 3. Methods (code mapping)

Registry: `configs/method_registry.yaml`.

| Name | Score field | Notes |
|------|-------------|--------|
| Representational | `p_pm` | Probe at claim onset; Shopping mitigation gate `p_pm > 0.05` |
| Jacobian-slot | `log_slot_mass` | J-lens slot mass only (no probe logit) |
| AGR-value | `agr_rho_logit` | Fusion logit; σ(·) for probability view; 48/274 test with anchor-value, 226 fallback |
| AGR-slot | `bind_surprise` | ℓ_pm + log slot mass; Table 2 headline; not TARGET_FUSION crossover (~0.915) |
| FAC | — | `prism_l_plus` stack; shared gate uses `p_pm`, not BindSurprise |

Implementation: `ccer/mechanism/` (`bind_surprise.py`, `agr/fusion.py`, `prism_l_plus.py`). Diagram: `assets/framework.pdf` (illustration only).

## 4. Evaluation protocol

- **Labels:** PM = binding failure; CB = correctly bound.
- **Ranking:** AUROC and average precision (scikit-learn); higher score = higher risk unless noted. Paper bootstrap: trajectory clusters, B=2000, seed=42 where stated.
- **Thresholds:** τ=0.05 for native probabilities and the Shopping mitigation gate only. Not for raw `bind_surprise` or `log_slot_mass`. Detection τ on D_f / dev: `results/rq2/table2_prf_audit/TABLE2_MISSING_PRF_AUDIT.json`.
- **Mitigation:** Branch counts in `results/rq3/FAC_BRANCH_STATS.json`. Output-claim ratio R: aggregation still under audit (see reference JSON).
- **Figures:** Replots use true coordinates; submitted Fig. 5 used display jitter. Fig. 6 gates are not inferred from detection τ.

## 5. Data

**In git:** `results/` manifests and frozen Shopping test scores (`results/MANIFEST.json`).

**External:** Benchmarks, rollouts, activations, weights per `reproduce/DATA_SYNC_MANIFEST.json`. Join keys: `instance_audit_key`, `trajectory_id`.

## 6. Provenance

Labels: expert adjudication and model-assisted pipelines (`ccer/adjudication/`). Model labels are not written into human gold.

**57 copy-rewrites:** Reported in the paper; per-claim adjudication notes are not redistributed in this anonymous bundle.

## 7. Paper-to-code map

| Paper item | Location | Status |
|------------|----------|--------|
| Table 2, App. C.1 | `results/rq2/table2_prf_audit/`, `results/agr/DETECTOR_COMPARISON_SUMMARY.json`, `results/prism/prism_audit.jsonl` | L1 reproduced |
| App. C.2 strata | `results/shopping/template_stratified_auroc.json` | reproduced |
| App. D thresholds | `TABLE2_MISSING_PRF_AUDIT.json` | reproduced |
| Fig. 2 / ROC | `per_claim_scores_test274.jsonl` | partial |
| Fig. 3 cross-domain | — | missing configs |
| Figs. 4–6, App. G | `FAC_BRANCH_STATS.json`, `prism_l_plus/` summaries | partial |
| Table 1 (4 backbones) | pending export | missing |
| TARGET_FUSION | `results/agr/TARGET_FUSION_CROSSOVER.json` | reference only |

## 8. Known issues

Manuscript: duplicate “AGR-slot score” wording; AP vs AUPRC naming; Fig. 5 jitter; Fig. 6 representational CB retention 73% in latest PDF.

Release: LICENSE not finalized; Fig. 3 / Fig. 6 / R aggregation incomplete; four-backbone characterization partial.
