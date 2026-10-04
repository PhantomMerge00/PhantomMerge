"""Frozen paths for CCER P0–P2 pipeline."""
from __future__ import annotations

import os
from pathlib import Path

# Override on remote machines: export PHANTOM_MERGE_ROOT=/path/to/the repository
ROOT = Path(
    os.environ.get(
        "PHANTOM_MERGE_ROOT",
        str(Path(__file__).resolve().parents[1]),
    )
).resolve()
ACTIVE = ROOT
CCER = ACTIVE / "ccer"
ARTIFACTS = ROOT / "results"
REPORTS = ROOT / "results/reports"

SHOPPING_ROLLOUT = (
    ROOT / "data/rollouts/shopping/shopping_qwen3-32b_rollout/rollout.jsonl"
)
SHOPPING_METADATA = (
    ROOT
    / "data/rollouts/shopping/shopping_qwen3-32b_rollout/rollout.run_metadata.json"
)
GOLD_DIR = (
    ROOT
    / "data/gold/gold/shopping/merged_v1"
)
TRAJECTORY_SHEET = GOLD_DIR / "trajectory_sheet.jsonl"
EXPERT_INSTANCES = GOLD_DIR / "expert_instances.jsonl"
GOLD_MANIFEST = GOLD_DIR / "manifest.json"

FHIR_ROLLOUT_PATHS = [
    ROOT / "data/rollouts/fhir/fhir_qwen3-32b_rollout/shard_a/rollout_unique.jsonl",
    ROOT / "data/rollouts/fhir/fhir_qwen3-32b_rollout/shard_b/rollout_unique.jsonl",
]
FHIR_METADATA = (
    ROOT
    / "data/rollouts/fhir/fhir_qwen3-32b_rollout/shard_a/rollout.jsonl.run_metadata.json"
)

NORMALIZED_SHOPPING = ARTIFACTS / "normalized/shopping_trajectories.jsonl"
NORMALIZED_FHIR = ARTIFACTS / "normalized/fhir_trajectories.jsonl"

DATA_AUDIT_JSON = ARTIFACTS / "data_audit.json"
MODEL_MANIFEST_JSON = ARTIFACTS / "model_manifest.json"
SPLIT_MANIFEST_JSON = ARTIFACTS / "split_manifest.json"
COHORT_MANIFEST_JSON = ARTIFACTS / "cohort_manifest.json"
ADJUDICATION_JSONL = (
    ROOT / "results/reports/manual_verification/cem_ah_final_adjudication.jsonl"
)
COHORT_MANIFEST_V1_JSON = ARTIFACTS / "cohort_manifest.v1.json"
REPLAY_AUDIT_JSON = ARTIFACTS / "replay_audit.json"

P1_DIR = ARTIFACTS / "p1"
P1_SEQUENCE_SCORES = P1_DIR / "sequence_scores.jsonl"
P2_DIR = ARTIFACTS / "p2"
P3_DIR = ARTIFACTS / "p3"
P3_ACTIVATIONS = P3_DIR / "activations"
P3_SEQUENCE_SCORES = P3_DIR / "sequence_scores.jsonl"
P3_BINDING_ROI = P3_DIR / "binding_roi.json"
P3_U_OWNER_NPZ = P3_DIR / "U_owner.npz"
P3_INTERCHANGE_ROWS = P3_DIR / "interchange_rows.jsonl"
P3_IIA_SUMMARY = P3_DIR / "iia_summary.json"
P3_0_MANIFEST = REPORTS / "p3_0_manifest.json"

PLAN_B_DIR = ARTIFACTS / "plan_b"
PLAN_B_CAP_IIA_SUMMARY = PLAN_B_DIR / "cap_iia_summary.json"
PLAN_B_CAP_CONTROL_TABLE = PLAN_B_DIR / "cap_control_table.json"
PLAN_B_MISMATCH_JSON = PLAN_B_DIR / "mismatch_stats.json"
PLAN_B_CAP_EVIDENCE_JSON = PLAN_B_DIR / "cap_evidence.json"
PLAN_B_EXPERT_BRIEF = REPORTS / "plan_b_expert_brief.md"
INCREMENTAL_ADJUDICATION_JSONL = ROOT / "shopping_incremental_adjudication.jsonl"

LINE_A_DIR = ARTIFACTS / "line_a"
LINE_A_SKIP_TRAJECTORIES = LINE_A_DIR / "p3_0_skip_trajectories.json"
LINE_A_INSTANCE_ACTIVATIONS = LINE_A_DIR / "instance_activations"
LINE_A_V3_DIR = ARTIFACTS / "line_a/v3"
LINE_A_V3_COHORT_MANIFEST = LINE_A_V3_DIR / "cohort_manifest.json"
LINE_A_V3_PROBE_SUMMARY = LINE_A_V3_DIR / "probe_vs_bow_summary.json"
LINE_K_DIR = ARTIFACTS / "line_k"
LINE_K_SUMMARY = LINE_K_DIR / "claim_filter_summary.json"
LINE_K_REPORT = LINE_K_DIR / "TASK_K_REPORT.md"
LINE_K_PROBE_SCORES = LINE_K_DIR / "probe_scores.jsonl"
LINE_K_DEV_THRESHOLDS = LINE_K_DIR / "dev_thresholds.json"
LINE_L_DIR = ARTIFACTS / "line_l"
LINE_L_SUMMARY = LINE_L_DIR / "claim_filter_summary.json"
LINE_L_REPORT = LINE_L_DIR / "TASK_L_REPORT.md"
LINE_L_REWRITE_LOG = LINE_L_DIR / "rewrite_log.jsonl"
LINE_L_REWRITE_SAMPLE = LINE_L_DIR / "rewrite_texts_sample.jsonl"
LINE_L_DIAGNOSTIC_REPORT = LINE_L_DIR / "LINE_L_DIAGNOSTIC.md"
LINE_L_DIAGNOSTIC_SUMMARY = LINE_L_DIR / "diagnostic_summary.json"
LINE_L_CEM_MISS_SAMPLES = LINE_L_DIR / "cem_miss_samples.jsonl"
LINE_L_PLUS_DIR = ARTIFACTS / "line_l_plus"
LINE_L_PLUS_SUMMARY = LINE_L_PLUS_DIR / "method_comparison_summary.json"
LINE_L_PLUS_REPORT = LINE_L_PLUS_DIR / "METHOD_COMPARISON.md"
LINE_L_PLUS_EVAL_REPORT = LINE_L_PLUS_DIR / "EVAL_BASELINES_REPORT.md"
LINE_L_PLUS_REWRITE_LOG = LINE_L_PLUS_DIR / "rewrite_log.jsonl"
PRISM_DIR = ARTIFACTS / "prism"
PRISM_AUDIT_JSONL = PRISM_DIR / "prism_audit.jsonl"
PRISM_SUMMARY = PRISM_DIR / "prism_summary.json"
PRISM_DIAGNOSIS_REPORT = PRISM_DIR / "PRISM_DIAGNOSIS_REPORT.md"
PRISM_MITIGATION_SUMMARY = PRISM_DIR / "mitigation_summary.json"
PRISM_REWRITE_LOG = PRISM_DIR / "rewrite_log_prism.jsonl"
PRISM_TASK_REPORT = PRISM_DIR / "TASK_PRISM_REPORT.md"
PRISM_L_PLUS_DIR = ARTIFACTS / "prism_l_plus"
PRISM_L_PLUS_SUMMARY = PRISM_L_PLUS_DIR / "method_comparison_summary.json"
PRISM_L_PLUS_REPORT = PRISM_L_PLUS_DIR / "METHOD_COMPARISON.md"
PRISM_L_PLUS_REWRITE_LOG = PRISM_L_PLUS_DIR / "rewrite_log.jsonl"
PRISM_L_PLUS_FORENSIC = PRISM_L_PLUS_DIR / "FORENSIC_CASES.md"
PRISM_L_PLUS_FORENSIC_JSON = PRISM_L_PLUS_DIR / "forensic_summary.json"
PRISM_L_PLUS_CACHE = PRISM_L_PLUS_DIR / "cache"
E2E_MITIGATION_DIR = ARTIFACTS / "e2e_mitigation"
E2E_MITIGATION_SUMMARY = E2E_MITIGATION_DIR / "method_comparison_summary.json"
E2E_MITIGATION_REPORT = E2E_MITIGATION_DIR / "METHOD_COMPARISON_E2E.md"
E2E_MITIGATION_REWRITE_LOG = E2E_MITIGATION_DIR / "rewrite_log.jsonl"
PRISM_L_PLUS_APPENDIX_REPORT = PRISM_L_PLUS_DIR / "APPENDIX_PROBE_GATED.md"
SONDE_LINEAR_PROBES = ROOT / "third_party/sonde_linear_probes"

AGR_DIR = ARTIFACTS / "agr"
AGR_SPLIT_MANIFEST = AGR_DIR / "split_manifest_agr.json"
AGR_SUMMARY = AGR_DIR / "AGR_VALIDATION_SUMMARY.json"
AGR_REPORT = AGR_DIR / "AGR_VALIDATION_REPORT.md"
AGR_ANCHOR_VALUE_MASS_CACHE = AGR_DIR / "anchor_value_mass_cache.jsonl"

TELECOM_GOLD_EXPORT = (
    ROOT
    / "data/rollouts/tau3_exports/telecom_qwen_1k_pm_gold_export/telecom_qwen_1k_pm_gold.jsonl"
)
NORMALIZED_TELECOM = ARTIFACTS / "normalized/telecom_trajectories.jsonl"

TAU3_DIR = ARTIFACTS / "tau3"
TAU3_TELECOM_DIR = TAU3_DIR / "telecom"
TAU3_ADJUDICATION = TAU3_DIR / "telecom_adjudication.jsonl"
TAU3_LINE_A_DIR = TAU3_DIR / "line_a"
TAU3_INSTANCE_ACTIVATIONS = TAU3_LINE_A_DIR / "instance_activations"
TAU3_SLOT_READOUT = TAU3_DIR / "prism_slot_readout.jsonl"
TAU3_SHOPPING_FROZEN_PROBE = TAU3_DIR / "shopping_frozen_probe.joblib"
TAU3_AGR_SPLIT_MANIFEST_TELECOM = TAU3_TELECOM_DIR / "split_manifest_agr_telecom.json"
TAU3_ZEROSHOT_SUMMARY = TAU3_TELECOM_DIR / "agr_slot_zeroshot_summary.json"
TAU3_INDOMAIN_SUMMARY = TAU3_TELECOM_DIR / "agr_slot_indomain_summary.json"
TAU3_CROSS_DOMAIN_REPORT = TAU3_TELECOM_DIR / "TAU3_TELECOM_CROSS_DOMAIN_REPORT.md"

AIRLINE_GOLD_EXPORT = (
    ROOT
    / "data/rollouts/tau3_exports/airline_qwen_1k_pm_gold_export/airline_qwen_1k_pm_gold.jsonl"
)
NORMALIZED_AIRLINE_QWEN_1K = ARTIFACTS / "normalized/airline_qwen_1k_trajectories.jsonl"
TAU3_AIRLINE_DIR = TAU3_DIR / "airline"
TAU3_ADJUDICATION_AIRLINE = TAU3_DIR / "airline_adjudication.jsonl"
TAU3_AIRLINE_ACTIVATIONS = TAU3_AIRLINE_DIR / "line_a/instance_activations"
TAU3_SLOT_READOUT_AIRLINE = TAU3_AIRLINE_DIR / "prism_slot_readout.jsonl"
TAU3_AGR_SPLIT_MANIFEST_AIRLINE = TAU3_AIRLINE_DIR / "split_manifest_agr_airline.json"
TAU3_ZEROSHOT_SUMMARY_AIRLINE = TAU3_AIRLINE_DIR / "agr_slot_zeroshot_summary.json"
TAU3_INDOMAIN_SUMMARY_AIRLINE = TAU3_AIRLINE_DIR / "agr_slot_indomain_summary.json"
TAU3_CROSS_DOMAIN_REPORT_AIRLINE = TAU3_AIRLINE_DIR / "TAU3_AIRLINE_CROSS_DOMAIN_REPORT.md"

RETAIL_GOLD_EXPORT = (
    ROOT
    / "data/rollouts/tau3_exports/retail_qwen_1k_pm_gold_export/retail_qwen_1k_pm_gold.jsonl"
)
NORMALIZED_RETAIL_QWEN_1K = ARTIFACTS / "normalized/retail_qwen_1k_trajectories.jsonl"
TAU3_RETAIL_DIR = TAU3_DIR / "retail"
TAU3_ADJUDICATION_RETAIL = TAU3_DIR / "retail_adjudication.jsonl"
TAU3_RETAIL_ACTIVATIONS = TAU3_RETAIL_DIR / "line_a/instance_activations"
TAU3_SLOT_READOUT_RETAIL = TAU3_RETAIL_DIR / "prism_slot_readout.jsonl"
TAU3_AGR_SPLIT_MANIFEST_RETAIL = TAU3_RETAIL_DIR / "split_manifest_zeroshot_test.json"
TAU3_ZEROSHOT_SUMMARY_RETAIL = TAU3_RETAIL_DIR / "agr_slot_zeroshot_summary.json"
TAU3_CROSS_DOMAIN_REPORT_RETAIL = TAU3_RETAIL_DIR / "TAU3_RETAIL_CROSS_DOMAIN_REPORT.md"

TAU3_MITIGATION_DIR = TAU3_DIR / "mitigation"
TAU3_MITIGATION_PLAN = TAU3_DIR / "TAU3_MITIGATION_EXPERIMENT_PLAN.md"
TAU3_E2E_MITIGATION_DIR = TAU3_MITIGATION_DIR / "e2e"
TAU3_E2E_MITIGATION_POOLED_SUMMARY = TAU3_E2E_MITIGATION_DIR / "pooled_method_comparison_summary.json"
TAU3_E2E_MITIGATION_POOLED_REPORT = TAU3_E2E_MITIGATION_DIR / "METHOD_COMPARISON_E2E_POOLED.md"


def tau3_e2e_mitigation_paths(domain: str) -> dict[str, Path]:
    d = str(domain).strip().lower()
    base = TAU3_E2E_MITIGATION_DIR / d
    return {
        "dir": base,
        "summary": base / "method_comparison_summary.json",
        "report": base / "METHOD_COMPARISON_E2E.md",
        "rewrite_log": base / "rewrite_log.jsonl",
    }


TAU3_PROBE_GATED_MITIGATION_DIR = TAU3_MITIGATION_DIR / "probe_gated"
TAU3_PROBE_GATED_MITIGATION_POOLED_SUMMARY = TAU3_PROBE_GATED_MITIGATION_DIR / "POOLED_SUMMARY.json"


def tau3_probe_gated_mitigation_paths(domain: str) -> dict[str, Path]:
    d = str(domain).strip().lower()
    base = TAU3_PROBE_GATED_MITIGATION_DIR / d
    return {
        "dir": base,
        "summary": base / "method_comparison_summary.json",
        "report": base / "METHOD_COMPARISON.md",
        "rewrite_log": base / "rewrite_log.jsonl",
    }


JSPACE_LENS_L49 = ARTIFACTS / "jspace" / "J_l_qwen32b_L49.pt"

DEFAULT_VLLM_BASE_URL = "http://127.0.0.1:8003/v1"
ANNOTATION_VERSION = "shopping_expert_v2_merged_gold_v1"
SCHEMA_VERSION = "ccer_v1"
