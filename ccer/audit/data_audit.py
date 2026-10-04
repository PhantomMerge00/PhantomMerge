"""Full data audit with PM_3 document value diff (§5)."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ccer.io_utils import load_json, load_jsonl, write_json
from ccer.paths import (
    DATA_AUDIT_JSON,
    EXPERT_INSTANCES,
    GOLD_MANIFEST,
    NORMALIZED_SHOPPING,
    REPORTS,
    TRAJECTORY_SHEET,
)

REPORTS.mkdir(parents=True, exist_ok=True)

PM3_DOC_VALUES = {
    "formal_trajectories": 1995,
    "calibration_trajectories": 5,
    "evaluable_trajectories": 1947,
    "damaged_or_incomplete": 48,
    "pm_core_trajectories": 924,
    "clean_trajectories": 1023,
    "pm_core_rate": 924 / 1947,
    "cem_instances": 332,
    "cap_instances": 701,
    "ah_instances": 80,
    "total_pm_instances": 1113,
}


def _cross_tab_action_text(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tab: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        rel = r.get("commitment", {}).get("commitment_relation", "unknown")
        outcome = r.get("trajectory_outcome", "unknown")
        has_pm = outcome == "has_pm_core"
        tab[rel]["total"] += 1
        if has_pm:
            tab[rel]["pm"] += 1
    out = {}
    for rel, c in tab.items():
        n = c["total"]
        pm = c["pm"]
        out[rel] = {"n": n, "pm_trajectories": pm, "pm_rate": pm / n if n else None}
    return out


def _slot_pm_rates(instances: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_slot: dict[str, Counter] = defaultdict(Counter)
    for inst in instances:
        slot = str(inst.get("slot") or "unknown")
        by_slot[slot]["total"] += 1
        if inst.get("pm_binary") == "PM":
            by_slot[slot]["pm"] += 1
    rows = []
    for slot, c in sorted(by_slot.items(), key=lambda x: -x[1]["total"]):
        n = c["total"]
        pm = c["pm"]
        rows.append({"slot": slot, "n_instances": n, "pm_instances": pm, "pm_rate": pm / n if n else None})
    return rows


def run_data_audit(
    normalized_path: Path = NORMALIZED_SHOPPING,
    out_json: Path = DATA_AUDIT_JSON,
) -> dict[str, Any]:
    rows = load_jsonl(normalized_path) if normalized_path.is_file() else []
    traj_sheet = load_jsonl(TRAJECTORY_SHEET)
    instances = load_jsonl(EXPERT_INSTANCES)
    gold_manifest = load_json(GOLD_MANIFEST) if GOLD_MANIFEST.is_file() else {}

    n_traj = len(traj_sheet)
    n_cal = sum(1 for r in traj_sheet if "cal" in str(r.get("notes") or "").lower())
    n_formal = n_traj - n_cal
    outcomes = Counter(r.get("trajectory_outcome") for r in traj_sheet)
    pm_core_traj = outcomes.get("has_pm_core", 0)
    clean_traj = outcomes.get("clean", 0)
    verdicts = Counter(i.get("gold_verdict") for i in instances)
    n_instances = len(instances)
    pm_instances = sum(1 for i in instances if i.get("pm_binary") == "PM")

    computed = {
        "trajectory_sheet_n": n_traj,
        "formal_trajectories_est": n_formal,
        "calibration_trajectories_est": n_cal,
        "evaluable_trajectories_est": len(rows),
        "missing_rollout_join": n_traj - len(rows),
        "pm_core_trajectories": pm_core_traj,
        "clean_trajectories": clean_traj,
        "pm_core_rate_among_gold": pm_core_traj / n_traj if n_traj else None,
        "instance_n": n_instances,
        "pm_instance_n": pm_instances,
        "cem_instances": verdicts.get("cross_object_merge", 0),
        "cap_instances": verdicts.get("constraint_projection", 0),
        "ah_instances": verdicts.get("anchored_hallucination", 0),
        "cb_instances": verdicts.get("correct_binding", 0),
        "eligible_native_replay": sum(1 for r in rows if r.get("eligible", {}).get("native_replay")),
        "eligible_counterfactual": sum(1 for r in rows if r.get("eligible", {}).get("counterfactual")),
    }

    diff_explanations = []
    for key, doc_val in PM3_DOC_VALUES.items():
        comp_key = {
            "formal_trajectories": "formal_trajectories_est",
            "calibration_trajectories": "calibration_trajectories_est",
            "evaluable_trajectories": "evaluable_trajectories_est",
            "pm_core_trajectories": "pm_core_trajectories",
            "clean_trajectories": "clean_trajectories",
            "cem_instances": "cem_instances",
            "cap_instances": "cap_instances",
            "ah_instances": "ah_instances",
        }.get(key)
        if comp_key and comp_key in computed:
            cv = computed[comp_key]
            if cv != doc_val:
                diff_explanations.append(
                    {
                        "metric": key,
                        "pm3_document_value": doc_val,
                        "computed_value": cv,
                        "delta": cv - doc_val if isinstance(cv, (int, float)) and isinstance(doc_val, (int, float)) else None,
                        "note": _diff_note(key, doc_val, cv, gold_manifest),
                    }
                )

    audit = {
        "computed": computed,
        "gold_manifest_self_report": gold_manifest,
        "pm3_document_values": PM3_DOC_VALUES,
        "diff_explanations": diff_explanations,
        "action_text_cross_tab": _cross_tab_action_text(rows),
        "slot_pm_rates": _slot_pm_rates(instances),
        "exclusion_reason_counts": Counter(
            ex for r in rows for ex in r.get("exclusion_reason") or []
        ),
        "eligible_coverage": {
            k: sum(1 for r in rows if r.get("eligible", {}).get(k))
            for k in ("semantic_eval", "native_replay", "counterfactual", "patching")
        },
    }
    write_json(out_json, audit)

    md_path = REPORTS / "data_audit.md"
    md_path.write_text(_render_md(audit), encoding="utf-8")
    return audit


def _diff_note(key: str, doc: Any, comp: Any, manifest: dict) -> str:
    notes = {
        "formal_trajectories": "Gold pack has 2000 trajectory_sheet rows; PM_3 may exclude 5 calibration separately.",
        "pm_core_trajectories": f"merged_gold manifest reports pm_core_trajectories={manifest.get('pm_core_trajectories')}.",
        "cap_instances": "CAP count differs if AH/CAP boundary or instance dedupe changed in v2 merge.",
        "evaluable_trajectories": "Computed from successful rollout join, not hardcoded.",
    }
    return notes.get(key, "Recomputed from raw gold + rollout; see computed fields.")


def _render_md(audit: dict[str, Any]) -> str:
    lines = ["# CCER data_audit", "", "## Computed counts", ""]
    for k, v in audit["computed"].items():
        lines.append(f"- **{k}**: {v}")
    lines.extend(["", "## PM_3 diff explanations", ""])
    for d in audit["diff_explanations"]:
        lines.append(f"- {d['metric']}: doc={d['pm3_document_value']} computed={d['computed_value']} — {d['note']}")
    lines.extend(["", "## Action–text cross-tab", ""])
    for rel, stats in audit["action_text_cross_tab"].items():
        lines.append(f"- {rel}: n={stats['n']} pm_rate={stats.get('pm_rate')}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    print(json.dumps(run_data_audit(), indent=2, default=str))
