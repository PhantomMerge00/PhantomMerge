"""Offline gate preflight: adjudication pools + operator validation before P1/P3 GPU runs."""
from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path("${PHANTOM_MERGE_ROOT}")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ccer.adjudication.loader import load_adjudication_index, resolve_cem_swap_target
from ccer.counterfactual.bundles import build_ah_bundle, build_cap_bundle, build_cem_bundle
from ccer.io_utils import load_json, load_jsonl, write_json
from ccer.paths import COHORT_MANIFEST_JSON, NORMALIZED_SHOPPING, REPORTS

REPORT_MD = REPORTS / "gate_preflight_v2.md"
REPORT_JSON = REPORTS / "gate_preflight_v2.json"


def _validate_condition(traj: dict[str, Any], bundle_fn, condition_id: str) -> dict[str, Any]:
    for bundle_name, op_fn, kwargs in bundle_fn(traj):
        if bundle_name == "original" or op_fn is None or bundle_name != condition_id:
            continue
        op_result = op_fn(traj["messages_final_call"], **kwargs)
        return {
            "condition_id": op_result.condition_id,
            "valid": not op_result.invalid_reason and op_result.semantic_validation != "invalid",
            "invalid_reason": op_result.invalid_reason,
        }
    return {"condition_id": condition_id, "valid": False, "invalid_reason": "bundle_empty"}


def run_preflight() -> dict[str, Any]:
    manifest = load_json(COHORT_MANIFEST_JSON)
    rows = {r["trajectory_id"]: r for r in load_jsonl(NORMALIZED_SHOPPING)}
    index = load_adjudication_index()

    cem_tids = manifest.get("cohorts", {}).get("CEM", [])
    ah_tids = manifest.get("cohorts", {}).get("AH", [])
    cap_tids = manifest.get("cohorts", {}).get("CAP", [])

    cem_results: list[dict[str, Any]] = []
    for tid in cem_tids:
        traj = rows.get(tid)
        if not traj:
            cem_results.append({"trajectory_id": tid, "valid": False, "invalid_reason": "missing_trajectory"})
            continue
        target = resolve_cem_swap_target(traj, index)
        val = _validate_condition(traj, build_cem_bundle, "rival_value_swap")
        old_rival = str((traj.get("text_anchor") or {}).get("compared_pid") or "")
        cem_results.append(
            {
                "trajectory_id": tid,
                **val,
                "donor_pid": target.donor_pid if target else None,
                "old_compared_pid": old_rival,
                "donor_eq_old_rival": bool(target and target.donor_pid == old_rival),
            }
        )

    ah_results: list[dict[str, Any]] = []
    for tid in ah_tids:
        traj = rows.get(tid)
        if not traj:
            ah_results.append({"trajectory_id": tid, "valid": False, "invalid_reason": "missing_trajectory"})
            continue
        val = _validate_condition(traj, build_ah_bundle, "anchor_value_swap")
        ah_results.append({"trajectory_id": tid, **val})

    cap_sample = cap_tids[:5]
    cap_valid = 0
    for tid in cap_sample:
        traj = rows.get(tid)
        if not traj:
            continue
        val = _validate_condition(traj, build_cap_bundle, "query_value_swap")
        if val.get("valid"):
            cap_valid += 1

    cem_scorable = sum(1 for r in cem_results if r.get("valid"))
    ah_scorable = sum(1 for r in ah_results if r.get("valid"))
    donor_aligned = sum(1 for r in cem_results if r.get("donor_eq_old_rival"))

    invalid_buckets = Counter(
        r.get("invalid_reason") or "ok" for r in cem_results if not r.get("valid")
    )

    summary = {
        "schema_version": "ccer_gate_preflight_v2",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "label_authority": manifest.get("label_authority"),
        "cohort_schema": manifest.get("schema_version"),
        "pools": {
            "CEM": {"n": len(cem_tids), "rival_value_swap_scorable": cem_scorable},
            "AH": {"n": len(ah_tids), "anchor_value_swap_scorable": ah_scorable},
            "CAP": {"n": len(cap_tids), "query_value_swap_sample_valid": f"{cap_valid}/{len(cap_sample)}"},
        },
        "cem_donor_alignment": {
            "donor_eq_old_compared_pid": donor_aligned,
            "donor_ne_old_compared_pid": cem_scorable - donor_aligned if cem_scorable else 0,
            "note": "v2 targets donor_pid from adjudication, not compared_pid",
        },
        "cem_invalid_buckets": dict(invalid_buckets),
        "gates": {
            "G0_label_authority": manifest.get("schema_version") == "ccer_cohort_v2",
            "G1_cem_pool_nonempty": len(cem_tids) >= 20,
            "G2_cem_operator_scorable": cem_scorable >= 15,
            "G3_ah_pool_nonempty": len(ah_tids) >= 5,
            "G4_cap_baseline": cap_valid >= 3,
        },
        "ready_for_p1_refresh": all(
            [
                len(cem_tids) >= 20,
                cem_scorable >= 15,
                len(ah_tids) >= 5,
            ]
        ),
    }

    write_json(REPORT_JSON, {"summary": summary, "cem": cem_results, "ah": ah_results})

    lines = [
        "# Gate Preflight v2（adjudication-aware）",
        "",
        f"- 时间: {summary['timestamp']}",
        f"- 标签权威: `{manifest.get('label_authority')}`",
        f"- Cohort schema: `{manifest.get('schema_version')}`",
        "",
        "## 机制池 + 算子离线验证",
        "",
        "| 队列 | pool n | scorable (offline) |",
        "|---|---:|---:|",
        f"| CEM rival_value_swap | {len(cem_tids)} | **{cem_scorable}** |",
        f"| AH anchor_value_swap | {len(ah_tids)} | **{ah_scorable}** |",
        f"| CAP query_value_swap (sample 5) | {len(cap_tids)} | {cap_valid}/5 |",
        "",
        "## CEM donor 对齐（v1→v2）",
        "",
        f"- donor == 旧 compared_pid: **{donor_aligned}/{cem_scorable}** scorable",
        f"- donor ≠ 旧 compared_pid: **{cem_scorable - donor_aligned}**（预期多数，算子已切换 donor）",
        "",
        "## Invalid 桶（CEM 不可 scorable）",
        "",
        "| reason | n |",
        "|---|---:|",
    ]
    for reason, n in invalid_buckets.most_common():
        lines.append(f"| `{reason}` | {n} |")

    lines.extend(["", "## Gate 判定", ""])
    for gate, ok in summary["gates"].items():
        lines.append(f"- `{gate}`: **{'PASS' if ok else 'FAIL'}**")
    lines.extend(
        [
            "",
            f"**ready_for_p1_refresh**: `{summary['ready_for_p1_refresh']}`",
            "",
            "## 下一步",
            "",
            "1. `python ccer/pipelines/run_p1_dev.py --cohort CEM --offline --fresh --refresh-conditions rival_value_swap,source_null`",
            "2. 通过后 GPU 重跑 P1 CEM + AH",
            "3. P1 follow 达标后启动 P3 attention path patching",
            "",
        ]
    )
    REPORTS.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    return summary


def main() -> None:
    print(json.dumps(run_preflight(), indent=2))


if __name__ == "__main__":
    main()
