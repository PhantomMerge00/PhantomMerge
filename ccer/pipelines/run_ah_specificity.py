"""AH Tier-1 specificity check: swaps should NOT directionally follow."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path("${PHANTOM_MERGE_ROOT}")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ccer.counterfactual.bundles import build_ah_bundle
from ccer.eval.source_following import evaluate_source_following
from ccer.io_utils import load_json, load_jsonl, write_json
from ccer.paths import COHORT_MANIFEST_JSON, NORMALIZED_SHOPPING, REPORTS
from ccer.replay.fixed_history import run_fixed_history
from ccer.replay.vllm_replay import check_vllm_available

REPORT_MD = REPORTS / "ah_specificity_check.md"
REPORT_JSON = REPORTS / "ah_specificity_check.json"


def _primary_ah(traj: dict) -> bool:
    for c in traj.get("claims") or []:
        if c.get("legacy_label") == "anchored_hallucination":
            return True
        if c.get("legacy_label") in ("cross_object_merge", "constraint_projection"):
            return False
    return False


def ah_dev_ids() -> list[str]:
    manifest = load_json(COHORT_MANIFEST_JSON)
    if manifest.get("schema_version") == "ccer_cohort_v2":
        return list(manifest.get("cohorts", {}).get("AH", []))
    # legacy fallback
    from ccer.paths import SPLIT_MANIFEST_JSON

    split = load_json(SPLIT_MANIFEST_JSON)["splits"]
    rows = load_jsonl(NORMALIZED_SHOPPING)
    return [
        r["trajectory_id"]
        for r in rows
        if split.get(r["trajectory_id"]) == "dev"
        and (r.get("eligible") or {}).get("counterfactual")
        and _primary_ah(r)
    ]


def run_ah_specificity(*, dry_run: bool = False, offline: bool = False, limit: int | None = None) -> dict:
    tids = ah_dev_ids()
    if limit:
        tids = tids[:limit]
    rows = {r["trajectory_id"]: r for r in load_jsonl(NORMALIZED_SHOPPING)}
    vllm_ok = check_vllm_available()
    if not dry_run and not offline and not vllm_ok:
        offline = True

    results: list[dict] = []
    for tid in tids:
        traj = rows[tid]
        for bundle_name, op_fn, kwargs in build_ah_bundle(traj):
            if bundle_name == "original" or op_fn is None:
                continue
            op_result = op_fn(traj["messages_final_call"], **kwargs)
            cid = op_result.condition_id
            if op_result.invalid_reason or op_result.semantic_validation == "invalid":
                results.append(
                    {
                        "trajectory_id": tid,
                        "condition_id": cid,
                        "scorable": False,
                        "source_following": None,
                        "invalid_reason": op_result.invalid_reason,
                    }
                )
                continue
            if dry_run or offline:
                results.append(
                    {
                        "trajectory_id": tid,
                        "condition_id": cid,
                        "scorable": True,
                        "source_following": None,
                        "status": "not_run",
                    }
                )
                continue
            out = run_fixed_history(traj, op_result)
            if out["status"] == "invalid":
                results.append(
                    {
                        "trajectory_id": tid,
                        "condition_id": cid,
                        "scorable": False,
                        "source_following": None,
                        "invalid_reason": out["counterfactual"].get("invalid_reason"),
                    }
                )
                continue
            gen = out["generation"] or {}
            score = evaluate_source_following(
                condition_id=cid,
                original_answer=str(gen.get("original_answer") or ""),
                new_answer=str(gen.get("text") or ""),
                expected=op_result.expected_high_level_change,
                edit_manifest=op_result.edit_manifest,
            )
            results.append({"trajectory_id": tid, "cohort": "AH", "condition_id": cid, **score})

    scorable = [r for r in results if r.get("scorable")]
    directional = [r for r in scorable if r.get("source_following")]
    by_cond = Counter(r.get("condition_id") for r in scorable)
    summary = {
        "n_ah_dev": len(tids),
        "n_results": len(results),
        "n_scorable": len(scorable),
        "n_directional_follow": len(directional),
        "directional_follow_rate": len(directional) / max(1, len(scorable)),
        "by_condition_scorable": dict(by_cond),
        "passes_specificity": len(directional) == 0 and len(scorable) >= 5,
        "vllm_available": vllm_ok,
        "offline_mode": offline,
    }
    write_json(REPORT_JSON, {"summary": summary, "results": results})
    REPORTS.mkdir(parents=True, exist_ok=True)
    lines = [
        "# AH Tier-1 Specificity Check",
        "",
        f"- dev AH trajectories: {len(tids)}",
        f"- scorable swap tests: {len(scorable)}",
        f"- directional follow (should be 0): **{len(directional)}**",
        f"- passes_specificity: **{summary['passes_specificity']}**",
        "",
        "## By condition",
        "",
        json.dumps(dict(by_cond), indent=2),
        "",
        "> Expected: AH does not directionally follow query/rival swaps (mechanistic specificity control).",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    print(json.dumps(run_ah_specificity(dry_run=args.dry_run, offline=args.offline, limit=args.limit), indent=2))
