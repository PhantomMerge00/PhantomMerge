"""P1 dev cohort counterfactual run."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path("${PHANTOM_MERGE_ROOT}")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ccer.counterfactual.bundles import build_cap_bundle, build_cem_bundle
from ccer.eval.source_following import aggregate_following_stats, evaluate_source_following, validate_replay_output
from ccer.io_utils import append_jsonl, load_json, load_jsonl, write_json, write_jsonl
from ccer.paths import COHORT_MANIFEST_JSON, NORMALIZED_SHOPPING, P1_DIR
from ccer.replay.fixed_history import run_fixed_history
from ccer.replay.vllm_replay import check_vllm_available, vllm_generate

P1_GENERATIONS = P1_DIR / "generations.jsonl"
P1_COUNTERFACTUALS = P1_DIR / "counterfactuals.jsonl"
P1_INVALID = P1_DIR / "invalid_counterfactuals.jsonl"
P1_EFFECTS_ROWS = P1_DIR / "source_effects_rows.jsonl"
P1_SEQUENCE_SCORES = P1_DIR / "sequence_scores.jsonl"


def _generation_row(
    *,
    trajectory_id: str,
    cohort: str,
    condition_id: str,
    text: str,
    meta: dict[str, Any],
    original_answer: str | None = None,
) -> dict[str, Any]:
    _, harness_err = validate_replay_output(text)
    row: dict[str, Any] = {
        "trajectory_id": trajectory_id,
        "cohort": cohort,
        "condition_id": condition_id,
        "answer": text,
        "latency": meta,
        "harness_valid": harness_err is None,
    }
    if harness_err:
        row["harness_error"] = harness_err
    if original_answer is not None:
        row["original_answer"] = original_answer
    return row


def _run_key(trajectory_id: str, condition_id: str) -> tuple[str, str]:
    return trajectory_id, condition_id


class P1IncrementalWriter:
    """Append-after-each-record checkpointing for long P1 runs."""

    def __init__(self, *, resume: bool = True, fresh: bool = False) -> None:
        P1_DIR.mkdir(parents=True, exist_ok=True)
        self.paths = {
            "generations": P1_GENERATIONS,
            "counterfactuals": P1_COUNTERFACTUALS,
            "invalids": P1_INVALID,
            "effects": P1_EFFECTS_ROWS,
        }
        if fresh:
            for path in (*self.paths.values(), P1_SEQUENCE_SCORES, P1_DIR / "source_effects.json"):
                if path.is_file():
                    path.unlink()
        self.done_keys = self._load_done_keys() if resume else set()
        self.counts = {
            "generations": len(load_jsonl(P1_GENERATIONS)),
            "counterfactuals": len(load_jsonl(P1_COUNTERFACTUALS)),
            "invalids": len(load_jsonl(P1_INVALID)),
            "effects": len(load_jsonl(P1_EFFECTS_ROWS)),
        }
        self.effects = load_jsonl(P1_EFFECTS_ROWS) if resume else []

    @staticmethod
    def _load_done_keys() -> set[tuple[str, str]]:
        done: set[tuple[str, str]] = set()
        for row in load_jsonl(P1_GENERATIONS):
            tid = str(row.get("trajectory_id") or "")
            cid = str(row.get("condition_id") or "")
            if not tid or not cid:
                continue
            if row.get("answer") is not None and row.get("harness_valid") is not False:
                done.add((tid, cid))
            elif row.get("status") == "not_run":
                done.add((tid, cid))
        for row in load_jsonl(P1_INVALID):
            tid = str(row.get("trajectory_id") or row.get("root_id") or "")
            cid = str(row.get("condition_id") or "")
            if tid and cid:
                done.add((tid, cid))
        return done

    def is_done(self, trajectory_id: str, condition_id: str) -> bool:
        return _run_key(trajectory_id, condition_id) in self.done_keys

    def mark_done(self, trajectory_id: str, condition_id: str) -> None:
        self.done_keys.add(_run_key(trajectory_id, condition_id))

    def append_generation(self, row: dict[str, Any]) -> None:
        append_jsonl(P1_GENERATIONS, row)
        self.counts["generations"] += 1
        tid = str(row.get("trajectory_id") or "")
        cid = str(row.get("condition_id") or "")
        if tid and cid:
            self.mark_done(tid, cid)

    def append_counterfactual(self, row: dict[str, Any]) -> None:
        append_jsonl(P1_COUNTERFACTUALS, row)
        self.counts["counterfactuals"] += 1

    def append_invalid(self, row: dict[str, Any]) -> None:
        append_jsonl(P1_INVALID, row)
        self.counts["invalids"] += 1
        tid = str(row.get("trajectory_id") or row.get("root_id") or "")
        cid = str(row.get("condition_id") or "")
        if tid and cid:
            self.mark_done(tid, cid)

    def append_effect(self, row: dict[str, Any]) -> None:
        append_jsonl(P1_EFFECTS_ROWS, row)
        self.effects.append(row)
        self.counts["effects"] += 1
        self.flush_summary()

    def flush_summary(self, *, vllm_ok: bool = True, offline: bool = False) -> None:
        stats = aggregate_following_stats(self.effects)
        write_json(
            P1_DIR / "source_effects.json",
            {
                "n_generations": self.counts["generations"],
                "n_invalid": self.counts["invalids"],
                "n_counterfactuals": self.counts["counterfactuals"],
                "n_effects": self.counts["effects"],
                "vllm_available": vllm_ok,
                "offline_mode": offline,
                "source_following_rate": stats["source_following_rate"],
                "n_scorable": stats["n_scorable"],
                "n_harness_invalid": stats["n_harness_invalid"],
                "n_follow": stats["n_follow"],
                "by_cohort": stats["by_cohort"],
                "checkpoint": True,
            },
        )


def run_p1_dev(
    *,
    dry_run: bool = False,
    limit_per_cohort: int | None = None,
    offline: bool = False,
    resume: bool = True,
    fresh: bool = False,
    cohorts: list[str] | None = None,
    refresh_conditions: list[str] | None = None,
) -> dict[str, Any]:
    rows = {r["trajectory_id"]: r for r in load_jsonl(NORMALIZED_SHOPPING)}
    cohort_manifest = load_json(COHORT_MANIFEST_JSON)
    writer = P1IncrementalWriter(resume=resume, fresh=fresh)
    cohorts = cohorts or ["CEM", "CAP"]

    if refresh_conditions and resume and not fresh:
        refresh_set = set(refresh_conditions)
        writer.done_keys = {
            k for k in writer.done_keys if k[1] not in refresh_set
        }
        writer.effects = [
            e
            for e in writer.effects
            if not (e.get("cohort") in cohorts and e.get("condition_id") in refresh_set)
        ]
        write_jsonl(P1_EFFECTS_ROWS, writer.effects)
        for path in (P1_GENERATIONS, P1_COUNTERFACTUALS, P1_INVALID):
            kept = [
                r
                for r in load_jsonl(path)
                if not (
                    r.get("cohort") in cohorts
                    and r.get("condition_id") in refresh_set
                )
            ]
            write_jsonl(path, kept)

    vllm_ok = check_vllm_available()
    if not dry_run and not offline and not vllm_ok:
        offline = True

    if resume and writer.done_keys:
        print(f"[P1] resume: skipping {len(writer.done_keys)} completed conditions", flush=True)

    for cohort_name, bundle_fn in (("CEM", build_cem_bundle), ("CAP", build_cap_bundle)):
        if cohort_name not in cohorts:
            continue
        tids = cohort_manifest.get("cohorts", {}).get(cohort_name, [])
        if limit_per_cohort:
            tids = tids[:limit_per_cohort]
        print(f"[P1] cohort={cohort_name} n={len(tids)} vllm={vllm_ok} offline={offline}", flush=True)
        for ti, tid in enumerate(tids, 1):
            traj = rows.get(tid)
            if not traj:
                continue
            for bundle_name, op_fn, kwargs in bundle_fn(traj):
                if bundle_name == "original":
                    if writer.is_done(tid, "original"):
                        continue
                    if dry_run or offline:
                        writer.append_generation(
                            {"trajectory_id": tid, "condition_id": "original", "status": "not_run"}
                        )
                        continue
                    print(f"[P1] {cohort_name} {ti}/{len(tids)} {tid} original", flush=True)
                    text, meta = vllm_generate(traj["messages_final_call"], final_synthesis=True)
                    writer.append_generation(
                        _generation_row(
                            trajectory_id=tid,
                            cohort=cohort_name,
                            condition_id="original",
                            text=text,
                            meta=meta,
                        )
                    )
                    continue
                if op_fn is None:
                    continue
                op_result = op_fn(traj["messages_final_call"], **kwargs)
                cid = op_result.condition_id
                if writer.is_done(tid, cid):
                    continue
                base_hash = traj.get("input_hash") or ""
                if op_result.invalid_reason or op_result.semantic_validation == "invalid":
                    cf = op_result.to_counterfactual_record(
                        root_id=traj["root_id"],
                        pair_id=f"{traj['root_id']}:{cid}",
                        estimand="fixed_history_final_synthesis",
                        split=traj.get("split", "unassigned"),
                        base_hash=base_hash,
                    )
                    writer.append_counterfactual(cf)
                    writer.append_invalid({**cf, "trajectory_id": tid})
                    continue
                if dry_run or offline:
                    writer.append_generation(
                        {
                            "trajectory_id": tid,
                            "cohort": cohort_name,
                            "condition_id": cid,
                            "status": "not_run",
                            "reason": "offline_operator_validated" if offline else "dry_run",
                        }
                    )
                    continue
                print(f"[P1] {cohort_name} {ti}/{len(tids)} {tid} {cid}", flush=True)
                out = run_fixed_history(traj, op_result)
                cf = out["counterfactual"]
                writer.append_counterfactual(cf)
                if out["status"] == "invalid":
                    writer.append_invalid({**cf, "trajectory_id": tid})
                    continue
                gen = out["generation"] or {}
                writer.append_generation(
                    _generation_row(
                        trajectory_id=tid,
                        cohort=cohort_name,
                        condition_id=cid,
                        text=str(gen.get("text") or ""),
                        meta=dict(gen.get("latency") or {}),
                        original_answer=str(gen.get("original_answer") or ""),
                    )
                )
                score = evaluate_source_following(
                    condition_id=cid,
                    original_answer=str(gen.get("original_answer") or ""),
                    new_answer=str(gen.get("text") or ""),
                    expected=op_result.expected_high_level_change,
                    edit_manifest=op_result.edit_manifest,
                )
                writer.append_effect(
                    {
                        "trajectory_id": tid,
                        "cohort": cohort_name,
                        "condition_id": cid,
                        **score,
                    }
                )

    if not P1_SEQUENCE_SCORES.is_file():
        write_jsonl(P1_SEQUENCE_SCORES, [])
    writer.flush_summary(vllm_ok=vllm_ok, offline=offline)
    summary = load_json(P1_DIR / "source_effects.json")
    summary["resume"] = resume
    summary["fresh"] = fresh
    write_json(P1_DIR / "source_effects.json", summary)
    return summary


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="Limit per cohort; default all dev cohort")
    ap.add_argument("--offline", action="store_true", help="Validate operators without vLLM generation")
    ap.add_argument("--resume", action="store_true", default=True, help="Skip completed conditions (default on)")
    ap.add_argument("--no-resume", action="store_false", dest="resume")
    ap.add_argument("--fresh", action="store_true", help="Clear P1 jsonl artifacts before run")
    ap.add_argument("--cohort", action="append", choices=["CEM", "CAP"], help="Restrict to cohort(s)")
    ap.add_argument(
        "--refresh-conditions",
        default="",
        help="Comma-separated condition_ids to re-run even if checkpointed",
    )
    args = ap.parse_args()
    refresh = [c.strip() for c in args.refresh_conditions.split(",") if c.strip()] or None
    print(
        json.dumps(
            run_p1_dev(
                dry_run=args.dry_run,
                limit_per_cohort=args.limit,
                offline=args.offline,
                resume=args.resume,
                fresh=args.fresh,
                cohorts=args.cohort or None,
                refresh_conditions=refresh,
            ),
            indent=2,
        )
    )
