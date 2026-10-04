"""Line A / Track K claim-level mitigation audit (fixed-anchor post-hoc filtering)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from ccer.mechanism.supervised_probe import PM_VERDICTS


def is_pm_claim(row: dict) -> bool:
    if row.get("_rewritten") or row.get("_extraction_hit"):
        return False
    if int(row.get("y_pm", row.get("y", 0)) or 0) == 1:
        return True
    verdict = str(row.get("gold_verdict", ""))
    return verdict in PM_VERDICTS


def is_correct_binding_claim(row: dict) -> bool:
    if row.get("_rewritten") or row.get("_extraction_hit"):
        return False
    if is_pm_claim(row):
        return False
    return int(row.get("y_pm", row.get("y", 0)) or 0) == 0


def trajectory_has_pm_rows(claim_rows: list[dict]) -> bool:
    return any(is_pm_claim(r) for r in claim_rows)


def _records(grp: pd.DataFrame) -> list[dict]:
    return [dict(rec) for rec in grp.to_dict("records")]


@dataclass
class ClaimAuditResult:
    n_trajectories: int
    baseline_pm_count: int
    baseline_pm_rate: float
    gated_pm_count: int
    gated_pm_rate: float
    claims_total_baseline: int
    claims_total_gated: int
    claims_retained_mean: float
    pm_claims_total_baseline: int
    pm_claims_total_gated: int
    cb_claims_total_baseline: int
    cb_claims_retained: int
    cb_claims_removed: int
    cb_retention_rate: float
    removed_cb_fraction_of_all_removed: float
    pm_claims_removed: int
    non_cb_non_pm_removed: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_trajectories": self.n_trajectories,
            "baseline_pm_count": self.baseline_pm_count,
            "baseline_pm_rate": self.baseline_pm_rate,
            "gated_pm_count": self.gated_pm_count,
            "gated_pm_rate": self.gated_pm_rate,
            "claims_total_baseline": self.claims_total_baseline,
            "claims_total_gated": self.claims_total_gated,
            "claims_retained_mean": self.claims_retained_mean,
            "pm_claims_total_baseline": self.pm_claims_total_baseline,
            "pm_claims_total_gated": self.pm_claims_total_gated,
            "cb_claims_total_baseline": self.cb_claims_total_baseline,
            "cb_claims_retained": self.cb_claims_retained,
            "cb_claims_removed": self.cb_claims_removed,
            "cb_retention_rate": self.cb_retention_rate,
            "removed_cb_fraction_of_all_removed": self.removed_cb_fraction_of_all_removed,
            "pm_claims_removed": self.pm_claims_removed,
            "non_cb_non_pm_removed": self.non_cb_non_pm_removed,
            "pm_reduction": self.baseline_pm_rate - self.gated_pm_rate,
        }


def audit_gating(
    df: pd.DataFrame,
    drop_fn: Callable[[dict], bool],
) -> ClaimAuditResult:
    """Evaluate trajectory PM + claim retention under per-claim drop_fn(row)->drop."""
    baseline_pm: list[int] = []
    gated_pm: list[int] = []
    claims_kept: list[int] = []
    claims_base = claims_gated = 0
    pm_base = pm_gated = 0
    cb_base = cb_ret = cb_rem = 0
    pm_removed = non_cb_removed = 0

    for _, grp in df.groupby("group_id", sort=False):
        rows = _records(grp)
        baseline_pm.append(int(trajectory_has_pm_rows(rows)))
        claims_base += len(rows)
        pm_base += sum(1 for r in rows if is_pm_claim(r))
        cb_base += sum(1 for r in rows if is_correct_binding_claim(r))

        kept: list[dict] = []
        for r in rows:
            if drop_fn(r):
                if is_pm_claim(r):
                    pm_removed += 1
                elif is_correct_binding_claim(r):
                    cb_rem += 1
                else:
                    non_cb_removed += 1
            else:
                kept.append(r)
                if is_correct_binding_claim(r):
                    cb_ret += 1

        claims_kept.append(len(kept))
        claims_gated += len(kept)
        gated_pm.append(int(trajectory_has_pm_rows(kept)))
        pm_gated += sum(1 for r in kept if is_pm_claim(r))

    n = len(baseline_pm) or 1
    removed_total = pm_removed + cb_rem + non_cb_removed
    return ClaimAuditResult(
        n_trajectories=n,
        baseline_pm_count=sum(baseline_pm),
        baseline_pm_rate=sum(baseline_pm) / n,
        gated_pm_count=sum(gated_pm),
        gated_pm_rate=sum(gated_pm) / n,
        claims_total_baseline=claims_base,
        claims_total_gated=claims_gated,
        claims_retained_mean=float(np.mean(claims_kept)) if claims_kept else 0.0,
        pm_claims_total_baseline=pm_base,
        pm_claims_total_gated=pm_gated,
        cb_claims_total_baseline=cb_base,
        cb_claims_retained=cb_ret,
        cb_claims_removed=cb_rem,
        cb_retention_rate=(cb_ret / cb_base) if cb_base else 1.0,
        removed_cb_fraction_of_all_removed=(cb_rem / removed_total) if removed_total else 0.0,
        pm_claims_removed=pm_removed,
        non_cb_non_pm_removed=non_cb_removed,
    )


def _effective_rows_after_rewrite(rows: list[dict], rewrite_fn: Callable[[dict], bool]) -> list[dict]:
    """Rewritten PM claims no longer count as PM; rewritten clean claims no longer count as CB."""
    effective: list[dict] = []
    for r in rows:
        if rewrite_fn(r):
            neutral = dict(r)
            neutral["y_pm"] = 0
            neutral["y"] = 0
            neutral["_rewritten"] = True
            effective.append(neutral)
        else:
            effective.append(r)
    return effective


def audit_rewrite(
    df: pd.DataFrame,
    rewrite_fn: Callable[[dict], bool],
) -> ClaimAuditResult:
    """Rewrite mode: all claims remain in the answer; PM/CB use post-rewrite effective labels."""
    baseline_pm: list[int] = []
    gated_pm: list[int] = []
    claims_kept: list[int] = []
    claims_base = claims_gated = 0
    pm_base = pm_gated = 0
    cb_base = cb_ret = cb_rem = 0
    pm_removed = non_cb_removed = 0

    for _, grp in df.groupby("group_id", sort=False):
        rows = _records(grp)
        baseline_pm.append(int(trajectory_has_pm_rows(rows)))
        claims_base += len(rows)
        pm_base += sum(1 for r in rows if is_pm_claim(r))
        cb_base += sum(1 for r in rows if is_correct_binding_claim(r))

        effective = _effective_rows_after_rewrite(rows, rewrite_fn)
        for r, eff in zip(rows, effective):
            if rewrite_fn(r):
                if is_pm_claim(r):
                    pm_removed += 1
                elif is_correct_binding_claim(r):
                    cb_rem += 1
                else:
                    non_cb_removed += 1
            elif is_correct_binding_claim(eff):
                cb_ret += 1

        claims_kept.append(len(rows))
        claims_gated += len(rows)
        gated_pm.append(int(trajectory_has_pm_rows(effective)))
        pm_gated += sum(1 for r in effective if is_pm_claim(r))

    n = len(baseline_pm) or 1
    removed_total = pm_removed + cb_rem + non_cb_removed
    return ClaimAuditResult(
        n_trajectories=n,
        baseline_pm_count=sum(baseline_pm),
        baseline_pm_rate=sum(baseline_pm) / n,
        gated_pm_count=sum(gated_pm),
        gated_pm_rate=sum(gated_pm) / n,
        claims_total_baseline=claims_base,
        claims_total_gated=claims_gated,
        claims_retained_mean=float(np.mean(claims_kept)) if claims_kept else 0.0,
        pm_claims_total_baseline=pm_base,
        pm_claims_total_gated=pm_gated,
        cb_claims_total_baseline=cb_base,
        cb_claims_retained=cb_ret,
        cb_claims_removed=cb_rem,
        cb_retention_rate=(cb_ret / cb_base) if cb_base else 1.0,
        removed_cb_fraction_of_all_removed=(cb_rem / removed_total) if removed_total else 0.0,
        pm_claims_removed=pm_removed,
        non_cb_non_pm_removed=non_cb_removed,
    )


def kept_counts_per_group(df: pd.DataFrame, drop_fn: Callable[[dict], bool]) -> dict[str, int]:
    out: dict[str, int] = {}
    for gid, grp in df.groupby("group_id", sort=False):
        rows = _records(grp)
        out[str(gid)] = sum(1 for r in rows if not drop_fn(r))
    return out


def random_retention_matched_eval(
    df: pd.DataFrame,
    target_kept: dict[str, int],
    *,
    n_seeds: int = 200,
    base_seed: int = 42,
) -> dict[str, Any]:
    pm_rates: list[float] = []
    pm_counts: list[int] = []
    cb_removed_fracs: list[float] = []

    for seed in range(base_seed, base_seed + n_seeds):
        rng = np.random.default_rng(seed)
        frames: list[pd.DataFrame] = []
        for gid, grp in df.groupby("group_id", sort=False):
            rows = _records(grp)
            k = int(target_kept.get(str(gid), len(rows)))
            n = len(rows)
            if n == 0:
                continue
            drop_mask = np.zeros(n, dtype=bool)
            if k < n:
                drop_idx = rng.choice(n, size=n - k, replace=False)
                drop_mask[drop_idx] = True
            sub = grp.copy()
            sub["_random_drop"] = drop_mask
            frames.append(sub)

        if not frames:
            continue
        sub_df = pd.concat(frames, ignore_index=True)

        def _drop(row: dict) -> bool:
            return bool(row.get("_random_drop", False))

        audit = audit_gating(sub_df, _drop)
        pm_rates.append(audit.gated_pm_rate)
        pm_counts.append(audit.gated_pm_count)
        cb_removed_fracs.append(audit.removed_cb_fraction_of_all_removed)

    arr = np.array(pm_rates) if pm_rates else np.array([0.0])
    return {
        "n_seeds": n_seeds,
        "pm_rate_mean": float(arr.mean()),
        "pm_rate_std": float(arr.std()),
        "pm_rate_median": float(np.median(arr)),
        "pm_rate_p05": float(np.percentile(arr, 5)),
        "pm_rate_p95": float(np.percentile(arr, 95)),
        "pm_count_mean": float(np.mean(pm_counts)) if pm_counts else 0.0,
        "pm_count_std": float(np.std(pm_counts)) if pm_counts else 0.0,
        "removed_cb_fraction_mean": float(np.mean(cb_removed_fracs)) if cb_removed_fracs else 0.0,
    }


def bootstrap_traj_pm(
    df: pd.DataFrame,
    drop_fn: Callable[[dict], bool],
    *,
    B: int = 2000,
    seed: int = 42,
) -> dict[str, float]:
    gids = list(df["group_id"].astype(str).unique())
    by = {gid: g for gid, g in df.groupby(df["group_id"].astype(str), sort=False)}
    flags = np.zeros(len(gids), dtype=np.float64)
    for i, gid in enumerate(gids):
        rows = by[gid].to_dict("records")
        kept = [r for r in rows if not drop_fn(r)]
        flags[i] = float(trajectory_has_pm_rows(kept))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(gids), size=(B, len(gids)))
    rates = flags[idx].mean(axis=1)
    return {
        "boot_mean": float(rates.mean()),
        "boot_std": float(rates.std()),
        "boot_ci95_lo": float(np.quantile(rates, 0.025)),
        "boot_ci95_hi": float(np.quantile(rates, 0.975)),
        "B": B,
    }


def _effective_line_l_row(
    r: dict,
    *,
    conservative: bool = False,
    verified_fn: Callable[[dict], bool] | None = None,
) -> tuple[dict, str]:
    """Return (effective_row, disposition) for one Line L action."""
    action = str(r.get("_line_l_action") or "keep")
    if action == "random_edit":
        return dict(r), "keep"
    if action == "delete":
        return dict(r), "delete"
    eff = dict(r)
    if action == "rewrite":
        clears_pm = True
        if conservative and verified_fn is not None:
            clears_pm = bool(verified_fn(r))
        if clears_pm:
            eff["y_pm"] = 0
            eff["y"] = 0
            eff["_extraction_hit"] = True
        return eff, "rewrite"
    return eff, "keep"


def audit_line_l_conservative(
    df: pd.DataFrame,
    verified_fn: Callable[[dict], bool],
) -> ClaimAuditResult:
    """Conservative audit: only verified_grounded rewrites clear PM labels."""
    baseline_pm: list[int] = []
    gated_pm: list[int] = []
    claims_kept: list[int] = []
    claims_base = claims_gated = 0
    pm_base = pm_gated = 0
    cb_base = cb_ret = cb_rem = 0
    pm_removed = non_cb_removed = 0

    for _, grp in df.groupby("group_id", sort=False):
        rows = _records(grp)
        baseline_pm.append(int(trajectory_has_pm_rows(rows)))
        claims_base += len(rows)
        pm_base += sum(1 for r in rows if is_pm_claim(r))
        cb_base += sum(1 for r in rows if is_correct_binding_claim(r))

        kept: list[dict] = []
        for r in rows:
            action = str(r.get("_line_l_action") or "keep")
            if action == "delete":
                if is_pm_claim(r):
                    pm_removed += 1
                elif is_correct_binding_claim(r):
                    cb_rem += 1
                else:
                    non_cb_removed += 1
                continue
            eff, _ = _effective_line_l_row(r, conservative=True, verified_fn=verified_fn)
            kept.append(eff)
            if action == "rewrite" and verified_fn(r) and is_pm_claim(r):
                pm_removed += 1
            elif action == "rewrite" and verified_fn(r) and is_correct_binding_claim(r):
                cb_rem += 1
            elif is_correct_binding_claim(eff):
                cb_ret += 1

        claims_kept.append(len(kept))
        claims_gated += len(kept)
        gated_pm.append(int(trajectory_has_pm_rows(kept)))
        pm_gated += sum(1 for r in kept if is_pm_claim(r))

    n = len(baseline_pm) or 1
    removed_total = pm_removed + cb_rem + non_cb_removed
    return ClaimAuditResult(
        n_trajectories=n,
        baseline_pm_count=sum(baseline_pm),
        baseline_pm_rate=sum(baseline_pm) / n,
        gated_pm_count=sum(gated_pm),
        gated_pm_rate=sum(gated_pm) / n,
        claims_total_baseline=claims_base,
        claims_total_gated=claims_gated,
        claims_retained_mean=float(np.mean(claims_kept)) if claims_kept else 0.0,
        pm_claims_total_baseline=pm_base,
        pm_claims_total_gated=pm_gated,
        cb_claims_total_baseline=cb_base,
        cb_claims_retained=cb_ret,
        cb_claims_removed=cb_rem,
        cb_retention_rate=(cb_ret / cb_base) if cb_base else 1.0,
        removed_cb_fraction_of_all_removed=(cb_rem / removed_total) if removed_total else 0.0,
        pm_claims_removed=pm_removed,
        non_cb_non_pm_removed=non_cb_removed,
    )


def bootstrap_traj_pm_line_l_conservative(
    df: pd.DataFrame,
    verified_fn: Callable[[dict], bool],
    *,
    B: int = 2000,
    seed: int = 42,
) -> dict[str, float]:
    gids = list(df["group_id"].astype(str).unique())
    by = {gid: g for gid, g in df.groupby(df["group_id"].astype(str), sort=False)}
    flags = np.zeros(len(gids), dtype=np.float64)
    for i, gid in enumerate(gids):
        rows = by[gid].to_dict("records")
        effective = []
        for r in rows:
            action = str(r.get("_line_l_action") or "keep")
            if action == "delete":
                continue
            eff, _ = _effective_line_l_row(r, conservative=True, verified_fn=verified_fn)
            effective.append(eff)
        flags[i] = float(trajectory_has_pm_rows(effective))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(gids), size=(B, len(gids)))
    rates = flags[idx].mean(axis=1)
    return {
        "boot_mean": float(rates.mean()),
        "boot_std": float(rates.std()),
        "boot_ci95_lo": float(np.quantile(rates, 0.025)),
        "boot_ci95_hi": float(np.quantile(rates, 0.975)),
        "B": B,
    }


def audit_line_l(df: pd.DataFrame) -> ClaimAuditResult:
    """Audit Line L interventions using precomputed _line_l_action on each row."""
    baseline_pm: list[int] = []
    gated_pm: list[int] = []
    claims_kept: list[int] = []
    claims_base = claims_gated = 0
    pm_base = pm_gated = 0
    cb_base = cb_ret = cb_rem = 0
    pm_removed = non_cb_removed = 0

    for _, grp in df.groupby("group_id", sort=False):
        rows = _records(grp)
        baseline_pm.append(int(trajectory_has_pm_rows(rows)))
        claims_base += len(rows)
        pm_base += sum(1 for r in rows if is_pm_claim(r))
        cb_base += sum(1 for r in rows if is_correct_binding_claim(r))

        kept: list[dict] = []
        for r in rows:
            action = str(r.get("_line_l_action") or "keep")
            if action == "random_edit":
                kept.append(dict(r))
                if is_correct_binding_claim(r):
                    cb_ret += 1
                continue
            if action == "delete":
                if is_pm_claim(r):
                    pm_removed += 1
                elif is_correct_binding_claim(r):
                    cb_rem += 1
                else:
                    non_cb_removed += 1
                continue
            eff = dict(r)
            if action == "rewrite":
                eff["y_pm"] = 0
                eff["y"] = 0
                eff["_extraction_hit"] = True
                if is_pm_claim(r):
                    pm_removed += 1
                elif is_correct_binding_claim(r):
                    cb_rem += 1
            kept.append(eff)
            if is_correct_binding_claim(eff):
                cb_ret += 1

        claims_kept.append(len(kept))
        claims_gated += len(kept)
        gated_pm.append(int(trajectory_has_pm_rows(kept)))
        pm_gated += sum(1 for r in kept if is_pm_claim(r))

    n = len(baseline_pm) or 1
    removed_total = pm_removed + cb_rem + non_cb_removed
    return ClaimAuditResult(
        n_trajectories=n,
        baseline_pm_count=sum(baseline_pm),
        baseline_pm_rate=sum(baseline_pm) / n,
        gated_pm_count=sum(gated_pm),
        gated_pm_rate=sum(gated_pm) / n,
        claims_total_baseline=claims_base,
        claims_total_gated=claims_gated,
        claims_retained_mean=float(np.mean(claims_kept)) if claims_kept else 0.0,
        pm_claims_total_baseline=pm_base,
        pm_claims_total_gated=pm_gated,
        cb_claims_total_baseline=cb_base,
        cb_claims_retained=cb_ret,
        cb_claims_removed=cb_rem,
        cb_retention_rate=(cb_ret / cb_base) if cb_base else 1.0,
        removed_cb_fraction_of_all_removed=(cb_rem / removed_total) if removed_total else 0.0,
        pm_claims_removed=pm_removed,
        non_cb_non_pm_removed=non_cb_removed,
    )


def compute_extraction_hit_rate(df: pd.DataFrame, tau: float) -> dict[str, Any]:
    flagged = df[df["p_pm"].astype(float) > float(tau)]
    n_flagged = len(flagged)
    if n_flagged == 0:
        return {"n_flagged": 0, "n_hit": 0, "extraction_hit_rate": None}
    n_hit = int((flagged["_line_l_branch"] == "extraction_hit").sum())
    return {
        "n_flagged": n_flagged,
        "n_hit": n_hit,
        "extraction_hit_rate": n_hit / n_flagged,
        "n_miss": int((flagged["_line_l_branch"] == "extraction_miss").sum()),
        "n_no_slot": int((flagged["_line_l_branch"] == "skip_no_slot").sum()),
    }


def bootstrap_traj_pm_line_l(df: pd.DataFrame, *, B: int = 2000, seed: int = 42) -> dict[str, float]:
    gids = list(df["group_id"].astype(str).unique())
    by = {gid: g for gid, g in df.groupby(df["group_id"].astype(str), sort=False)}
    flags = np.zeros(len(gids), dtype=np.float64)
    for i, gid in enumerate(gids):
        rows = by[gid].to_dict("records")
        effective = []
        for r in rows:
            action = str(r.get("_line_l_action") or "keep")
            if action == "delete":
                continue
            eff = dict(r)
            if action == "rewrite":
                eff["y_pm"] = 0
                eff["y"] = 0
                eff["_extraction_hit"] = True
            effective.append(eff)
        flags[i] = float(trajectory_has_pm_rows(effective))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(gids), size=(B, len(gids)))
    rates = flags[idx].mean(axis=1)
    return {
        "boot_mean": float(rates.mean()),
        "boot_std": float(rates.std()),
        "boot_ci95_lo": float(np.quantile(rates, 0.025)),
        "boot_ci95_hi": float(np.quantile(rates, 0.975)),
        "B": B,
    }


def permutation_pm_rate_diff(
    df_target: pd.DataFrame,
    df_wrong: pd.DataFrame,
    *,
    n_perm: int = 2000,
    seed: int = 42,
) -> dict[str, float]:
    """Permutation test on trajectory-level PM rate difference (target - wrong)."""
    gids = sorted(set(df_target["group_id"].astype(str)) & set(df_wrong["group_id"].astype(str)))
    if not gids:
        return {"observed_diff": 0.0, "p_value_two_sided": 1.0, "n_trajectories": 0}

    def _pm_flags(df: pd.DataFrame) -> dict[str, float]:
        out: dict[str, float] = {}
        for gid in gids:
            sub = df[df["group_id"].astype(str) == gid]
            rows = sub.to_dict("records")
            effective = []
            for r in rows:
                action = str(r.get("_line_l_action") or "keep")
                if action == "delete":
                    continue
                eff = dict(r)
                if action == "rewrite":
                    eff["y_pm"] = 0
                    eff["y"] = 0
                    eff["_extraction_hit"] = True
                effective.append(eff)
            out[gid] = float(trajectory_has_pm_rows(effective))
        return out

    t_flags = _pm_flags(df_target)
    w_flags = _pm_flags(df_wrong)
    t_arr = np.array([t_flags[g] for g in gids])
    w_arr = np.array([w_flags[g] for g in gids])
    observed = float(t_arr.mean() - w_arr.mean())

    rng = np.random.default_rng(seed)
    diffs: list[float] = []
    for _ in range(n_perm):
        swap = rng.random(len(gids)) < 0.5
        t_p = np.where(swap, w_arr, t_arr)
        w_p = np.where(swap, t_arr, w_arr)
        diffs.append(float(t_p.mean() - w_p.mean()))
    diffs_arr = np.array(diffs)
    p = float((np.abs(diffs_arr) >= abs(observed)).mean())
    return {
        "observed_diff": observed,
        "target_pm_rate": float(t_arr.mean()),
        "wrong_pm_rate": float(w_arr.mean()),
        "p_value_two_sided": p,
        "n_trajectories": len(gids),
        "n_perm": n_perm,
    }


def instances_to_dataframe(
    instances: list[dict[str, Any]],
    scores: dict[str, float],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for inst in instances:
        iak = str(inst["instance_audit_key"])
        rows.append(
            {
                "group_id": str(inst["trajectory_id"]),
                "claim_id": iak,
                "instance_audit_key": iak,
                "trajectory_id": str(inst["trajectory_id"]),
                "split": str(inst["split"]),
                "y_pm": int(inst["y"]),
                "y": int(inst["y"]),
                "gold_verdict": str(inst.get("gold_verdict", "")),
                "response_quote": str(inst.get("response_quote", "")),
                "p_pm": float(scores.get(iak, 0.0)),
            }
        )
    return pd.DataFrame(rows)
