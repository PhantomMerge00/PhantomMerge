"""Obs_τ evidence helpers for P2 vendor PM detection (fair anchor, multi-product passages)."""

from __future__ import annotations

import json
from typing import Any

from ccer.io_utils import load_jsonl
from ccer.mechanism.line_l_rewrite_quality import build_evidence_corpus
from ccer.mechanism.modernbert_pm_detector import (
    _load_rollout_obs,
    _resolve_anchor_pid,
    build_detection_text_rows,
)
from ccer.paths import NORMALIZED_SHOPPING


def shop_observed_from_rollout(obs_list: list[dict], anchor_pid: str) -> list[dict]:
    observed: list[dict] = []
    for obs in obs_list:
        results = obs.get("results") or {}
        if isinstance(results, str):
            try:
                results = json.loads(results)
            except json.JSONDecodeError:
                results = {}
        if not isinstance(results, dict):
            continue
        for pr in results.get("products") or []:
            if not isinstance(pr, dict):
                continue
            pid = str(pr.get("product_id") or "")
            if not pid:
                continue
            observed.append(
                {
                    "product_id": pid,
                    "is_selected_anchor": pid == str(anchor_pid),
                    "evidence_excerpt": json.dumps(pr, ensure_ascii=False)[:2000],
                }
            )
    return observed


def fair_obs_tau_corpus(traj: dict[str, Any], anchor_pid: str) -> str:
    if anchor_pid:
        return build_evidence_corpus(traj, anchor_pid)
    return ""


def multi_sample_passages(
    traj: dict[str, Any],
    anchor_pid: str,
    rollout_obs: list[dict] | None,
    *,
    k: int = 5,
) -> list[str]:
    """K passages for SelfCheck-style multi-sample NLI (rollout product JSON + anchor corpus)."""
    observed = shop_observed_from_rollout(rollout_obs or [], anchor_pid)
    anchor_first = sorted(observed, key=lambda o: (not o.get("is_selected_anchor"), o.get("product_id") or ""))
    passages = [str(o.get("evidence_excerpt") or "") for o in anchor_first if o.get("evidence_excerpt")]
    corpus = fair_obs_tau_corpus(traj, anchor_pid)
    if corpus and (not passages or passages[0] != corpus):
        if passages and anchor_pid:
            passages[0] = corpus
        elif corpus:
            passages.insert(0, corpus)
    if not passages and corpus:
        passages = [corpus]
    if len(passages) < k and corpus:
        chunk = 1500
        for i in range(0, len(corpus), chunk):
            extra = corpus[i : i + chunk]
            if extra not in passages:
                passages.append(extra)
            if len(passages) >= k:
                break
    return passages[:k] if passages else [""]


def iter_detection_rows_with_traj():
    norm = {r["trajectory_id"]: r for r in load_jsonl(NORMALIZED_SHOPPING)}
    rollout_by_tid = _load_rollout_obs()
    for row in build_detection_text_rows(require_activation=True):
        tid = str(row["trajectory_id"])
        traj = norm.get(tid) or {}
        anchor_pid = _resolve_anchor_pid(
            {"trajectory_id": tid, "final_answer": (traj.get("metadata") or {}).get("final_answer", "")},
            traj,
            None,
            use_oracle_pid=False,
        )
        yield row, traj, anchor_pid, rollout_by_tid.get(tid) or []
