"""Export CEM cross-pair audit JSONL for expert review before mechanism GPU runs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ccer.adjudication.loader import load_adjudication_index, resolve_cem_swap_target
from ccer.counterfactual.base import find_product_record_snippets
from ccer.mechanism.activation_store import activation_path, get_vector, load_activation_npz
from ccer.mechanism.pair_select import cem_valid_pair_ids, load_trajectory_index, messages_for_condition
from ccer.replay.evidence_mask import classify_anchor_title_quality, mask_cem_evidence_spans


def export_cem_cross_pair_audit(
    *,
    track: str = "CEM",
    layer: int = 32,
    position: str = "commitment",
    out_path: str | Path | None = None,
    pool: str = "dev",
) -> dict[str, Any]:
    if pool == "mechanism_research":
        from ccer.mechanism.mechanism_pool import build_cem_mechanism_cross_pairs

        sel = build_cem_mechanism_cross_pairs(pool="mechanism_research")
    else:
        sel = cem_valid_pair_ids(include_matched_clean=True)
    cross_pairs = sel["pm_clean_cross_pairs"]
    if out_path is None:
        out_path = (
            Path("${PHANTOM_MERGE_ROOT}/results/reports/manual_verification")
            / ("round12_cem_cross_pair_audit.jsonl" if pool == "mechanism_research" else "round10_cem_cross_pair_audit.jsonl")
        )
    rows_index = load_trajectory_index()
    adj = load_adjudication_index()
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    n_eligible = 0
    n_shell = 0
    n_real = 0
    n_leak_free = 0
    manual_review_indices = {2, 3, 4, 7}

    for i, cp in enumerate(cross_pairs):
        pair_index = i + 1
        pm_tid = str(cp["pm_trajectory_id"])
        clean_tid = str(cp["clean_trajectory_id"])
        pm_traj = rows_index.get(pm_tid)
        target = resolve_cem_swap_target(pm_traj) if pm_traj else None
        anchor_pid = str(
            (getattr(target, "committed_anchor_pid", None) or getattr(target, "textual_selected_pid", None) or "")
            if target
            else ""
        )

        anchor_q = classify_anchor_title_quality(pm_traj or {}, anchor_pid) if pm_traj and anchor_pid else {}
        if anchor_q.get("anchor_title_quality") == "shell_pid_only":
            n_shell += 1
        elif anchor_q.get("anchor_title_quality") == "real_text":
            n_real += 1

        donor_snip = anchor_snip = ""
        if pm_traj and target:
            user = ""
            for m in pm_traj.get("messages_final_call") or []:
                if m.get("role") == "user":
                    user = str(m.get("content") or "")
            if target.donor_pid:
                ds = find_product_record_snippets(user, str(target.donor_pid))
                donor_snip = (ds[0][:240] + "…") if ds and len(ds[0]) > 240 else (ds[0] if ds else "")
            if anchor_pid:
                as_ = find_product_record_snippets(user, anchor_pid)
                anchor_snip = (as_[0][:240] + "…") if as_ and len(as_[0]) > 240 else (as_[0] if as_ else "")

        mask_meta: dict[str, Any] = {}
        masked_user_excerpt = ""
        if pm_traj:
            msgs = messages_for_condition(pm_traj, "original", track=track)
            if msgs:
                masked_msgs, mask_meta = mask_cem_evidence_spans(msgs, pm_traj)
                masked_user = next(m["content"] for m in masked_msgs if m.get("role") == "user")
                masked_user_excerpt = masked_user[:1200] + ("…" if len(masked_user) > 1200 else "")

        pm_act = activation_path(pm_tid, "original")
        clean_act = activation_path(clean_tid, "original")
        pm_npz = load_activation_npz(pm_act) if pm_traj and pm_act.is_file() else None
        clean_npz = load_activation_npz(clean_act) if pm_traj and clean_act.is_file() else None
        pm_vec = get_vector(pm_npz, position=position, layer=layer) if pm_npz else None
        clean_vec = get_vector(clean_npz, position=position, layer=layer) if clean_npz else None
        eligible = pm_vec is not None and clean_vec is not None and bool(mask_meta.get("masked"))
        if eligible:
            n_eligible += 1
        if mask_meta.get("leak_free"):
            n_leak_free += 1

        adj_row = adj.pick_primary_cem_instance(pm_tid) if pm_traj else None
        row = {
            "pair_index": pair_index,
            "pm_trajectory_id": pm_tid,
            "clean_trajectory_id": clean_tid,
            "pairing_method": cp.get("pairing_method"),
            "slot_norm": cp.get("slot_norm"),
            "donor_pid": getattr(target, "donor_pid", None) if target else None,
            "anchor_pid": anchor_pid or None,
            "claim_value": getattr(target, "claim_value", None) if target else None,
            "owner_uniqueness": getattr(target, "owner_uniqueness", None) if target else None,
            "instance_audit_key": getattr(target, "instance_audit_key", None) if target else None,
            "adjudication_status": (adj_row or {}).get("adjudication_status") if adj_row else None,
            "anchor_title_quality": anchor_q.get("anchor_title_quality"),
            "anchor_title_preview": anchor_q.get("anchor_title_preview"),
            "has_pins_not_viewed": anchor_q.get("has_pins_not_viewed", False),
            "donor_snippet_preview": donor_snip,
            "anchor_snippet_preview": anchor_snip,
            "strong_mask": mask_meta,
            "masked_user_excerpt": masked_user_excerpt if pair_index in manual_review_indices else None,
            "requires_manual_leak_review": pair_index in manual_review_indices,
            "interchange_eligible": eligible,
            "pm_has_activation": pm_vec is not None,
            "clean_has_activation": clean_vec is not None,
            "expert_review_prompt": (
                "Confirm donor=rival PID, anchor=committed selection PID, clean donor appropriate; "
                "note anchor_title_quality shell vs real."
            ),
        }
        rows.append(row)

    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    shell_indices = [r["pair_index"] for r in rows if r.get("anchor_title_quality") == "shell_pid_only"]
    real_indices = [r["pair_index"] for r in rows if r.get("anchor_title_quality") == "real_text"]

    return {
        "schema_version": "ccer_cem_cross_pair_audit_v2",
        "track": track.upper(),
        "n_pairs": len(rows),
        "n_interchange_eligible": n_eligible,
        "n_strong_mask_leak_free": n_leak_free,
        "anchor_title_stratification": {
            "shell_pid_only": {"n": n_shell, "pair_indices": shell_indices},
            "real_text": {"n": n_real, "pair_indices": real_indices},
        },
        "manual_leak_review_pairs": sorted(manual_review_indices),
        "out_path": str(path),
        "pairing_note": "Native n=22 cem_valid_pair_ids; no multi_clean_expansion.",
    }
