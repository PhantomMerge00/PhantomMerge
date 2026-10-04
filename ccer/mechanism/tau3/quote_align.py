"""Build tau3 adjudication rows with quote span alignment."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ccer.io_utils import load_jsonl, write_json, write_jsonl
from ccer.mechanism.supervised_probe import CLEAN_VERDICT, PM_VERDICTS
from ccer.mechanism.tau3.domain import Tau3DomainConfig, get_tau3_domain
from ccer.mechanism.tau3.replay import (
    find_quote_span,
    infer_slot_norm_from_quote,
    structure_matched_clean_quote,
    tau2_to_chat_messages,
    tokenize_tau2_trajectory,
)
from ccer.normalize.telecom import _norm_slot
from ccer.replay.hf_forward import load_hf_tokenizer


def _pm_instance_row(
    inst: dict[str, Any],
    traj_row: dict[str, Any],
    cfg: Tau3DomainConfig,
    *,
    quote_start: int,
    quote_end: int,
) -> dict[str, Any]:
    legacy = str(inst.get("gold_verdict") or "")
    return {
        "schema": cfg.adjudication_schema,
        "instance_audit_key": str(inst["instance_audit_key"]),
        "trajectory_id": str(traj_row["trajectory_id"]),
        "trajectory_audit_key": traj_row.get("trajectory_audit_key"),
        "pack_source": traj_row.get("pack_source"),
        "response_quote": str(inst.get("response_quote") or ""),
        "quote_start": int(quote_start),
        "quote_end": int(quote_end),
        "slot": inst.get("slot") or inst.get("slot_norm"),
        "slot_norm": _norm_slot(str(inst.get("slot_norm") or inst.get("slot") or "")),
        "value": inst.get("value"),
        "claim_scope": inst.get("claim_scope") or "anchor",
        "gold_verdict": legacy,
        "y_pm": 1 if legacy in PM_VERDICTS else 0,
        "commitment_eligible": True,
        "pm_binary": inst.get("pm_binary"),
    }


def _clean_instance_row(
    traj_row: dict[str, Any],
    cfg: Tau3DomainConfig,
    *,
    response_quote: str,
    quote_start: int,
    quote_end: int,
    slot_norm: str = "",
) -> dict[str, Any]:
    tid = str(traj_row["trajectory_id"])
    slot_norm = _norm_slot(slot_norm or infer_slot_norm_from_quote(response_quote, domain=cfg.domain))
    return {
        "schema": cfg.adjudication_schema,
        "instance_audit_key": f"{cfg.clean_iak_prefix}:{tid}",
        "trajectory_id": tid,
        "trajectory_audit_key": traj_row.get("trajectory_audit_key"),
        "pack_source": traj_row.get("pack_source"),
        "response_quote": response_quote,
        "quote_start": int(quote_start),
        "quote_end": int(quote_end),
        "slot": slot_norm,
        "slot_norm": slot_norm,
        "value": "",
        "claim_scope": "anchor",
        "gold_verdict": CLEAN_VERDICT,
        "y_pm": 0,
        "commitment_eligible": True,
        "pm_binary": "clean",
    }


def build_adjudication(
    cfg: Tau3DomainConfig | None = None,
    *,
    normalized_path: Path | None = None,
    out_path: Path | None = None,
) -> dict[str, Any]:
    cfg = cfg or get_tau3_domain("telecom")
    normalized_path = normalized_path or cfg.normalized
    out_path = out_path or cfg.adjudication

    tokenizer = load_hf_tokenizer()
    adjudication_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    stats = {
        "domain": cfg.domain,
        "n_pm_instances": 0,
        "n_pm_aligned": 0,
        "n_clean_trajectories": 0,
        "n_clean_aligned": 0,
        "n_skipped_no_replay": 0,
    }

    for traj_row in load_jsonl(normalized_path):
        tid = str(traj_row["trajectory_id"])
        if not (traj_row.get("eligible") or {}).get("native_replay", True):
            stats["n_skipped_no_replay"] += 1
            continue

        messages = tau2_to_chat_messages(traj_row.get("messages") or [])
        tok_pack = tokenize_tau2_trajectory(messages, tokenizer)
        serialized = tok_pack["serialized_text"]
        final_answer = str((traj_row.get("metadata") or {}).get("final_answer") or "")

        for inst in traj_row.get("claims") or []:
            stats["n_pm_instances"] += 1
            quote = str(inst.get("response_span") or "")
            span = find_quote_span(serialized, quote)
            if span is None:
                failures.append(
                    {
                        "trajectory_id": tid,
                        "instance_audit_key": inst.get("instance_audit_key"),
                        "response_quote": quote,
                        "reason": "quote_not_in_serialized_text",
                    }
                )
                continue
            qs, qe = span
            stats["n_pm_aligned"] += 1
            adjudication_rows.append(
                _pm_instance_row(
                    {
                        **inst,
                        "response_quote": quote,
                        "gold_verdict": inst.get("legacy_label"),
                    },
                    traj_row,
                    cfg,
                    quote_start=qs,
                    quote_end=qe,
                )
            )

        if (traj_row.get("trajectory_outcome") or "") == "clean":
            stats["n_clean_trajectories"] += 1
            parsed = structure_matched_clean_quote(
                traj_row.get("messages") or [],
                serialized,
                final_answer=final_answer,
            )
            if parsed is None:
                failures.append({"trajectory_id": tid, "reason": "no_structure_matched_clean_quote"})
                continue
            quote, qs, qe = parsed
            stats["n_clean_aligned"] += 1
            adjudication_rows.append(
                _clean_instance_row(
                    traj_row,
                    cfg,
                    response_quote=quote,
                    quote_start=qs,
                    quote_end=qe,
                )
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_path, adjudication_rows)
    if failures:
        write_jsonl(cfg.quote_align_failures, failures)
    total = stats["n_pm_instances"] + stats["n_clean_trajectories"]
    aligned = stats["n_pm_aligned"] + stats["n_clean_aligned"]
    summary = {
        **stats,
        "n_adjudication_rows": len(adjudication_rows),
        "align_rate": aligned / total if total else 0.0,
        "out_path": str(out_path),
        "n_failures": len(failures),
    }
    write_json(cfg.quote_align_stats, summary)
    return summary


def build_telecom_adjudication(**kwargs: Any) -> dict[str, Any]:
    return build_adjudication(get_tau3_domain("telecom"), **kwargs)


def build_airline_adjudication(**kwargs: Any) -> dict[str, Any]:
    return build_adjudication(get_tau3_domain("airline"), **kwargs)


if __name__ == "__main__":
    import sys

    domain = sys.argv[1] if len(sys.argv) > 1 else "telecom"
    print(json.dumps(build_adjudication(get_tau3_domain(domain)), indent=2))
