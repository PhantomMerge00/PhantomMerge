#!/usr/bin/env python3
"""Diagnose dynamic_answer_anchor patch sites (Task J)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path("${PHANTOM_MERGE_ROOT}")
sys.path.insert(0, str(ROOT))

from ccer.mechanism.activation_store import activation_path, load_activation_npz
from ccer.mechanism.line_b_protocol import filter_cross_pairs_by_quality
from ccer.mechanism.mechanism_pool import build_cem_mechanism_cross_pairs
from ccer.mechanism.pair_select import load_trajectory_index, messages_for_condition
from ccer.replay.hf_forward import load_hf_model, tokenize_ccer_messages
from ccer.replay.live_position import (
    cross_pair_passes_line_d_v3_gate,
    dynamic_answer_anchor_patch_positions,
    resolve_answer_region_token_idx,
)


def _diagnose_offsets(tokenizer, seq: list[int]) -> dict:
    text = tokenizer.decode(seq, skip_special_tokens=False)
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    rt_len = len(enc["offset_mapping"])
    return {
        "seq_len": len(seq),
        "re_tokenized_len": rt_len,
        "len_match": rt_len == len(seq),
        "len_delta": rt_len - len(seq),
    }


def _would_patch_at_step(
    *,
    partial: list[int],
    prompt_len: int,
    anchor_pos: int,
    tokenizer,
    position: str,
) -> bool:
    live = resolve_answer_region_token_idx(
        tokenizer=tokenizer,
        seq=partial,
        prompt_len=prompt_len,
        position=position,
    )
    sites = dynamic_answer_anchor_patch_positions(
        seq_len=len(partial),
        prompt_len=prompt_len,
        decode_step=len(partial) > prompt_len,
        anchor_pos=anchor_pos,
        live_idx=live,
    )
    return bool(sites)


def _scan_decode_steps(
    *,
    tokenizer,
    messages,
    prompt_len: int,
    position: str,
    anchor_pos: int,
    final_answer: str,
    max_new_tokens: int = 384,
) -> dict:
    tok = tokenize_ccer_messages(messages, tokenizer, output_text=None)
    seq = list(tok["full_ids"][:prompt_len])
    full_tf = tokenize_ccer_messages(messages, tokenizer, output_text=final_answer)
    gen_suffix = list(full_tf["full_ids"][prompt_len:])
    steps_to_sim = min(len(gen_suffix), max_new_tokens)

    hits: list[dict] = []
    first_live_step: int | None = None
    first_patch_step: int | None = None

    for step in range(steps_to_sim + 1):
        partial = seq + gen_suffix[:step]
        live = resolve_answer_region_token_idx(
            tokenizer=tokenizer,
            seq=partial,
            prompt_len=prompt_len,
            position=position,
        )
        decode_step = step > 0
        sites = dynamic_answer_anchor_patch_positions(
            seq_len=len(partial),
            prompt_len=prompt_len,
            decode_step=decode_step,
            anchor_pos=anchor_pos,
            live_idx=live,
        )
        if live >= 0 and first_live_step is None:
            first_live_step = step
        if sites and first_patch_step is None:
            first_patch_step = step
        if step in (0, 1, 5, 50, 100, steps_to_sim) or sites:
            decode_text = tokenizer.decode(partial, skip_special_tokens=False)
            snippet_start = max(0, len(decode_text) - 120)
            hits.append(
                {
                    "step": step,
                    "seq_len": len(partial),
                    "live_idx": live,
                    "patch_sites": sites,
                    "decode_snippet": decode_text[snippet_start:],
                    "offsets": _diagnose_offsets(tokenizer, partial),
                }
            )

    from ccer.replay.answer_utils import extract_answer, normalize_final_synthesis_text
    from ccer.replay.position_registry import resolve_symmetric_claim_pair

    final_text = tokenizer.decode(seq + gen_suffix[:steps_to_sim], skip_special_tokens=False)
    body = extract_answer(normalize_final_synthesis_text(final_text))
    claim = resolve_symmetric_claim_pair(body) if body else None
    return {
        "prompt_len": prompt_len,
        "anchor_pos": anchor_pos,
        "simulated_decode_steps": steps_to_sim,
        "first_live_hit_step": first_live_step,
        "first_patch_step": first_patch_step,
        "final_has_answer_body": bool(body),
        "final_has_symmetric_claim": claim is not None,
        "claim_source": claim[2] if claim else None,
        "hit_trace": hits[-12:],
    }


def main() -> None:
    position = "claim_onset"
    pool = "mechanism_research"
    cross = build_cem_mechanism_cross_pairs(pool=pool)["pm_clean_cross_pairs"]
    cross_pairs, _ = filter_cross_pairs_by_quality(cross, min_quality=0.35)
    gated = []
    for cp in cross_pairs:
        pm_tid = str(cp["pm_trajectory_id"])
        clean_tid = str(cp["clean_trajectory_id"])
        pm_path = activation_path(pm_tid, "original")
        clean_path = activation_path(clean_tid, "original")
        if not pm_path.is_file() or not clean_path.is_file():
            continue
        pm_npz = load_activation_npz(pm_path)
        clean_npz = load_activation_npz(clean_path)
        if cross_pair_passes_line_d_v3_gate(pm_npz, clean_npz, position):
            gated.append(cp)
        if len(gated) >= 8:
            break
    cross_pairs = gated
    rows_index = load_trajectory_index()

    print("[diag] loading tokenizer only (no GPU)...", flush=True)
    _, tokenizer, _ = load_hf_model(device_map="cpu")

    report_rows: list[dict] = []
    for cp in cross_pairs:
        pm_tid = str(cp["pm_trajectory_id"])
        pm_traj = rows_index[pm_tid]
        pm_npz = load_activation_npz(activation_path(pm_tid, "original"))
        messages = messages_for_condition(pm_traj, "original", track="CEM")
        tok = tokenize_ccer_messages(messages, tokenizer, output_text=None)
        prompt_len = int(tok["prompt_token_count"])
        stored = int((pm_npz.get("positions") or {}).get(position) or -1)
        final_answer = str((pm_traj.get("metadata") or {}).get("final_answer") or "")
        scan = _scan_decode_steps(
            tokenizer=tokenizer,
            messages=messages,
            prompt_len=prompt_len,
            position=position,
            anchor_pos=stored,
            final_answer=final_answer,
        )
        full_tf = tokenize_ccer_messages(messages, tokenizer, output_text=final_answer)
        live_at_full = resolve_answer_region_token_idx(
            tokenizer=tokenizer,
            seq=list(full_tf["full_ids"]),
            prompt_len=prompt_len,
            position=position,
        )
        report_rows.append(
            {
                "pm_trajectory_id": pm_tid,
                "stored_npz_idx": stored,
                "stored_in_prompt": stored < prompt_len if stored >= 0 else None,
                "stored_vs_prompt_len": stored - prompt_len if stored >= 0 else None,
                "live_at_teacher_forced_full": live_at_full,
                "stored_matches_live_full": stored == live_at_full if stored >= 0 and live_at_full >= 0 else False,
                "v3_gate": bool((pm_npz.get("metadata") or {}).get("refreshed_line_d_v3_symmetric")),
                **scan,
            }
        )

    n_never_patch = sum(1 for r in report_rows if r.get("first_patch_step") is None)
    n_never_live = sum(1 for r in report_rows if r.get("first_live_hit_step") is None)
    summary = {
        "schema_version": "ccer_dynamic_anchor_patch_diagnosis_v2",
        "steering_fix": "dynamic_answer_anchor_patch_positions (live > stored > decode [-1])",
        "n_pairs": len(report_rows),
        "n_never_patch_before_fix": 8,
        "n_never_patch_after_fix": n_never_patch,
        "n_never_live_hit": n_never_live,
        "pairs": report_rows,
    }
    audit_dir = ROOT / "results/p3/round12_line_b/task_j_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    trace_path = audit_dir / "step1_coordinate_trace.jsonl"
    with trace_path.open("w", encoding="utf-8") as fh:
        for row in report_rows[:5]:
            trace_row = {
                "pm_trajectory_id": row["pm_trajectory_id"],
                "stored_npz_idx": row["stored_npz_idx"],
                "prompt_len": row["prompt_len"],
                "first_live_hit_step": row.get("first_live_hit_step"),
                "first_patch_step": row.get("first_patch_step"),
                "live_at_teacher_forced_full": row["live_at_teacher_forced_full"],
                "stored_matches_live_full": row["stored_matches_live_full"],
                "hit_trace": row.get("hit_trace") or [],
            }
            fh.write(json.dumps(trace_row, ensure_ascii=False) + "\n")

    out = ROOT / "results/p3/round12_line_b/dynamic_anchor_patch_diagnosis.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "pairs"}, indent=2))
    print(f"[diag] wrote {out}", flush=True)
    print(f"[diag] wrote {trace_path}", flush=True)


if __name__ == "__main__":
    main()
