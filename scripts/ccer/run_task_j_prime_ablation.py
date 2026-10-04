#!/usr/bin/env python3
"""Task J′: CPU A/B ablation — patch tier distribution J vs J′ (live anchor enhancement)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path("${PHANTOM_MERGE_ROOT}")
sys.path.insert(0, str(ROOT))

from ccer.mechanism.activation_store import activation_path, load_activation_npz
from ccer.mechanism.line_b_protocol import filter_cross_pairs_by_quality
from ccer.mechanism.mechanism_pool import build_cem_mechanism_cross_pairs
from ccer.mechanism.pair_select import load_trajectory_index, messages_for_condition
from ccer.replay.hf_forward import load_hf_tokenizer, tokenize_ccer_messages
from ccer.replay.live_position import (
    J_PRIME_ENV_VAR,
    LINE_D_LIVE_RESOLVER_J_PRIME,
    LINE_D_PROTOCOL_VERSION,
    cross_pair_passes_line_d_v3_gate,
    dynamic_answer_anchor_patch_positions,
    j_prime_live_anchor_enabled,
    resolve_answer_region_token_idx_with_meta,
)

AUDIT_DIR = ROOT / "results/p3/round12_line_b/task_j_prime"
OUT_JSON = AUDIT_DIR / "j_prime_ablation_v1.json"
OUT_MD = AUDIT_DIR / "TASK_J_PRIME_AUDIT_REPORT.md"


def _select_pairs(*, n_pairs: int, pool: str) -> list[dict[str, Any]]:
    cross = build_cem_mechanism_cross_pairs(pool=pool)["pm_clean_cross_pairs"]
    cross_pairs, _ = filter_cross_pairs_by_quality(cross, min_quality=0.35)
    gated: list[dict[str, Any]] = []
    for cp in cross_pairs:
        pm_tid = str(cp["pm_trajectory_id"])
        clean_tid = str(cp["clean_trajectory_id"])
        pm_path = activation_path(pm_tid, "original")
        clean_path = activation_path(clean_tid, "original")
        if not pm_path.is_file() or not clean_path.is_file():
            continue
        pm_npz = load_activation_npz(pm_path)
        clean_npz = load_activation_npz(clean_path)
        if cross_pair_passes_line_d_v3_gate(pm_npz, clean_npz, "claim_onset"):
            gated.append(cp)
        if len(gated) >= n_pairs:
            break
    return gated


def _classify_tier(
    *,
    seq_len: int,
    prompt_len: int,
    anchor_pos: int,
    live_idx: int,
    positions: list[int],
    live_claim_source: str | None,
) -> str | None:
    from ccer.mechanism.interchange import _classify_dynamic_fallback

    return _classify_dynamic_fallback(
        seq_len=seq_len,
        prompt_len=prompt_len,
        anchor_pos=anchor_pos,
        live_idx=live_idx,
        positions=positions,
        live_claim_source=live_claim_source,
    )


def _scan_pair(
    *,
    tokenizer,
    pm_traj: dict[str, Any],
    pm_npz: dict[str, Any],
    position: str = "claim_onset",
    max_new_tokens: int = 384,
) -> dict[str, Any]:
    messages = messages_for_condition(pm_traj, "original", track="CEM")
    final_answer = str((pm_traj.get("metadata") or {}).get("final_answer") or "")
    tok = tokenize_ccer_messages(messages, tokenizer, output_text=None)
    seq = list(tok["full_ids"][: tok["prompt_token_count"]])
    prompt_len = int(tok["prompt_token_count"])
    anchor_pos = int((pm_npz.get("positions") or {}).get(position) or -1)
    full_tf = tokenize_ccer_messages(messages, tokenizer, output_text=final_answer)
    gen_suffix = list(full_tf["full_ids"][prompt_len:])
    steps = min(len(gen_suffix), max_new_tokens)

    tier_counts: Counter[str] = Counter()
    first_live_step: int | None = None
    first_live_source: str | None = None
    first_stored_step: int | None = None
    first_decode_step: int | None = None
    live_steps = 0

    for step in range(steps + 1):
        partial = seq + gen_suffix[:step]
        live_idx, source = resolve_answer_region_token_idx_with_meta(
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
            live_idx=live_idx,
        )
        tier = _classify_tier(
            seq_len=len(partial),
            prompt_len=prompt_len,
            anchor_pos=anchor_pos,
            live_idx=live_idx,
            positions=sites,
            live_claim_source=source,
        )
        if sites and tier:
            tier_counts[tier] += 1
        if live_idx >= 0:
            live_steps += 1
            if first_live_step is None:
                first_live_step = step
                first_live_source = source
        if sites and first_stored_step is None and tier == "stored":
            first_stored_step = step
        if sites and first_decode_step is None and tier == "decode_head":
            first_decode_step = step

    decode_steps = max(steps, 1)
    live_rate = live_steps / decode_steps
    return {
        "first_live_hit_step": first_live_step,
        "first_live_claim_source": first_live_source,
        "first_stored_patch_step": first_stored_step,
        "first_decode_head_patch_step": first_decode_step,
        "live_step_rate": live_rate,
        "tier_step_counts": dict(tier_counts),
        "stored_npz_idx": anchor_pos,
    }


def _run_mode(*, mode: str, pairs: list[dict[str, Any]], tokenizer) -> list[dict[str, Any]]:
    os.environ[J_PRIME_ENV_VAR] = "1" if mode == "j_prime" else "0"
    rows_index = load_trajectory_index()
    out: list[dict[str, Any]] = []
    for cp in pairs:
        pm_tid = str(cp["pm_trajectory_id"])
        pm_traj = rows_index[pm_tid]
        pm_npz = load_activation_npz(activation_path(pm_tid, "original"))
        row = _scan_pair(tokenizer=tokenizer, pm_traj=pm_traj, pm_npz=pm_npz)
        row["pm_trajectory_id"] = pm_tid
        row["mode"] = mode
        out.append(row)
    return out


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tier_totals: Counter[str] = Counter()
    for row in rows:
        tier_totals.update(row.get("tier_step_counts") or {})
    n = len(rows)
    return {
        "n_pairs": n,
        "mean_first_live_hit_step": (
            sum(r["first_live_hit_step"] for r in rows if r.get("first_live_hit_step") is not None)
            / max(1, sum(1 for r in rows if r.get("first_live_hit_step") is not None))
        ),
        "pairs_with_any_live": sum(1 for r in rows if r.get("first_live_hit_step") is not None),
        "mean_live_step_rate": sum(float(r.get("live_step_rate") or 0) for r in rows) / n if n else 0.0,
        "tier_step_totals": dict(tier_totals),
    }


def _write_report(payload: dict[str, Any]) -> None:
    j = payload["j_baseline"]
    jp = payload["j_prime"]
    lines = [
        "# Task J′ Audit: Live Anchor Enhancement A/B",
        "",
        f"**J protocol**: `{LINE_D_PROTOCOL_VERSION}` (symmetric About-only live)",
        f"**J′ protocol**: `{LINE_D_LIVE_RESOLVER_J_PRIME}` (About + Compared bullet + Selected PID)",
        f"**Enable J′**: `{J_PRIME_ENV_VAR}=1`",
        "",
        "## Aggregate (teacher-forced decode trace, claim_onset)",
        "",
        "| Metric | J (baseline) | J′ (enhanced) |",
        "|--------|--------------|---------------|",
        f"| pairs with any live hit | {j['pairs_with_any_live']}/{j['n_pairs']} | "
        f"{jp['pairs_with_any_live']}/{jp['n_pairs']} |",
        f"| mean first live step | {j['mean_first_live_hit_step']:.1f} | {jp['mean_first_live_hit_step']:.1f} |",
        f"| mean live step rate | {j['mean_live_step_rate']:.1%} | {jp['mean_live_step_rate']:.1%} |",
        "",
        "### Tier step totals (J)",
        "",
        f"```json\n{json.dumps(j['tier_step_totals'], indent=2)}\n```",
        "",
        "### Tier step totals (J′)",
        "",
        f"```json\n{json.dumps(jp['tier_step_totals'], indent=2)}\n```",
        "",
        "## Post–H-sweep GPU A/B",
        "",
        "1. Re-run Task H **without** J′ (current baseline checkpoint / env unset).",
        "2. Re-run Task H with `CCER_J_PRIME_LIVE_ANCHOR=1` on **same pairs**.",
        "3. Compare `patch_tier_stratum` in `TASK_H_REPORT.md` — target: ↑ `live*` , ↓ `decode_head`.",
        "",
        f"Full JSON: `{OUT_JSON}`",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Task J′ live anchor A/B (CPU trace)")
    ap.add_argument("--n-pairs", type=int, default=19)
    ap.add_argument("--pool", default="mechanism_research")
    args = ap.parse_args()

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    pairs = _select_pairs(n_pairs=args.n_pairs, pool=args.pool)
    if not pairs:
        raise SystemExit("no gated pairs")

    print(f"[task_j_prime] scanning {len(pairs)} pairs (CPU, no GPU)...", flush=True)
    tokenizer = load_hf_tokenizer()
    j_rows = _run_mode(mode="j_baseline", pairs=pairs, tokenizer=tokenizer)
    jp_rows = _run_mode(mode="j_prime", pairs=pairs, tokenizer=tokenizer)

    payload = {
        "schema_version": "ccer_task_j_prime_ablation_v1",
        "j_protocol": LINE_D_PROTOCOL_VERSION,
        "j_prime_protocol": LINE_D_LIVE_RESOLVER_J_PRIME,
        "j_prime_env_var": J_PRIME_ENV_VAR,
        "n_pairs": len(pairs),
        "j_baseline": _aggregate(j_rows),
        "j_prime": _aggregate(jp_rows),
        "pair_rows": {"j_baseline": j_rows, "j_prime": jp_rows},
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_report(payload)
    print(f"[task_j_prime] wrote {OUT_JSON}", flush=True)
    print(f"[task_j_prime] wrote {OUT_MD}", flush=True)
    print(
        f"[task_j_prime] live pairs J={payload['j_baseline']['pairs_with_any_live']} "
        f"J'={payload['j_prime']['pairs_with_any_live']} "
        f"mean_live_rate J={payload['j_baseline']['mean_live_step_rate']:.1%} "
        f"J'={payload['j_prime']['mean_live_step_rate']:.1%}",
        flush=True,
    )


if __name__ == "__main__":
    main()
