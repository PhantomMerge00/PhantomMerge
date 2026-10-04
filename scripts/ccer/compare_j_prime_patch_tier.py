#!/usr/bin/env python3
"""Compare patch_tier distribution: Task H baseline (J) vs J′ GPU re-run."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path("${PHANTOM_MERGE_ROOT}")
BASELINE_DIR = ROOT / "results/p3/task_h"
J_PRIME_DIR = ROOT / "results/p3/task_h_j_prime"
AUDIT_DIR = ROOT / "results/p3/round12_line_b/task_j_prime"
OUT_JSON = AUDIT_DIR / "j_prime_gpu_ab_patch_tier_v1.json"
OUT_MD = AUDIT_DIR / "TASK_J_PRIME_GPU_AB_REPORT.md"


def _load_pair_details(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _tier_bucket(tier: str | None) -> str:
    t = str(tier or "none")
    if t == "live" or t.startswith("live_"):
        return "live*"
    if t in ("stored", "decode_head", "none", "unknown"):
        return t
    return t


def _count_tiers(details: list[dict[str, Any]], *, arm: str = "target_interchange") -> dict[str, Any]:
    subset = [d for d in details if str(d.get("arm") or "target_interchange") == arm]
    raw = Counter(str(d.get("patch_tier") or "none") for d in subset)
    bucket = Counter(_tier_bucket(d.get("patch_tier")) for d in subset)
    live_sources = Counter(
        str(d.get("live_claim_source") or "none")
        for d in subset
        if str(d.get("patch_tier") or "").startswith("live")
    )
    return {
        "n_rows": len(subset),
        "n_patched": sum(1 for d in subset if d.get("patched")),
        "patch_tier_raw": dict(raw),
        "patch_tier_bucket": dict(bucket),
        "live_claim_source": dict(live_sources),
    }


def _delta_table(j: dict[str, int], jp: dict[str, int]) -> list[dict[str, Any]]:
    keys = sorted(set(j) | set(jp))
    rows: list[dict[str, Any]] = []
    for k in keys:
        jv = int(j.get(k) or 0)
        jpv = int(jp.get(k) or 0)
        rows.append({"tier": k, "j": jv, "j_prime": jpv, "delta": jpv - jv})
    return rows


def _write_md(payload: dict[str, Any]) -> None:
    j = payload["baseline"]["target_interchange"]
    jp = payload["j_prime"]["target_interchange"]
    j_b = j["patch_tier_bucket"]
    jp_b = jp["patch_tier_bucket"]
    delta = _delta_table(j_b, jp_b)

    lines = [
        "# Task J′ GPU A/B: patch_tier Distribution",
        "",
        f"**Baseline (J)**: `{payload['baseline_dir']}`",
        f"**J′ run**: `{payload['j_prime_dir']}` (`CCER_J_PRIME_LIVE_ANCHOR=1`)",
        "",
        "## target_interchange — bucketed patch_tier",
        "",
        "| tier | J (baseline) | J′ | Δ |",
        "|------|--------------|-----|---|",
    ]
    for row in delta:
        lines.append(
            f"| {row['tier']} | {row['j']} | {row['j_prime']} | {row['delta']:+d} |"
        )
    lines += [
        "",
        "### Raw tiers (J)",
        "",
        f"```json\n{json.dumps(j['patch_tier_raw'], indent=2)}\n```",
        "",
        "### Raw tiers (J′)",
        "",
        f"```json\n{json.dumps(jp['patch_tier_raw'], indent=2)}\n```",
        "",
        "### live_claim_source (J′ only)",
        "",
        f"```json\n{json.dumps(jp.get('live_claim_source') or {}, indent=2)}\n```",
        "",
        "**Target**: ↑ `live*` , ↓ `decode_head` vs baseline.",
        "",
        f"Full JSON: `{OUT_JSON}`",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare J vs J′ patch_tier from Task H pair_details")
    ap.add_argument("--baseline-dir", type=Path, default=BASELINE_DIR)
    ap.add_argument("--j-prime-dir", type=Path, default=J_PRIME_DIR)
    args = ap.parse_args()

    baseline_path = args.baseline_dir / "claim_onset_pair_details.jsonl"
    jprime_path = args.j_prime_dir / "claim_onset_pair_details.jsonl"
    baseline_rows = _load_pair_details(baseline_path)
    jprime_rows = _load_pair_details(jprime_path)

    if not baseline_rows:
        raise SystemExit(f"baseline pair_details missing or empty: {baseline_path}")
    if not jprime_rows:
        raise SystemExit(f"j_prime pair_details missing or empty: {jprime_path}")

    payload = {
        "schema_version": "ccer_task_j_prime_gpu_ab_v1",
        "baseline_dir": str(args.baseline_dir),
        "j_prime_dir": str(args.j_prime_dir),
        "baseline": {
            "target_interchange": _count_tiers(baseline_rows, arm="target_interchange"),
            "wrong_owner_donor": _count_tiers(baseline_rows, arm="wrong_owner_donor"),
        },
        "j_prime": {
            "target_interchange": _count_tiers(jprime_rows, arm="target_interchange"),
            "wrong_owner_donor": _count_tiers(jprime_rows, arm="wrong_owner_donor"),
        },
    }

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_md(payload)
    print(f"[j_prime_gpu_ab] wrote {OUT_JSON}", flush=True)
    print(f"[j_prime_gpu_ab] wrote {OUT_MD}", flush=True)

    j_b = payload["baseline"]["target_interchange"]["patch_tier_bucket"]
    jp_b = payload["j_prime"]["target_interchange"]["patch_tier_bucket"]
    print(
        f"[j_prime_gpu_ab] target_interchange decode_head: J={j_b.get('decode_head',0)} "
        f"J'={jp_b.get('decode_head',0)} live*: J={j_b.get('live*',0)} J'={jp_b.get('live*',0)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
