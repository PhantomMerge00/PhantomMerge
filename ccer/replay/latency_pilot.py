"""Replay pilot runner → replay_audit.json (§5)."""
from __future__ import annotations

import json
import re
import statistics
import sys
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from ccer.io_utils import load_json, load_jsonl, write_json
from ccer.paths import MODEL_MANIFEST_JSON, NORMALIZED_SHOPPING, REPLAY_AUDIT_JSON, REPORTS
from ccer.replay.select_pilot import length_bucket, select_pilot_cohort, write_pilot_manifest
from ccer.replay.vllm_replay import check_vllm_available, vllm_generate

ROOT = Path("${PHANTOM_MERGE_ROOT}")
REPLAY_TOOLS = ROOT / "tools/replay_audit"
if str(REPLAY_TOOLS) not in sys.path:
    sys.path.insert(0, str(REPLAY_TOOLS))


def _extract_response(text: str) -> str:
    m = re.search(r"<response>(.+?)</response>", text, re.DOTALL | re.I)
    if m:
        return m.group(1).strip()
    return text.strip()


def _original_output(row: dict[str, Any]) -> str:
    for step in reversed((row.get("messages_raw") or {}).get("steps") or []):
        content = str((step.get("completion") or {}).get("content") or "")
        if "<response>" in content.lower():
            return content
    return ""


def run_replay_pilot(*, dry_run: bool = False, use_teacher_forced: bool = True) -> dict[str, Any]:
    rows_by_id = {r["trajectory_id"]: r for r in load_jsonl(NORMALIZED_SHOPPING)}
    pilot = select_pilot_cohort(list(rows_by_id.values()))
    write_pilot_manifest(pilot)

    vllm_ok = check_vllm_available()
    greedy_rows: list[dict[str, Any]] = []
    tf_rows: list[dict[str, Any]] = []
    latencies: list[float] = []
    prompt_tokens: list[int] = []

    manifest = load_json(MODEL_MANIFEST_JSON) if MODEL_MANIFEST_JSON.is_file() else {}
    decoding = (manifest.get("manifests") or {}).get("qwen3-32b_shopping_v1", {}).get("decoding") or {}

    for p in pilot:
        tid = p["trajectory_id"]
        row = rows_by_id[tid]
        messages = row.get("messages_final_call") or []
        orig = _original_output(row)
        rec: dict[str, Any] = {
            "trajectory_id": tid,
            "length_bucket": length_bucket(row),
            "exact_token_equality": "not_available",
        }

        if dry_run or not vllm_ok:
            rec["greedy"] = {"status": "not_run", "reason": "dry_run" if dry_run else "vllm_unavailable"}
        else:
            try:
                gen_text, meta = vllm_generate(
                    messages,
                    seed=int(decoding.get("seed", 42)),
                    max_tokens=int(decoding.get("max_tokens", 2048)),
                )
                latencies.append(meta["latency_sec"])
                if meta.get("prompt_tokens"):
                    prompt_tokens.append(int(meta["prompt_tokens"]))
                rec["greedy"] = {
                    "status": "ok",
                    "exact_output_match": gen_text == orig,
                    "edit_similarity": SequenceMatcher(None, orig, gen_text).ratio(),
                    "generated_answer_excerpt": _extract_response(gen_text)[:300],
                    **meta,
                }
            except Exception as exc:
                rec["greedy"] = {"status": "error", "error": str(exc)}
        greedy_rows.append(rec)

        if use_teacher_forced and not dry_run:
            try:
                from ccer.replay.hf_forward import load_hf_tokenizer, tokenize_ccer_messages

                tokenizer = load_hf_tokenizer()
                tok = tokenize_ccer_messages(messages, tokenizer)
                rec["teacher_forced"] = {
                    "status": "tokenized",
                    "prompt_token_count": tok["prompt_token_count"],
                    "output_token_count": len(tok["output_ids"]),
                    "full_token_count": len(tok["full_ids"]),
                }
            except Exception as exc:
                rec["teacher_forced"] = {"status": "error", "error": str(exc)}
        else:
            rec["teacher_forced"] = {"status": "not_run"}
        tf_rows.append(rec)

    audit = {
        "schema_version": "ccer_replay_audit_v1",
        "pilot_n": len(pilot),
        "vllm_available": vllm_ok,
        "limitations": [
            "original_prompt_token_ids never logged",
            "exact_token_equality: not_available",
            "greedy may diverge if template differs from original run",
        ],
        "latency": {
            "p50_sec": statistics.median(latencies) if latencies else None,
            "p95_sec": sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) >= 2 else (latencies[0] if latencies else None),
            "n_samples": len(latencies),
        },
        "prompt_tokens": {
            "p50": int(statistics.median(prompt_tokens)) if prompt_tokens else None,
            "p95": sorted(prompt_tokens)[int(len(prompt_tokens) * 0.95)] if len(prompt_tokens) >= 2 else (prompt_tokens[0] if prompt_tokens else None),
        },
        "greedy_results": greedy_rows,
        "teacher_forced_results": tf_rows,
        "go_no_go": "GO_WITH_LIMITATIONS" if vllm_ok or dry_run else "BLOCKED_VLLM_UNAVAILABLE",
    }
    write_json(REPLAY_AUDIT_JSON, audit)
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "replay_pilot.md").write_text(
        f"# Replay pilot\n\n- n={len(pilot)}\n- vllm={vllm_ok}\n- go={audit['go_no_go']}\n",
        encoding="utf-8",
    )
    return audit


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    print(json.dumps(run_replay_pilot(dry_run=args.dry_run), indent=2, default=str))
