"""Fallback generative rewrite when anchor extractive lookup misses (flagged claims only)."""

from __future__ import annotations

import json
from typing import Any, Literal

from ccer.mechanism.line_k_claim_filter import REWRITE_STUB
from ccer.mechanism.line_l_anchor_extract import _anchor_product_record, _final_call_user_content

RewriteFallback = Literal["stub", "llm", "none"]

_LLM_CACHE: dict[str, str] = {}
_DEFAULT_CACHE_PATH = (
    __import__("pathlib").Path("${PHANTOM_MERGE_ROOT}/results/line_l/llm_rewrite_cache.jsonl")
)


def _cache_key(row: dict[str, Any], anchor_pid: str) -> str:
    return "|".join(
        [
            str(row.get("instance_audit_key") or row.get("claim_id") or ""),
            str(anchor_pid),
            str(row.get("response_quote") or ""),
        ]
    )


def stub_rewrite_quote(row: dict[str, Any]) -> str:
    """Deterministic rewrite: retain slot label, replace value with evidence-boundary stub."""
    slot = str(row.get("slot") or row.get("slot_norm") or "Attribute").strip()
    if slot:
        return f"{slot}: {REWRITE_STUB}"
    return REWRITE_STUB


def _anchor_evidence_snippet(traj: dict[str, Any], anchor_pid: str, limit: int = 1200) -> str:
    record = _anchor_product_record(traj, anchor_pid)
    if record:
        return json.dumps(record, ensure_ascii=False)[:limit]
    user = _final_call_user_content(traj)
    idx = user.find(str(anchor_pid))
    if idx >= 0:
        return user[max(0, idx - 80) : idx + 400]
    return ""


def load_llm_cache(path: Any | None = None) -> None:
    """Load disk cache into process memory."""
    from ccer.io_utils import load_jsonl

    cache_path = path or _DEFAULT_CACHE_PATH
    if not cache_path.exists():
        return
    for row in load_jsonl(cache_path):
        key = str(row.get("cache_key") or "")
        quote = str(row.get("quote_after") or "")
        if key and quote:
            _LLM_CACHE[key] = quote


def append_llm_cache(
    cache_key: str,
    *,
    quote_after: str,
    meta: dict[str, Any] | None = None,
    path: Any | None = None,
) -> None:
    from ccer.io_utils import append_jsonl

    cache_path = path or _DEFAULT_CACHE_PATH
    append_jsonl(
        cache_path,
        {
            "cache_key": cache_key,
            "quote_after": quote_after,
            "meta": meta or {},
        },
    )


def llm_rewrite_quote(
    row: dict[str, Any],
    traj: dict[str, Any],
    *,
    anchor_pid: str,
    use_cache: bool = True,
    base_url: str | None = None,
    cache_path: Any | None = None,
) -> tuple[str | None, dict[str, Any]]:
    """Single-line claim rewrite via vLLM; returns None if backend unavailable."""
    import os

    from ccer.paths import DEFAULT_VLLM_BASE_URL
    from ccer.replay.vllm_replay import check_vllm_available, vllm_generate

    url = base_url or os.environ.get("LINE_L_VLLM_BASE_URL") or DEFAULT_VLLM_BASE_URL
    if use_cache and not _LLM_CACHE:
        load_llm_cache(cache_path)

    key = _cache_key(row, anchor_pid)
    if use_cache and key in _LLM_CACHE:
        return _LLM_CACHE[key], {"backend": "cache", "base_url": url}

    if not check_vllm_available(url):
        return None, {"backend": "unavailable", "base_url": url}

    quote = str(row.get("response_quote") or "")
    slot = str(row.get("slot") or row.get("slot_norm") or "")
    evidence = _anchor_evidence_snippet(traj, anchor_pid)
    messages = [
        {
            "role": "system",
            "content": (
                "You rewrite ONE product-attribute claim using ONLY the anchor product evidence. "
                "Output a single line 'Slot: value'. If evidence is insufficient, output exactly: "
                + repr(REWRITE_STUB)
            ),
        },
        {
            "role": "user",
            "content": (
                f"Anchor PID: {anchor_pid}\n"
                f"Original claim: {quote}\n"
                f"Slot: {slot}\n"
                f"Anchor evidence JSON/snippet:\n{evidence}\n"
                "Rewrite:"
            ),
        },
    ]
    text, meta = vllm_generate(
        messages,
        base_url=url,
        model=None,
        final_synthesis=False,
        max_tokens=256,
        temperature=0.0,
    )
    meta = {**meta, "base_url": url}
    out = str(text or "").strip().splitlines()[0].strip()
    if not out:
        return None, meta
    if use_cache:
        _LLM_CACHE[key] = out
        append_llm_cache(key, quote_after=out, meta=meta, path=cache_path)
    return out, meta


def apply_fallback_rewrite(
    row: dict[str, Any],
    traj: dict[str, Any],
    *,
    anchor_pid: str,
    arm: str,
    miss_reason: str | None,
    prior_branch: str,
    fallback: RewriteFallback = "stub",
) -> dict[str, Any]:
    """Rewrite a flagged claim when extractive path would have deleted it."""
    response_quote = str(row.get("response_quote") or "")
    slot_norm = str(row.get("slot_norm") or "")

    quote_after = stub_rewrite_quote(row)
    rewrite_kind = "fallback_stub"
    source_field = "rewrite_stub"
    llm_meta: dict[str, Any] | None = None

    if fallback == "llm":
        llm_quote, llm_meta = llm_rewrite_quote(row, traj, anchor_pid=anchor_pid)
        if llm_quote:
            quote_after = llm_quote
            rewrite_kind = "fallback_llm"
            source_field = "llm_rewrite"

    return {
        "arm": arm,
        "branch": "fallback_rewrite",
        "action": "rewrite",
        "quote_before": response_quote,
        "quote_after": quote_after,
        "v_anchor": None,
        "source_field": source_field,
        "evidence_kind": rewrite_kind,
        "slot_norm": slot_norm,
        "anchor_pid": anchor_pid,
        "miss_reason": miss_reason,
        "prior_branch": prior_branch,
        "llm_meta": llm_meta,
        "mitigation_tier": "rewrite",
        "mitigation_policy": "rewrite_all_flagged_v2",
    }
