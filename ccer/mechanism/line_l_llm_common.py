"""Shared LLM utilities for Line L+ baselines (RARR, CoVe)."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from ccer.mechanism.line_l_rewrite_quality import build_evidence_corpus, format_valid, verify_grounded

_DEFAULT_MODEL = "Llama-3.1-8B-Instruct"
_DEFAULT_MODEL_PATH = (
    "${PHANTOM_MERGE_ROOT}/runtime/.cache/huggingface/hub/Llama-3.1-8B-Instruct"
)
_DEFAULT_CACHE_DIR = Path("${PHANTOM_MERGE_ROOT}/results/line_l_plus/cache")

_LLM_CACHE: dict[str, str] = {}
_DISK_CACHE_LOADED: set[str] = set()


def default_llm_base_url() -> str:
    return (
        os.environ.get("LINE_L_PLUS_VLLM_BASE_URL")
        or os.environ.get("LINE_L_VLLM_BASE_URL")
        or "http://127.0.0.1:8012/v1"
    )


def ensure_disk_cache_loaded(cache_path: Path | None) -> None:
    """Merge one jsonl cache file into memory (safe across RARR/CoVe cache files)."""
    if not cache_path or not cache_path.is_file():
        return
    path_key = str(cache_path.resolve())
    if path_key in _DISK_CACHE_LOADED:
        return
    load_disk_cache(cache_path)
    _DISK_CACHE_LOADED.add(path_key)


def default_llm_model() -> str:
    return os.environ.get("LINE_L_PLUS_LLM_MODEL") or _DEFAULT_MODEL


def _cache_key(prefix: str, row: dict[str, Any], anchor_pid: str) -> str:
    return "|".join(
        [
            prefix,
            str(row.get("instance_audit_key") or row.get("claim_id") or ""),
            str(anchor_pid),
            str(row.get("response_quote") or ""),
        ]
    )


def load_disk_cache(path: Path) -> None:
    from ccer.io_utils import load_jsonl

    if not path.is_file():
        return
    for row in load_jsonl(path):
        key = str(row.get("cache_key") or "")
        val = str(row.get("response") or row.get("quote_after") or "")
        if key and val:
            _LLM_CACHE[key] = val


def append_disk_cache(path: Path, cache_key: str, response: str, meta: dict[str, Any] | None = None) -> None:
    from ccer.io_utils import append_jsonl

    path.parent.mkdir(parents=True, exist_ok=True)
    append_jsonl(path, {"cache_key": cache_key, "response": response, "meta": meta or {}})


def llm_chat(
    messages: list[dict[str, str]],
    *,
    cache_key: str | None = None,
    cache_path: Path | None = None,
    base_url: str | None = None,
    model: str | None = None,
    max_tokens: int = 256,
) -> tuple[str | None, dict[str, Any]]:
    from ccer.replay.vllm_replay import check_vllm_available, vllm_generate

    url = base_url or default_llm_base_url()
    if cache_path:
        ensure_disk_cache_loaded(cache_path)
    if cache_key and cache_key in _LLM_CACHE:
        return _LLM_CACHE[cache_key], {"backend": "cache", "base_url": url}
    if not check_vllm_available(url):
        return None, {"backend": "unavailable", "base_url": url}
    text, meta = vllm_generate(
        messages,
        base_url=url,
        model=model or default_llm_model(),
        final_synthesis=False,
        max_tokens=max_tokens,
        temperature=0.0,
    )
    out = str(text or "").strip()
    if cache_key and out:
        _LLM_CACHE[cache_key] = out
        if cache_path:
            append_disk_cache(cache_path, cache_key, out, meta)
    return out or None, meta


def parse_slot_value_line(text: str, slot_norm: str) -> tuple[str | None, str | None]:
    """Parse 'SlotLabel: value' from LLM output; reject 'Slot:' prefix errors."""
    line = str(text or "").strip().splitlines()[0].strip()
    if not line:
        return None, None
    if re.match(r"^slot\s*:", line, re.I):
        return None, None
    m = re.match(r"^([^:]+)\s*:\s*(.+)$", line)
    if not m:
        return None, None
    slot = m.group(1).strip()
    val = m.group(2).strip()
    if not format_valid(line, slot_norm):
        return slot, val
    return slot, val


def retrieve_evidence_snippet(
    traj: dict[str, Any],
    anchor_pid: str,
    slot_norm: str,
    *,
    limit: int = 1500,
) -> str:
    corpus = build_evidence_corpus(traj, anchor_pid)
    slot_key = slot_norm.lower()
    lines = corpus.split("\n")
    prioritized = [ln for ln in lines if slot_key in ln.lower()] + lines
    return "\n".join(prioritized[:50])[:limit]


def build_rewrite_messages(
    row: dict[str, Any],
    traj: dict[str, Any],
    anchor_pid: str,
) -> list[dict[str, str]]:
    quote = str(row.get("response_quote") or "")
    slot = str(row.get("slot") or row.get("slot_norm") or "Attribute")
    evidence = retrieve_evidence_snippet(traj, anchor_pid, slot)
    return [
        {
            "role": "system",
            "content": (
                "You rewrite ONE product-attribute claim using ONLY the provided anchor evidence. "
                f"Output exactly one line in the format: {slot}: <value>. "
                "The value MUST be copied verbatim from the evidence. "
                "Do NOT use the word 'Slot' as the label. Do NOT invent values."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Anchor product ID: {anchor_pid}\n"
                f"Original claim: {quote}\n"
                f"Attribute slot: {slot}\n"
                f"Anchor evidence:\n{evidence}\n"
                "Rewritten claim:"
            ),
        },
    ]
