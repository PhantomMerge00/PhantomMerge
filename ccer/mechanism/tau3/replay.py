"""Tau2 multi-turn trajectory replay for activation extraction."""
from __future__ import annotations

import json
import re
from typing import Any

import numpy as np

from ccer.replay.hf_forward import teacher_force_on_ids
from ccer.replay.position_registry import char_span_to_token_indices

_SKIP_GREETING_RE = re.compile(
    r"^(hi[!,.]?\s|hello[!,.]?\s|how can i help|you are being transferred)",
    re.IGNORECASE,
)
_JSON_LIKE = re.compile(r"^\s*[\{\[]")


def _normalize_tool_call(tc: dict[str, Any]) -> dict[str, Any]:
    fn = tc.get("function") or {}
    if fn.get("name"):
        name = str(fn["name"])
        args = fn.get("arguments")
    else:
        name = str(tc.get("name") or "")
        args = tc.get("arguments")
    if isinstance(args, dict):
        args_str = json.dumps(args, ensure_ascii=False)
    else:
        args_str = str(args or "{}")
    return {
        "id": str(tc.get("id") or f"call_{name}"),
        "type": "function",
        "function": {"name": name, "arguments": args_str},
    }


def tau2_to_chat_messages(full_trajectory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert tau2 rollout messages to Qwen3 chat-template compatible format."""
    out: list[dict[str, Any]] = []
    for msg in full_trajectory:
        role = str(msg.get("role") or "")
        if role not in ("user", "assistant", "tool", "system"):
            continue
        content = msg.get("content")
        tool_calls = msg.get("tool_calls") or []
        if role == "assistant":
            entry: dict[str, Any] = {"role": "assistant"}
            if content:
                entry["content"] = str(content)
            elif tool_calls:
                entry["content"] = None
            else:
                entry["content"] = ""
            if tool_calls:
                entry["tool_calls"] = [_normalize_tool_call(tc) for tc in tool_calls]
            out.append(entry)
        elif role == "tool":
            out.append(
                {
                    "role": "tool",
                    "content": str(content or ""),
                    "tool_call_id": str(msg.get("tool_call_id") or msg.get("id") or ""),
                }
            )
        else:
            out.append({"role": role, "content": str(content or "")})
    return out


def tokenize_tau2_trajectory(
    messages: list[dict[str, Any]],
    tokenizer: Any,
) -> dict[str, Any]:
    """Tokenize full tau2 conversation for teacher-forced forward."""
    kwargs: dict[str, Any] = {
        "tokenize": True,
        "add_generation_prompt": False,
        "return_dict": True,
    }
    try:
        enc = tokenizer.apply_chat_template(
            messages,
            **kwargs,
            chat_template_kwargs={"enable_thinking": False},
        )
    except TypeError:
        enc = tokenizer.apply_chat_template(messages, **kwargs)

    full_ids = list(enc["input_ids"])
    enc_full = tokenizer(
        tokenizer.decode(full_ids, skip_special_tokens=False),
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    offsets = [(int(a), int(b)) for a, b in enc_full["offset_mapping"]]
    if len(offsets) != len(full_ids):
        offsets = offsets[: len(full_ids)] if len(offsets) > len(full_ids) else offsets + [
            (0, 0)
        ] * (len(full_ids) - len(offsets))

    serialized_text = tokenizer.decode(full_ids, skip_special_tokens=False)
    return {
        "full_ids": full_ids,
        "offset_mapping": offsets,
        "serialized_text": serialized_text,
        "prompt_token_count": 0,
        "n_tokens": len(full_ids),
    }


def _json_quote_variants(quote: str) -> list[str]:
    variants = [quote]
    try:
        obj = json.loads(quote)
        variants.extend(
            [
                json.dumps(obj, ensure_ascii=False, separators=(",", ":")),
                json.dumps(obj, ensure_ascii=False, separators=(",", ": ")),
                json.dumps(obj, ensure_ascii=False),
                json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ": ")),
            ]
        )
    except (json.JSONDecodeError, TypeError):
        pass
    out: list[str] = []
    seen: set[str] = set()
    for v in variants:
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _compact_json(s: str) -> str:
    try:
        return json.dumps(json.loads(s), ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    except (json.JSONDecodeError, TypeError):
        return re.sub(r"\s+", "", s)


def find_quote_span(
    serialized_text: str,
    quote: str,
) -> tuple[int, int] | None:
    """Locate response_quote in serialized replay text."""
    q = str(quote or "").strip()
    if not q:
        return None
    for candidate in _json_quote_variants(q):
        start = serialized_text.find(candidate)
        if start >= 0:
            return start, start + len(candidate)

    compact_q = _compact_json(q)
    if compact_q and compact_q != q:
        compact_text = _compact_json(serialized_text)
        cstart = compact_text.find(compact_q)
        if cstart >= 0:
            # Map compact offset back to original via anchor search on first 24 chars
            anchor = compact_q[: min(24, len(compact_q))]
            for m in re.finditer(re.escape(anchor[:12]) if len(anchor) >= 8 else anchor, serialized_text):
                seg = _compact_json(serialized_text[m.start() : m.start() + len(q) + 120])
                if seg.startswith(compact_q):
                    return m.start(), m.start() + len(q)

    norm_q = re.sub(r"\s+", " ", q)
    for i in range(len(serialized_text)):
        chunk = re.sub(r"\s+", " ", serialized_text[i : i + len(q) + 120])
        if chunk.startswith(norm_q):
            return i, i + len(q)

    lower_text = serialized_text.lower()
    lower_q = q.lower()
    start = lower_text.find(lower_q)
    if start >= 0:
        return start, start + len(q)
    return None


def resolve_tau3_quote_tokens(
    *,
    serialized_text: str,
    offset_mapping: list[tuple[int, int]],
    quote_start: int,
    quote_end: int,
    response_quote: str,
) -> dict[str, int | None]:
    """Map absolute char span → token indices for claim_onset / pre_value."""
    positions: dict[str, int | None] = {
        "prompt_end": None,
        "claim_onset": None,
        "pre_value": None,
    }
    quote_idx = char_span_to_token_indices(offset_mapping, quote_start, quote_end)
    if quote_idx:
        positions["claim_onset"] = quote_idx[0]

    quote_text = str(response_quote or "")
    if ":" in quote_text:
        value = quote_text.split(":", 1)[1].strip()
        if value:
            rel = quote_text.lower().find(value.lower())
            if rel >= 0:
                val_start = quote_start + rel
                pre_idx = char_span_to_token_indices(offset_mapping, quote_start, val_start)
                if pre_idx:
                    positions["pre_value"] = pre_idx[-1]
    return positions


def _extract_tool_json_quote_candidates(messages: list[dict[str, Any]]) -> list[str]:
    """Structure-matched clean quotes: tool-call JSON args and JSON tool observations."""
    quotes: list[str] = []
    seen: set[str] = set()
    for msg in messages or []:
        role = str(msg.get("role") or "")
        if role == "assistant":
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function") or {}
                args = fn.get("arguments") or tc.get("arguments") or ""
                if isinstance(args, dict):
                    args = json.dumps(args, ensure_ascii=False)
                args = str(args).strip()
                if len(args) >= 10 and _JSON_LIKE.match(args) and args not in seen:
                    seen.add(args)
                    quotes.append(args)
        elif role == "tool":
            content = str(msg.get("content") or "").strip()
            if len(content) >= 10 and _JSON_LIKE.match(content) and content not in seen:
                seen.add(content)
                quotes.append(content)
    return quotes


def infer_slot_norm_from_quote(quote: str, *, domain: str = "telecom") -> str:
    """Infer coarse slot_norm for BindSurprise z2 from tool-JSON / attr quote."""
    q = str(quote or "").strip()
    if not q:
        if domain == "retail":
            return "order_id"
        if domain == "telecom":
            return "line_id"
        return "reservation_id"
    try:
        obj = json.loads(q)
        if isinstance(obj, dict) and obj:
            preferred = (
                "line_id",
                "customer_id",
                "phone_number",
                "bill_id",
                "user_id",
                "reservation_id",
                "flight_number",
            )
            for key in preferred:
                if key in obj:
                    return re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")
            first = next(iter(obj.keys()))
            return re.sub(r"[^a-z0-9]+", "_", str(first).lower()).strip("_")
    except (json.JSONDecodeError, TypeError):
        pass
    if ":" in q and not _JSON_LIKE.match(q):
        head = q.split(":", 1)[0].strip()
        if head:
            return re.sub(r"[^a-z0-9]+", "_", head.lower()).strip("_")
    if domain == "retail":
        return "order_id"
    if domain == "telecom":
        return "line_id"
    return "reservation_id"


def structure_matched_clean_quote(
    messages: list[dict[str, Any]],
    serialized_text: str,
    *,
    final_answer: str = "",
) -> tuple[str, int, int] | None:
    """Pick a clean quote from the same structural family as PM tool-JSON anchors."""
    aligned: list[tuple[str, int, int, int]] = []
    for quote in _extract_tool_json_quote_candidates(messages):
        span = find_quote_span(serialized_text, quote)
        if span is not None:
            qs, qe = span
            aligned.append((quote, qs, qe, len(quote)))
    if aligned:
        aligned.sort(key=lambda x: (-x[3], x[1]))
        quote, qs, qe, _ = aligned[0]
        return quote, qs, qe

    if final_answer:
        pseudo = pseudo_clean_quote(final_answer)
        if pseudo is not None:
            quote, _, _ = pseudo
            span = find_quote_span(serialized_text, quote) or find_quote_span(final_answer, quote)
            if span is not None:
                return quote, span[0], span[1]
    return None


def pseudo_clean_quote(final_answer: str) -> tuple[str, int, int] | None:
    """First substantive sentence from final_agent_text for clean trajectories."""
    text = str(final_answer or "").strip()
    if not text:
        return None
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para or _SKIP_GREETING_RE.match(para):
            continue
        for sent in re.split(r"(?<=[.!?])\s+", para):
            sent = sent.strip()
            if len(sent) < 20 or _SKIP_GREETING_RE.match(sent):
                continue
            start = text.find(sent)
            if start < 0:
                start = text.lower().find(sent.lower())
            if start < 0:
                continue
            end = start + len(sent)
            return sent, start, end
    return None


def teacher_force_tau2_trajectory(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    layer_indices: list[int],
) -> dict[str, Any]:
    """One forward pass over full tau2 conversation."""
    tok_pack = tokenize_tau2_trajectory(messages, tokenizer)
    full_ids = tok_pack["full_ids"]
    if not full_ids:
        return {"error": "empty_token_ids", **tok_pack}

    fwd = teacher_force_on_ids(
        model,
        tokenizer,
        full_ids,
        prompt_token_count=1,
        output_ids=full_ids[1:],
    )
    hidden_states = fwd.get("hidden_states")
    return {
        **tok_pack,
        "hidden_states": hidden_states,
        "layer_indices": layer_indices,
    }
