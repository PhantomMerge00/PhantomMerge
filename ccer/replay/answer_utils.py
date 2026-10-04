"""Final-answer extraction and replay output validation."""
from __future__ import annotations

import copy
import re

FINAL_SYNTH_USER = (
    "Write the final answer only. Do NOT call any tools. "
    "Use exactly one block: <response>...</response> comparing the selected product."
)


def prepare_final_synthesis_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    out = copy.deepcopy(messages)
    out.append({"role": "user", "content": FINAL_SYNTH_USER})
    out.append({"role": "assistant", "content": "<response>\n"})
    return out


def normalize_final_synthesis_text(text: str) -> str:
    t = str(text or "")
    if not t.lstrip().lower().startswith("<response"):
        t = "<response>\n" + t
    if "</response>" not in t.lower():
        t = t.rstrip() + "\n</response>"
    return t


def extract_answer(text: str) -> str:
    """Extract the final <response> block (last closed, else tail after last opener)."""
    raw = str(text or "")
    blocks = re.findall(r"<response>(.+?)</response>", raw, re.DOTALL | re.IGNORECASE)
    if blocks:
        return blocks[-1].strip()
    tail = re.search(r"<response>\s*(.+)$", raw, re.DOTALL | re.IGNORECASE)
    if tail:
        return tail.group(1).strip()
    return raw.strip()


def extract_selected_product_block(text: str) -> str:
    """Selected-product region only (ID + attributes), excluding Compared section."""
    excerpt = extract_answer(text)
    m = re.search(
        r"(###\s*Selected\s+product\s+ID:.*?)(?=###\s*Compared|\Z)",
        excerpt,
        re.DOTALL | re.IGNORECASE,
    )
    return m.group(1).strip() if m else excerpt.strip()


def extract_selected_product_id(text: str) -> str | None:
    """Primary behavioral metric: Selected product ID from final answer region."""
    raw = str(text or "")
    excerpt = extract_answer(raw)
    ids = re.findall(r"Selected\s+product\s+ID:\s*(\d+)", excerpt, re.IGNORECASE)
    if not ids:
        synth_idx = raw.lower().rfind("write the final answer only")
        tail = raw[synth_idx:] if synth_idx >= 0 else raw
        resp_idx = tail.lower().rfind("<response>")
        if resp_idx >= 0:
            tail = tail[resp_idx:]
        ids = re.findall(r"Selected\s+product\s+ID:\s*(\d+)", tail, re.IGNORECASE)
    return ids[-1] if ids else None


def full_response_excerpt_differ(a: str, b: str) -> bool:
    """Diagnostic: entire <response> excerpt differs (includes Compared prose)."""
    return extract_answer(a) != extract_answer(b)


def selected_product_blocks_differ(a: str, b: str) -> bool:
    """Secondary strict metric: Selected-product block only (no Compared section)."""
    return extract_selected_product_block(a) != extract_selected_product_block(b)


def answers_differ(a: str, b: str) -> bool:
    """Primary behavioral metric: product ID flip, else selected-product block diff."""
    pa, pb = extract_selected_product_id(a), extract_selected_product_id(b)
    if pa and pb:
        if pa != pb:
            return True
        return selected_product_blocks_differ(a, b)
    return selected_product_blocks_differ(a, b)


def validate_replay_output(raw_text: str) -> tuple[str, str | None]:
    """Return (extracted_answer_text, error_code). error_code None => harness OK."""
    text = str(raw_text or "").strip()
    if not text:
        return "", "empty_output"
    lower = text.lower()
    if "<tool_call>" in lower and "<response>" not in lower:
        return extract_answer(text), "tool_call_without_response"
    if "<response>" not in lower:
        return text, "missing_response_tag"
    excerpt = extract_answer(text)
    if not excerpt:
        return excerpt, "empty_response_excerpt"
    return excerpt, None
