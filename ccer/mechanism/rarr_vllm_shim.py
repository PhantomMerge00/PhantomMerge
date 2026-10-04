"""vLLM chat/completion shim for vendor RARR prompts (replaces OpenAI API)."""

from __future__ import annotations

import re
from typing import Any

from ccer.mechanism.line_l_llm_common import default_llm_base_url, default_llm_model, llm_chat

_COMPLETION_CACHE: dict[str, str] = {}


def _prompt_to_messages(prompt: str) -> list[dict[str, str]]:
    """Convert RARR completion-style prompt to chat messages."""
    text = str(prompt or "").strip()
    if "\nYou said:" in text or text.startswith("You said:"):
        return [
            {
                "role": "system",
                "content": (
                    "Continue the pattern exactly. Follow the format shown in the examples. "
                    "Do not add commentary outside the expected lines."
                ),
            },
            {"role": "user", "content": text},
        ]
    return [
        {
            "role": "system",
            "content": "You are a careful fact-checking assistant following the given template.",
        },
        {"role": "user", "content": text},
    ]


def completion_create(
    *,
    prompt: str,
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 512,
    stop: list[str] | None = None,
    cache_key: str | None = None,
    cache_path: Any = None,
) -> dict[str, Any]:
    """Drop-in subset of openai.Completion.create for RARR utils."""
    key = cache_key or f"rarr|{hash(prompt)}|{temperature}|{max_tokens}"
    if key in _COMPLETION_CACHE:
        text = _COMPLETION_CACHE[key]
    else:
        messages = _prompt_to_messages(prompt)
        text, _meta = llm_chat(
            messages,
            cache_key=key,
            cache_path=cache_path,
            base_url=default_llm_base_url(),
            model=model or default_llm_model(),
            max_tokens=max_tokens,
        )
        text = str(text or "")
        if stop:
            for s in stop:
                if s in text:
                    text = text.split(s)[0]
        _COMPLETION_CACHE[key] = text

    return {"choices": [{"text": text}]}


def parse_rarr_questions(api_response: str) -> list[str]:
    search_string = "I googled:"
    questions: list[str] = []
    for line in str(api_response or "").split("\n"):
        if search_string not in line:
            continue
        q = line.split(search_string, 1)[1].strip()
        if q:
            questions.append(q)
    return questions


def parse_agreement_gate(api_response: str) -> tuple[bool, str, str]:
    lines = str(api_response or "").strip().split("\n")
    if len(lines) < 2:
        return False, "Failed to parse.", ""
    reason = lines[0]
    decision = lines[1].split("Therefore:")[-1].strip() if "Therefore:" in lines[1] else lines[1]
    is_open = "disagrees" in lines[1].lower()
    return is_open, reason, decision


def parse_editor_response(api_response: str) -> str | None:
    lines = str(api_response or "").strip().split("\n")
    if len(lines) < 2:
        return None
    edited = lines[1].split("My fix:")[-1].strip()
    return edited or None


def parse_slot_line_from_edit(text: str, slot_norm: str) -> str | None:
    """Extract slot:value from RARR editor output or raw line."""
    line = str(text or "").strip().splitlines()[0].strip()
    if re.match(r"^slot\s*:", line, re.I):
        return None
    m = re.match(r"^([^:]+)\s*:\s*(.+)$", line)
    if m:
        return m.group(2).strip()
    return line if line else None
