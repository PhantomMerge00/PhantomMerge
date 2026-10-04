"""Shared token–value matching for J-lens readout mass (Eq.2 anchor value tokens)."""
from __future__ import annotations

import re


def norm_value(text: str) -> str:
    return re.sub(r"[^0-9a-z]+", "", str(text or "").lower())


def numeric_parts(value: str) -> list[str]:
    nums = re.findall(r"[0-9]+(?:\.[0-9]+)?", str(value or ""))
    return [n for n in nums if n]


def value_tokens_match(token_text: str, value: str) -> bool:
    """True if decoded top-k token text matches anchor/claim value string."""
    if not value:
        return False
    tok_n = norm_value(token_text)
    val_n = norm_value(value)
    if not tok_n or not val_n:
        return False
    if val_n in tok_n or tok_n in val_n:
        return True
    for num in numeric_parts(value):
        if num in tok_n or num in token_text:
            return True
    return False
