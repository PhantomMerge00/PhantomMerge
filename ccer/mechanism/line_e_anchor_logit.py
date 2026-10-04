"""Owner-guided decode: logit bias toward action_anchor digit tokens."""
from __future__ import annotations

from typing import Any


def build_anchor_logit_bias(
    tokenizer: Any,
    action_anchor: str | None,
    *,
    alpha: float,
    bias_per_token: float = 3.0,
) -> dict[int, float]:
    """
    Bias digit tokens from action_anchor during greedy decode.
    alpha scales bias strength; 0 → no bias.
    """
    if not action_anchor or float(alpha) <= 0:
        return {}
    strength = float(alpha) * float(bias_per_token)
    bias: dict[int, float] = {}

    for ch in str(action_anchor):
        if not ch.isdigit():
            continue
        for tid in tokenizer.encode(ch, add_special_tokens=False):
            bias[int(tid)] = max(bias.get(int(tid), 0.0), strength)

    for tid in tokenizer.encode(str(action_anchor), add_special_tokens=False):
        bias[int(tid)] = max(bias.get(int(tid), 0.0), strength * 0.5)

    return bias
