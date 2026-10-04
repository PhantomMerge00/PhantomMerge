"""Live token-index resolution for answer-region interchange (Line D v3)."""
from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from ccer.replay.answer_utils import extract_answer, normalize_final_synthesis_text
from ccer.replay.position_registry import (
    _resolve_claim_onset_pre_value,
    resolve_live_claim_pair,
    resolve_symmetric_claim_pair,
)

ANSWER_REGION_POSITIONS = frozenset({"claim_onset", "pre_value"})
LINE_D_PROTOCOL_VERSION = "line_d_v3_symmetric_live"
LINE_D_LIVE_RESOLVER_J_PRIME = "line_d_v3_symmetric_live_j_prime_v1"
LINE_D_REFRESH_TAG = "refreshed_line_d_v3_symmetric"
J_PRIME_ENV_VAR = "CCER_J_PRIME_LIVE_ANCHOR"
# Line A dev peak @ claim_onset; commitment/prompt_end stay on historical ROI L32.
LINE_D_LAYER_BY_POSITION: dict[str, int] = {
    "prompt_end": 32,
    "commitment": 32,
    "claim_onset": 49,
    "pre_value": 49,
}
LINE_D_CLAIM_LAYER_DEFAULT = 49
LINE_D_ROI_LAYER_DEFAULT = 32


def intervention_steering_apply(position: str) -> str:
    """Answer-region: live re-resolve; commitment: decode-step patch; else anchor_once."""
    if position in ANSWER_REGION_POSITIONS:
        return "dynamic_answer_anchor"
    if position == "commitment":
        return "generation_decode"
    return "anchor_once"


def _offsets_for_seq(tokenizer: Any, seq: list[int]) -> list[tuple[int, int]]:
    text = tokenizer.decode(seq, skip_special_tokens=False)
    enc = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    offsets = [(int(a), int(b)) for a, b in enc["offset_mapping"]]
    if len(offsets) != len(seq):
        if len(offsets) > len(seq):
            offsets = offsets[: len(seq)]
        else:
            offsets = offsets + [(0, 0)] * (len(seq) - len(offsets))
    return offsets


def j_prime_live_anchor_enabled() -> bool:
    """Task J′ live anchor enhancement (Compared bullet + expanded live tier)."""
    return os.environ.get(J_PRIME_ENV_VAR, "0").strip().lower() in ("1", "true", "yes")


def resolve_answer_region_token_idx_with_meta(
    *,
    tokenizer: Any,
    seq: list[int],
    prompt_len: int,
    position: str,
) -> tuple[int, str | None]:
    """Map claim_onset / pre_value to token index; return (idx, claim_source)."""
    if position not in ANSWER_REGION_POSITIONS or prompt_len <= 0 or len(seq) <= prompt_len:
        return -1, None
    text = normalize_final_synthesis_text(tokenizer.decode(seq, skip_special_tokens=False))
    answer_body = extract_answer(text)
    if not answer_body:
        return -1, None
    if j_prime_live_anchor_enabled():
        claim = resolve_live_claim_pair(answer_body)
    else:
        claim = resolve_symmetric_claim_pair(answer_body)
    if claim is None:
        return -1, None
    span, value, source = claim
    offsets = _offsets_for_seq(tokenizer, seq)
    onset, pre_val, _errors = _resolve_claim_onset_pre_value(
        answer_body=answer_body,
        serialized_text=text,
        offset_mapping=offsets,
        response_span=span,
        value=value,
    )
    if position == "claim_onset":
        idx = onset
    else:
        idx = pre_val
    if idx is None or idx < 0 or idx >= len(seq):
        return -1, source
    return int(idx), source


def resolve_answer_region_token_idx(
    *,
    tokenizer: Any,
    seq: list[int],
    prompt_len: int,
    position: str,
) -> int:
    """Map claim_onset / pre_value to a token index in the current generated sequence."""
    idx, _source = resolve_answer_region_token_idx_with_meta(
        tokenizer=tokenizer,
        seq=seq,
        prompt_len=prompt_len,
        position=position,
    )
    return idx


def dynamic_answer_anchor_patch_positions(
    *,
    seq_len: int,
    prompt_len: int,
    decode_step: bool,
    anchor_pos: int,
    live_idx: int,
) -> list[int]:
    """Patch sites for ``claim_onset`` / ``pre_value`` during greedy generation.

    Priority (Task J fix, 2026-09-17):
    1. Live symmetric re-parse (``resolve_answer_region_token_idx``)
    2. Stored npz anchor once ``seq_len`` has reached the teacher-forced index
    3. Decode-head ``[-1]`` on every answer-generation step until (1) or (2) hit

    Without (2)/(3), greedy prompt-only runs never patch: symmetric claim parses
    only after ~300+ answer tokens while PID regex succeeds much earlier → TI≡NI.
    """
    if seq_len <= 0:
        return []
    if 0 <= live_idx < seq_len:
        return [live_idx]
    if prompt_len > 0 and prompt_len <= anchor_pos < seq_len:
        return [anchor_pos]
    if decode_step and seq_len > prompt_len:
        return [-1]
    return []


def make_live_position_resolver(
    tokenizer: Any,
    position: str,
) -> Callable[[list[int], int], int]:
    last_claim_source: list[str | None] = [None]

    def _resolver(seq: list[int], prompt_len: int) -> int:
        idx, source = resolve_answer_region_token_idx_with_meta(
            tokenizer=tokenizer,
            seq=seq,
            prompt_len=prompt_len,
            position=position,
        )
        last_claim_source[0] = source
        return idx

    def peek_claim_source() -> str | None:
        return last_claim_source[0]

    _resolver.peek_claim_source = peek_claim_source  # type: ignore[attr-defined]
    return _resolver


def live_token_idx_for_intervention(
    *,
    pm_npz: dict[str, Any],
    position: str,
    prompt_len: int,
    tokenizer: Any | None = None,
    seq: list[int] | None = None,
) -> int:
    """Return hook token index for interchange at ``position``."""
    if position == "prompt_end":
        return prompt_len - 1 if prompt_len > 0 else -1
    stored = int((pm_npz.get("positions") or {}).get(position) or -1)
    if position in ANSWER_REGION_POSITIONS:
        if tokenizer is not None and seq is not None:
            live = resolve_answer_region_token_idx(
                tokenizer=tokenizer,
                seq=seq,
                prompt_len=prompt_len,
                position=position,
            )
            if live >= 0:
                return live
        if stored >= 0:
            return stored
        return -1
    if stored < 0:
        return -1
    if stored >= prompt_len and position not in ANSWER_REGION_POSITIONS:
        return -1
    return stored


def pair_has_symmetric_claim_anchor(trajectory: dict[str, Any]) -> bool:
    final_answer = str((trajectory.get("metadata") or {}).get("final_answer") or "")
    if not final_answer.strip():
        return False
    body = extract_answer(normalize_final_synthesis_text(final_answer))
    return resolve_symmetric_claim_pair(body) is not None


def line_d_layer_for_position(position: str, *, claim_layer: int = LINE_D_CLAIM_LAYER_DEFAULT) -> int:
    if position in ANSWER_REGION_POSITIONS:
        return int(claim_layer)
    return LINE_D_LAYER_BY_POSITION.get(position, LINE_D_ROI_LAYER_DEFAULT)


def activation_passes_line_d_v3_gate(npz: dict[str, Any], *, position: str | None = None) -> bool:
    """True when npz was extracted under symmetric Line D v3 refresh."""
    meta = npz.get("metadata") or {}
    if not meta.get(LINE_D_REFRESH_TAG):
        return False
    if str(meta.get("claim_anchor_mode") or "") != "symmetric":
        return False
    if position:
        idx = (npz.get("positions") or {}).get(position)
        if idx is None:
            return False
    return True


def cross_pair_passes_line_d_v3_gate(
    pm_npz: dict[str, Any],
    clean_npz: dict[str, Any],
    position: str,
) -> bool:
    if position not in ANSWER_REGION_POSITIONS:
        return True
    return activation_passes_line_d_v3_gate(pm_npz, position=position) and activation_passes_line_d_v3_gate(
        clean_npz, position=position
    )
