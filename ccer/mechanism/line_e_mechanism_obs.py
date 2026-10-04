"""Line E v4: mechanism-layer observables (hidden cosine, anchor logit rank)."""
from __future__ import annotations

import re
from typing import Any

import numpy as np
import torch

from ccer.mechanism.interchange import InterchangeSpec, _patch_hidden
from ccer.replay.hf_forward import _model_layers, model_input_device, tokenize_ccer_messages


def _anchor_digit_token_ids(tokenizer: Any, anchor_pid: str) -> list[int]:
    digits = re.sub(r"\D", "", str(anchor_pid or ""))
    if not digits:
        return []
    ids: list[int] = []
    for ch in digits:
        piece = tokenizer.encode(ch, add_special_tokens=False)
        if piece:
            ids.append(int(piece[-1]))
    return ids


def _cosine(a: np.ndarray, b: np.ndarray) -> float | None:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-8 or nb < 1e-8:
        return None
    return float(np.dot(a, b) / (na * nb))


def measure_claim_onset_mechanism(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    spec: InterchangeSpec,
    claim_token_idx: int,
    donor_vec: np.ndarray | None,
    action_anchor: str | None,
) -> dict[str, Any]:
    """Single prefill forward at claim_onset: hidden shift + anchor digit logit rank."""
    if claim_token_idx < 0:
        return {"mechanism_ok": False, "error": "invalid_claim_token_idx"}

    tok = tokenize_ccer_messages(messages, tokenizer, output_text=None)
    seq = list(tok["full_ids"][: tok["prompt_token_count"]])
    if claim_token_idx >= len(seq):
        return {"mechanism_ok": False, "error": "claim_idx_beyond_prefill"}

    device = model_input_device(model)
    input_ids = torch.tensor([seq], device=device)
    layer = int(spec.layer)
    captured: dict[str, torch.Tensor] = {}

    def _hook(_module, _inp, output):
        out = output[0] if isinstance(output, tuple) else output
        captured["h"] = out[0, claim_token_idx, :].detach().clone()
        return output

    handle = _model_layers(model)[layer].register_forward_hook(_hook)
    try:
        with torch.no_grad():
            outputs = model(input_ids=input_ids, use_cache=False)
            logits = outputs.logits[0, claim_token_idx].detach().cpu().float().numpy()
    finally:
        handle.remove()

    if "h" not in captured:
        return {"mechanism_ok": False, "error": "hidden_not_captured"}

    h_before = captured["h"]
    h_after = _patch_hidden(h_before, spec)
    donor = np.asarray(donor_vec, dtype=np.float32).reshape(-1) if donor_vec is not None else None

    obs: dict[str, Any] = {
        "mechanism_ok": True,
        "claim_token_idx": int(claim_token_idx),
        "hidden_norm_before": float(torch.linalg.norm(h_before.float()).item()),
        "hidden_delta_l2": float(torch.linalg.norm((h_after - h_before).float()).item()),
        "hidden_cosine_to_donor_before": _cosine(h_before.cpu().numpy(), donor) if donor is not None else None,
        "hidden_cosine_to_donor_after": _cosine(h_after.cpu().numpy(), donor) if donor is not None else None,
    }
    if donor is not None:
        obs["hidden_cosine_to_donor_delta"] = (
            (obs["hidden_cosine_to_donor_after"] or 0.0) - (obs["hidden_cosine_to_donor_before"] or 0.0)
            if obs["hidden_cosine_to_donor_after"] is not None and obs["hidden_cosine_to_donor_before"] is not None
            else None
        )

    anchor_ids = _anchor_digit_token_ids(tokenizer, action_anchor or "")
    if anchor_ids:
        ranks = []
        for tid in anchor_ids:
            target = float(logits[tid])
            ranks.append(int(1 + np.sum(logits > target)))
        obs["anchor_digit_token_ids"] = anchor_ids
        obs["anchor_logit_mean"] = float(np.mean([logits[tid] for tid in anchor_ids]))
        obs["anchor_logit_rank_mean"] = float(np.mean(ranks))
        obs["anchor_logit_top1_any"] = bool(any(int(np.argmax(logits)) == tid for tid in anchor_ids))
    else:
        obs["anchor_digit_token_ids"] = []
        obs["anchor_logit_mean"] = None
        obs["anchor_logit_rank_mean"] = None
        obs["anchor_logit_top1_any"] = None

    return obs
