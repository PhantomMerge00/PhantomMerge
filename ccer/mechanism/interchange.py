"""Interchange intervention hooks for P3b."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch

from ccer.mechanism.activation_store import get_vector, load_activation_npz
from ccer.replay.answer_utils import normalize_final_synthesis_text, prepare_final_synthesis_messages
from ccer.replay.hf_forward import _model_layers, model_input_device, tokenize_ccer_messages
from ccer.replay.live_position import dynamic_answer_anchor_patch_positions


@dataclass
class JointInterventionSlot:
    """One position in a simultaneous multi-point joint patch (Task I)."""

    position: str
    spec: InterchangeSpec
    anchor_pos: int
    live_resolver: Callable[[list[int], int], int] | None = None
    patched: bool = False


@dataclass
class InterchangeSpec:
    control_id: str
    layer: int
    position_token_idx: int
    alpha: float
    U: np.ndarray
    h_donor: np.ndarray
    h_recipient: np.ndarray
    mode: str = "interchange"  # interchange | ablation | rescue | steering | full_vector | donor_restore
    inject_site: str = "residual"  # residual | attention | mlp
    steering_vec: np.ndarray | None = None
    steering_direction: str = "subtract"  # subtract | add (Line E mitigation default: subtract)
    steering_apply: str = "anchor_once"  # anchor_once | generation_decode | generation_last | generation_all


def _patch_hidden(h: torch.Tensor, spec: InterchangeSpec) -> torch.Tensor:
    if spec.mode == "steering" and spec.steering_vec is not None:
        sv = torch.tensor(spec.steering_vec, device=h.device, dtype=h.dtype)
        sign = -1.0 if str(spec.steering_direction or "subtract").lower() == "subtract" else 1.0
        return h + sign * float(spec.alpha) * sv
    u = torch.tensor(spec.U, device=h.device, dtype=h.dtype)
    hd = torch.tensor(spec.h_donor, device=h.device, dtype=h.dtype)
    hr = torch.tensor(spec.h_recipient, device=h.device, dtype=h.dtype)
    delta = hd - hr
    proj = u @ (u.T @ delta)
    if spec.mode == "ablation":
        return h - u @ (u.T @ h)
    if spec.mode == "rescue":
        base = h - u @ (u.T @ h)
        return base + spec.alpha * (u @ (u.T @ delta))
    if spec.control_id == "self_donor":
        return h
    if spec.alpha == 0.0 or spec.control_id == "no_intervention":
        return h
    if spec.mode == "donor_restore":
        return h + float(spec.alpha) * (hd - h)
    if spec.mode == "full_vector":
        return h + spec.alpha * (hd - hr)
    return h + spec.alpha * proj


def _hook_module(model: Any, spec: InterchangeSpec) -> Any:
    block = _model_layers(model)[spec.layer]
    site = str(spec.inject_site or "residual").lower()
    if site == "attention" and hasattr(block, "self_attn"):
        return block.self_attn
    if site == "mlp" and hasattr(block, "mlp"):
        return block.mlp
    return block


def _slot_active(slot: JointInterventionSlot) -> bool:
    spec = slot.spec
    return (
        spec.control_id not in ("no_intervention", "self_donor")
        and float(spec.alpha) != 0.0
    )


def _slot_patch_positions(
    slot: JointInterventionSlot,
    seq: list[int],
    *,
    prompt_len: int,
    decode_step: bool,
) -> list[int]:
    if not _slot_active(slot) or len(seq) <= 0:
        return []
    spec = slot.spec
    anchor_pos = int(slot.anchor_pos)
    steering_apply = str(getattr(spec, "steering_apply", "anchor_once") or "anchor_once").lower()
    if steering_apply == "generation_decode":
        if decode_step:
            return [-1]
        return [anchor_pos] if 0 <= anchor_pos < len(seq) else []
    if steering_apply == "generation_all":
        return list(range(len(seq)))
    if steering_apply == "generation_last":
        return [len(seq) - 1]
    if steering_apply == "generation_suffix":
        return (
            list(range(max(0, prompt_len), len(seq)))
            if len(seq) > prompt_len
            else [len(seq) - 1]
        )
    if steering_apply == "dynamic_answer_anchor":
        live_idx = -1
        if slot.live_resolver is not None:
            live_idx = int(slot.live_resolver(seq, prompt_len))
        return dynamic_answer_anchor_patch_positions(
            seq_len=len(seq),
            prompt_len=prompt_len,
            decode_step=decode_step,
            anchor_pos=anchor_pos,
            live_idx=live_idx,
        )
    if 0 <= anchor_pos < len(seq):
        return [anchor_pos]
    return []


def _classify_dynamic_fallback(
    *,
    seq_len: int,
    prompt_len: int,
    anchor_pos: int,
    live_idx: int,
    positions: list[int],
    live_claim_source: str | None = None,
) -> str | None:
    if not positions:
        return None
    idx = positions[0]
    if idx == -1:
        return "decode_head"
    if 0 <= live_idx < seq_len and idx == live_idx:
        if live_claim_source and live_claim_source not in ("about_first_bullet",):
            return f"live_{live_claim_source}"
        return "live"
    if prompt_len > 0 and prompt_len <= anchor_pos < seq_len and idx == anchor_pos:
        return "stored"
    return "unknown"


def _peek_live_claim_source(resolver: Any | None) -> str | None:
    if resolver is None:
        return None
    peek = getattr(resolver, "peek_claim_source", None)
    return peek() if callable(peek) else None


def _joint_slot_group_key(slot: JointInterventionSlot) -> tuple[int, str]:
    site = str(slot.spec.inject_site or "residual").lower()
    return (int(slot.spec.layer), site)


def greedy_generate_with_joint_hooks(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    slots: list[JointInterventionSlot],
    output_text: str | None = None,
    max_new_tokens: int = 512,
    use_cache: bool | None = None,
) -> dict[str, Any]:
    """Simultaneous multi-position interchange during one greedy generation (Task I)."""
    tok = tokenize_ccer_messages(messages, tokenizer, output_text=output_text)
    seq = list(tok["full_ids"][: tok["prompt_token_count"]])
    device = model_input_device(model)
    prompt_len = int(tok["prompt_token_count"])
    generated: list[int] = []
    handles: list[Any] = []
    patch_fallback_used: dict[str, str | None] = {
        str(s.position): None for s in slots
    }
    if use_cache is None:
        use_cache = False

    def _clear_hooks() -> None:
        for handle in handles:
            handle.remove()
        handles.clear()

    def _register_joint_hooks(*, decode_step: bool) -> None:
        _clear_hooks()
        groups: dict[tuple[int, str], list[JointInterventionSlot]] = {}
        for slot in slots:
            if not _slot_active(slot):
                continue
            groups.setdefault(_joint_slot_group_key(slot), []).append(slot)

        for group_slots in groups.values():
            layer = group_slots[0].spec.layer
            ref_spec = group_slots[0].spec
            patch_plan: list[tuple[JointInterventionSlot, list[int]]] = []
            for slot in group_slots:
                positions = _slot_patch_positions(
                    slot,
                    seq,
                    prompt_len=prompt_len,
                    decode_step=decode_step,
                )
                if not positions:
                    continue
                patch_plan.append((slot, positions))
                steering_apply = str(
                    getattr(slot.spec, "steering_apply", "anchor_once") or "anchor_once"
                ).lower()
                if steering_apply == "dynamic_answer_anchor":
                    live_idx = -1
                    if slot.live_resolver is not None:
                        live_idx = int(slot.live_resolver(seq, prompt_len))
                    patch_fallback_used[str(slot.position)] = _classify_dynamic_fallback(
                        seq_len=len(seq),
                        prompt_len=prompt_len,
                        anchor_pos=int(slot.anchor_pos),
                        live_idx=live_idx,
                        positions=positions,
                        live_claim_source=_peek_live_claim_source(slot.live_resolver),
                    )
            if not patch_plan:
                continue

            def _make_hook(plan: list[tuple[JointInterventionSlot, list[int]]]):
                def _hook(_module, _inp, output):
                    out = output[0] if isinstance(output, tuple) else output
                    if out.ndim != 3:
                        return output
                    out = out.clone()
                    for slot, positions in plan:
                        spec = slot.spec
                        for pos in positions:
                            idx = pos if pos >= 0 else out.shape[1] - 1
                            if idx >= out.shape[1]:
                                continue
                            h = out[0, idx, :]
                            out[0, idx, :] = _patch_hidden(h, spec)
                            if spec.control_id not in ("self_donor", "no_intervention"):
                                slot.patched = True
                    if isinstance(output, tuple):
                        return (out,) + output[1:]
                    return out

                return _hook

            handles.append(
                _hook_module(model, ref_spec).register_forward_hook(_make_hook(patch_plan))
            )

    past = None
    try:
        with torch.no_grad():
            for _ in range(max_new_tokens):
                if use_cache:
                    decode_step = past is not None
                    _register_joint_hooks(decode_step=decode_step)
                    input_ids = (
                        torch.tensor([seq], device=device)
                        if past is None
                        else torch.tensor([[seq[-1]]], device=device)
                    )
                    outputs = model(input_ids=input_ids, past_key_values=past, use_cache=True)
                    _clear_hooks()
                    past = outputs.past_key_values
                else:
                    decode_step = len(generated) > 0
                    _register_joint_hooks(decode_step=decode_step)
                    input_ids = torch.tensor([seq], device=device)
                    outputs = model(input_ids=input_ids, use_cache=False)
                    _clear_hooks()
                next_id = int(torch.argmax(outputs.logits[0, -1]).item())
                generated.append(next_id)
                if next_id == tokenizer.eos_token_id:
                    seq.append(next_id)
                    break
                seq.append(next_id)
                if "</response>" in tokenizer.decode(seq[-8:]):
                    break
    finally:
        _clear_hooks()

    patched_by_position = {str(s.position): bool(s.patched) for s in slots}
    expected_positions = {str(s.position) for s in slots if _slot_active(s)}
    all_patched = bool(expected_positions) and all(
        patched_by_position.get(pos, False) for pos in expected_positions
    )
    gen_suffix = tokenizer.decode(seq[prompt_len:], skip_special_tokens=False)
    gen_text = normalize_final_synthesis_text(gen_suffix)
    return {
        "text": gen_text,
        "full_text": normalize_final_synthesis_text(tokenizer.decode(seq, skip_special_tokens=False)),
        "generated_token_ids": generated,
        "prompt_token_count": prompt_len,
        "patched_by_position": patched_by_position,
        "all_patched": all_patched,
        "patch_fallback_used": patch_fallback_used,
        "use_cache": use_cache,
        "n_active_slots": len(expected_positions),
    }


def greedy_generate_with_hook(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    spec: InterchangeSpec | None,
    output_text: str | None = None,
    max_new_tokens: int = 512,
    use_cache: bool | None = None,
    live_position_resolver: Callable[[list[int], int], int] | None = None,
    logit_bias: dict[int, float] | None = None,
    logit_bias_decode_only: bool = True,
) -> dict[str, Any]:
    tok = tokenize_ccer_messages(messages, tokenizer, output_text=output_text)
    seq = list(tok["full_ids"][: tok["prompt_token_count"]])
    device = model_input_device(model)
    prompt_len = int(tok["prompt_token_count"])
    anchor_pos = spec.position_token_idx if spec else -1
    steering_apply = str(getattr(spec, "steering_apply", "anchor_once") or "anchor_once").lower()
    generated: list[int] = []
    patched = False
    patch_tier: str | None = None
    live_claim_source: str | None = None
    first_patch_step: int | None = None
    handle = None
    active_intervention = (
        spec is not None
        and spec.control_id not in ("no_intervention", "self_donor")
        and float(spec.alpha) != 0.0
    )
    if use_cache is None:
        # Default OFF: NI/TI must share identical decode path (Line B fix #4).
        use_cache = False

    def _patch_positions(seq_len: int, *, decode_step: bool) -> list[int]:
        if not active_intervention or seq_len <= 0:
            return []
        if steering_apply == "generation_decode":
            if decode_step:
                return [-1]
            return [anchor_pos] if 0 <= anchor_pos < seq_len else []
        if steering_apply == "generation_all":
            return list(range(seq_len))
        if steering_apply == "generation_last":
            return [seq_len - 1]
        if steering_apply == "generation_suffix":
            return list(range(max(0, prompt_len), seq_len)) if seq_len > prompt_len else [seq_len - 1]
        if steering_apply == "dynamic_answer_anchor":
            live_idx = -1
            if live_position_resolver is not None:
                live_idx = int(live_position_resolver(seq, prompt_len))
            return dynamic_answer_anchor_patch_positions(
                seq_len=seq_len,
                prompt_len=prompt_len,
                decode_step=decode_step,
                anchor_pos=anchor_pos,
                live_idx=live_idx,
            )
        if 0 <= anchor_pos < seq_len:
            return [anchor_pos]
        return []

    def _clear_hook() -> None:
        nonlocal handle
        if handle is not None:
            handle.remove()
            handle = None

    def _register_hook(*, decode_step: bool) -> None:
        nonlocal handle, patched, patch_tier, live_claim_source, first_patch_step
        _clear_hook()
        positions = _patch_positions(len(seq), decode_step=decode_step)
        if not positions:
            return
        if steering_apply == "dynamic_answer_anchor":
            live_idx = -1
            if live_position_resolver is not None:
                live_idx = int(live_position_resolver(seq, prompt_len))
            live_claim_source = _peek_live_claim_source(live_position_resolver)
            tier = _classify_dynamic_fallback(
                seq_len=len(seq),
                prompt_len=prompt_len,
                anchor_pos=anchor_pos,
                live_idx=live_idx,
                positions=positions,
                live_claim_source=live_claim_source,
            )
            if tier and patch_tier is None:
                patch_tier = tier

        def _hook(_module, _inp, output):
            nonlocal patched, first_patch_step
            out = output[0] if isinstance(output, tuple) else output
            if out.ndim != 3:
                return output
            out = out.clone()
            for pos in positions:
                idx = pos if pos >= 0 else out.shape[1] - 1
                if idx >= out.shape[1]:
                    continue
                h = out[0, idx, :]
                out[0, idx, :] = _patch_hidden(h, spec)
                if spec.control_id not in ("self_donor", "no_intervention"):
                    if not patched and decode_step:
                        first_patch_step = len(generated)
                    patched = True
            if isinstance(output, tuple):
                return (out,) + output[1:]
            return out

        handle = _hook_module(model, spec).register_forward_hook(_hook)

    past = None
    try:
        with torch.no_grad():
            for _ in range(max_new_tokens):
                if use_cache:
                    decode_step = past is not None
                    if active_intervention:
                        _register_hook(decode_step=decode_step)
                    input_ids = (
                        torch.tensor([seq], device=device)
                        if past is None
                        else torch.tensor([[seq[-1]]], device=device)
                    )
                    outputs = model(input_ids=input_ids, past_key_values=past, use_cache=True)
                    _clear_hook()
                    past = outputs.past_key_values
                else:
                    # Full-sequence re-forward each step: after the first sampled token,
                    # treat as decode step so generation_decode patches [-1] (Line F v3).
                    decode_step = len(generated) > 0
                    if active_intervention:
                        _register_hook(decode_step=decode_step)
                    input_ids = torch.tensor([seq], device=device)
                    outputs = model(input_ids=input_ids, use_cache=False)
                    _clear_hook()
                logits = outputs.logits[0, -1]
                if logit_bias and (not logit_bias_decode_only or len(generated) > 0):
                    for tok_id, bias_val in logit_bias.items():
                        logits[int(tok_id)] = logits[int(tok_id)] + float(bias_val)
                next_id = int(torch.argmax(logits).item())
                generated.append(next_id)
                if next_id == tokenizer.eos_token_id:
                    seq.append(next_id)
                    break
                piece = tokenizer.decode([next_id])
                seq.append(next_id)
                if "</response>" in tokenizer.decode(seq[-8:]):
                    break
    finally:
        _clear_hook()

    gen_suffix = tokenizer.decode(seq[prompt_len:], skip_special_tokens=False)
    gen_text = normalize_final_synthesis_text(gen_suffix)
    return {
        "text": gen_text,
        "full_text": normalize_final_synthesis_text(tokenizer.decode(seq, skip_special_tokens=False)),
        "generated_token_ids": generated,
        "prompt_token_count": prompt_len,
        "spec": spec.control_id if spec else "no_intervention",
        "patched": patched,
        "patch_tier": patch_tier or ("none" if not patched else "unknown"),
        "live_claim_source": live_claim_source,
        "first_patch_step": first_patch_step,
        "target_position": anchor_pos,
        "steering_apply": steering_apply,
        "use_cache": use_cache,
        "inject_site": spec.inject_site if spec else "residual",
    }


def load_hidden_at(
    npz_path: str | Any,
    *,
    position: str,
    layer: int,
) -> np.ndarray | None:
    loaded = load_activation_npz(npz_path) if not isinstance(npz_path, dict) else npz_path
    return get_vector(loaded, position=position, layer=layer)


def build_spec_from_roi(
    *,
    control_id: str,
    roi: dict[str, Any],
    donor_vec: np.ndarray,
    recipient_vec: np.ndarray,
    position_token_idx: int,
    alpha: float = 1.0,
    U_override: np.ndarray | None = None,
    layer_override: int | None = None,
    mode: str = "interchange",
    inject_site: str = "residual",
) -> InterchangeSpec:
    primary = roi.get("primary_roi") or {}
    u = np.array(U_override if U_override is not None else roi.get("U_owner") or [], dtype=np.float32)
    if u.ndim == 1:
        u = u.reshape(-1, 1)
    if u.shape[0] != donor_vec.shape[0]:
        u = np.eye(len(donor_vec), min(4, len(donor_vec)), dtype=np.float32)
    layer = int(layer_override if layer_override is not None else primary.get("layer") or 0)
    site = str(inject_site or roi.get("inject_site") or "residual")
    return InterchangeSpec(
        control_id=control_id,
        layer=layer,
        position_token_idx=position_token_idx,
        alpha=alpha,
        U=u,
        h_donor=donor_vec.astype(np.float32),
        h_recipient=recipient_vec.astype(np.float32),
        mode=mode,
        inject_site=site,
        steering_apply=str(roi.get("steering_apply") or "anchor_once"),
    )


def build_steering_spec(
    *,
    control_id: str,
    layer: int,
    position_token_idx: int,
    steering_vec: np.ndarray,
    alpha: float = 1.0,
    steering_direction: str = "subtract",
    inject_site: str = "residual",
    steering_apply: str = "anchor_once",
) -> InterchangeSpec:
    """CAA-style additive steering (Line E)."""
    dim = int(np.asarray(steering_vec).reshape(-1).shape[0])
    return InterchangeSpec(
        control_id=control_id,
        layer=layer,
        position_token_idx=position_token_idx,
        alpha=float(alpha),
        U=np.zeros((dim, 1), dtype=np.float32),
        h_donor=np.zeros(dim, dtype=np.float32),
        h_recipient=np.zeros(dim, dtype=np.float32),
        mode="steering",
        inject_site=inject_site,
        steering_vec=np.asarray(steering_vec, dtype=np.float32).reshape(-1),
        steering_direction=steering_direction,
        steering_apply=steering_apply,
    )


def build_donor_restore_spec(
    *,
    control_id: str,
    layer: int,
    position_token_idx: int,
    donor_vec: np.ndarray,
    alpha: float = 1.0,
    inject_site: str = "residual",
    steering_apply: str = "dynamic_answer_anchor",
) -> InterchangeSpec:
    """Live convex restoration toward donor: h + α*(h_donor − h)."""
    dim = int(np.asarray(donor_vec).reshape(-1).shape[0])
    return InterchangeSpec(
        control_id=control_id,
        layer=layer,
        position_token_idx=position_token_idx,
        alpha=float(alpha),
        U=np.zeros((dim, 1), dtype=np.float32),
        h_donor=np.asarray(donor_vec, dtype=np.float32).reshape(-1),
        h_recipient=np.zeros(dim, dtype=np.float32),
        mode="donor_restore",
        inject_site=inject_site,
        steering_vec=None,
        steering_direction="subtract",
        steering_apply=steering_apply,
    )


def build_paired_full_vector_spec(
    *,
    control_id: str,
    layer: int,
    position_token_idx: int,
    pm_vec: np.ndarray,
    clean_vec: np.ndarray,
    alpha: float = 1.0,
    inject_site: str = "residual",
    steering_apply: str = "generation_decode",
) -> InterchangeSpec:
    """Trajectory-specific clean donor restoration (Line G style, Line E v2)."""
    dim = int(np.asarray(pm_vec).reshape(-1).shape[0])
    return InterchangeSpec(
        control_id=control_id,
        layer=layer,
        position_token_idx=position_token_idx,
        alpha=float(alpha),
        U=np.zeros((dim, 1), dtype=np.float32),
        h_donor=np.asarray(clean_vec, dtype=np.float32).reshape(-1),
        h_recipient=np.asarray(pm_vec, dtype=np.float32).reshape(-1),
        mode="full_vector",
        inject_site=inject_site,
        steering_vec=None,
        steering_direction="subtract",
        steering_apply=steering_apply,
    )
