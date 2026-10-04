"""HF teacher-forced scoring and activation extraction for CCER P3."""
from __future__ import annotations

import math
import os
from typing import Any

import numpy as np

from ccer.audit.model_manifest import build_model_manifest
from ccer.replay.answer_utils import normalize_final_synthesis_text, prepare_final_synthesis_messages
from ccer.replay.position_registry import POSITION_NAMES, ClaimAnchorMode, resolve_positions

_MODEL_CACHE: dict[str, tuple[Any, Any, int]] = {}


def sparse_layer_indices(n_layers: int) -> list[int]:
    """Legacy 5-point coarse scan (kept for P3-0 activation cache)."""
    if n_layers <= 0:
        return [0]
    return sorted({0, n_layers // 4, n_layers // 2, (3 * n_layers) // 4, max(0, n_layers - 1)})


def line_d_activation_layer_indices(n_layers: int) -> list[int]:
    """Sparse scan plus Line D ROI L32 and Line A claim peak L49."""
    base = sparse_layer_indices(n_layers)
    if n_layers <= 0:
        return base
    extra = [32, 49]
    merged = set(base)
    for layer in extra:
        merged.add(min(n_layers - 1, max(0, layer)))
    return sorted(merged)


def coarse_layer_indices(n_layers: int) -> list[int]:
    """Two-stage scan stage-1: 9 evenly spaced layers (expert_recorrect §3.1)."""
    if n_layers <= 1:
        return [0]
    fracs = [0, 1 / 8, 2 / 8, 3 / 8, 4 / 8, 5 / 8, 6 / 8, 7 / 8, 1.0]
    return sorted({min(n_layers - 1, max(0, int(round(f * (n_layers - 1))))) for f in fracs})


def dense_layer_indices(n_layers: int, center: int, *, radius: int = 2) -> list[int]:
    """Two-stage scan stage-2: step=1 around coarse peak."""
    lo = max(0, center - radius)
    hi = min(n_layers - 1, center + radius)
    return list(range(lo, hi + 1))


def probe_layer_indices(n_layers: int) -> list[int]:
    """Line A supervised probe: 20%-90% model depth, excluding final 15%."""
    if n_layers <= 0:
        return [0]
    if n_layers == 1:
        return [0]
    last = n_layers - 1
    lo = int(round(0.20 * last))
    hi = int(round(0.90 * last))
    exclude_from = int(round(0.85 * last))
    hi = min(hi, exclude_from - 1)
    if lo > hi:
        return [lo]
    return list(range(lo, hi + 1))


def get_model_manifest_entry() -> dict[str, Any]:
    manifest = build_model_manifest()
    entries = manifest.get("manifests") or {}
    return entries.get("qwen3-32b_shopping_v1") or {}


def load_hf_tokenizer() -> Any:
    from transformers import AutoTokenizer

    entry = get_model_manifest_entry()
    weights_path = str(entry.get("weights_path") or "")
    tokenizer = AutoTokenizer.from_pretrained(weights_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _model_layers(model: Any) -> Any:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "layers"):
        return model.layers
    raise AttributeError("cannot locate transformer layers on model")


def model_input_device(model: Any) -> Any:
    """Device for input_ids; embedding layer with multi-GPU device_map."""
    if hasattr(model, "get_input_embeddings"):
        emb = model.get_input_embeddings()
        if emb is not None and hasattr(emb, "weight"):
            return emb.weight.device
    return next(model.parameters()).device


def load_hf_model(
    *,
    device_map: str = "auto",
    shared_gpu: bool = False,
    gpu_reserve_gib: int = 14,
) -> tuple[Any, Any, int]:
    # Match nvidia-smi index ordering when selecting CUDA_VISIBLE_DEVICES.
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    entry = get_model_manifest_entry()
    weights_path = str(entry.get("weights_path") or "")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    cache_key = f"{weights_path}:{device_map}:{visible}:shared={shared_gpu}:res={gpu_reserve_gib}"
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(weights_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs: dict[str, Any] = {
        "dtype": torch.bfloat16,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if device_map == "cpu":
        load_kwargs["device_map"] = "cpu"
    elif device_map and device_map not in ("none", "auto"):
        load_kwargs["device_map"] = device_map
    elif torch.cuda.is_available():
        n_cuda = torch.cuda.device_count()
        if n_cuda >= 1:
            max_memory: dict[int | str, str] = {}
            caps_gib: list[int] = []
            for i in range(n_cuda):
                free_b, _ = torch.cuda.mem_get_info(i)
                cap_gib = max(1, int(free_b / (1024**3)) - 2)
                caps_gib.append(cap_gib)
                max_memory[i] = f"{cap_gib}GiB"
            total_cap_gib = sum(caps_gib)
            if n_cuda == 1 and caps_gib[0] >= 62 and not shared_gpu:
                load_kwargs["device_map"] = {"": 0}
            else:
                load_kwargs["device_map"] = "auto"
                if shared_gpu and n_cuda == 1:
                    cap = max(8, caps_gib[0] - gpu_reserve_gib)
                    max_memory = {0: f"{cap}GiB", "cpu": "96GiB"}
                load_kwargs["max_memory"] = max_memory
                if total_cap_gib < 62 or shared_gpu:
                    max_memory.setdefault("cpu", "96GiB")
        else:
            load_kwargs["device_map"] = "cpu"
    else:
        load_kwargs["device_map"] = "cpu"
    attn_impl = "flash_attention_2"
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        attn_impl = "sdpa"
    load_kwargs["attn_implementation"] = attn_impl

    model = AutoModelForCausalLM.from_pretrained(weights_path, **load_kwargs)
    model.eval()
    n_layers = int(
        getattr(model.config, "num_hidden_layers", None)
        or getattr(model.config, "n_layer", None)
        or 0
    )
    _MODEL_CACHE[cache_key] = (model, tokenizer, n_layers)
    return model, tokenizer, n_layers


def tokenize_ccer_messages(
    messages: list[dict[str, str]],
    tokenizer: Any,
    *,
    output_text: str | None = None,
) -> dict[str, Any]:
    """Tokenize fixed-history final synthesis messages + optional teacher-forced output."""
    req_messages = prepare_final_synthesis_messages(messages)
    kwargs: dict[str, Any] = {
        "tokenize": True,
        "add_generation_prompt": False,
        "return_dict": True,
    }
    try:
        enc_prompt = tokenizer.apply_chat_template(
            req_messages[:-1],
            **kwargs,
            chat_template_kwargs={"enable_thinking": False},
        )
    except TypeError:
        enc_prompt = tokenizer.apply_chat_template(req_messages[:-1], **kwargs)

    prompt_ids = list(enc_prompt["input_ids"])
    assistant_prefix = str(req_messages[-1].get("content") or "")
    prefix_ids = tokenizer.encode(assistant_prefix, add_special_tokens=False)

    if output_text is None:
        output_text = assistant_prefix
    else:
        norm = normalize_final_synthesis_text(output_text)
        if not norm.startswith(assistant_prefix):
            norm = assistant_prefix + norm[len(assistant_prefix) :] if norm.startswith("<response>") else assistant_prefix + norm
        output_text = norm

    full_text_ids = tokenizer.encode(output_text, add_special_tokens=False)
    output_ids = full_text_ids[len(prefix_ids) :] if len(full_text_ids) >= len(prefix_ids) else full_text_ids
    full_ids = prompt_ids + prefix_ids + output_ids

    enc_full = tokenizer(
        tokenizer.decode(full_ids, skip_special_tokens=False),
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    offsets = [(int(a), int(b)) for a, b in enc_full["offset_mapping"]]
    if len(offsets) != len(full_ids):
        offsets = offsets[: len(full_ids)] if len(offsets) > len(full_ids) else offsets + [(0, 0)] * (
            len(full_ids) - len(offsets)
        )

    prompt_token_count = len(prompt_ids) + len(prefix_ids)
    return {
        "prompt_ids": prompt_ids,
        "prefix_ids": prefix_ids,
        "output_ids": output_ids,
        "full_ids": full_ids,
        "prompt_token_count": prompt_token_count,
        "serialized_text": tokenizer.decode(full_ids, skip_special_tokens=False),
        "offset_mapping": offsets,
        "output_text": output_text,
    }


def _efficient_rank(logits: np.ndarray, token_id: int) -> int:
    target = logits[token_id]
    return int(1 + np.sum(logits > target))


def teacher_force_on_ids(
    model: Any,
    tokenizer: Any,
    full_ids: list[int],
    *,
    prompt_token_count: int,
    output_ids: list[int],
) -> dict[str, Any]:
    import torch

    if not output_ids:
        return {"error": "missing_output_ids", "token_stats": [], "trajectory_summary": {}}

    model_device = next(model.parameters()).device
    input_tensor = torch.tensor([full_ids], device=model_device)
    with torch.no_grad():
        outputs = model(input_tensor, output_hidden_states=True, use_cache=False)
    logits = outputs.logits[0].cpu().float().numpy()
    hidden_states = outputs.hidden_states

    token_stats: list[dict[str, Any]] = []
    start = prompt_token_count
    for i, tid in enumerate(output_ids):
        pos = start + i - 1
        if pos < 0:
            continue
        logit_vec = logits[pos]
        log_probs = logit_vec - np.max(logit_vec)
        probs = np.exp(log_probs)
        probs /= probs.sum()
        p = float(probs[tid])
        token_stats.append(
            {
                "output_index": i,
                "token_id": int(tid),
                "token_str": tokenizer.decode([tid]),
                "log_prob": math.log(max(p, 1e-45)),
                "nll": -math.log(max(p, 1e-45)),
                "prob": p,
                "rank": _efficient_rank(logit_vec, int(tid)),
                "top1_match": int(np.argmax(logit_vec)) == int(tid),
            }
        )

    if token_stats:
        summary = {
            "mean_nll": float(np.mean([t["nll"] for t in token_stats])),
            "sum_log_prob": float(np.sum([t["log_prob"] for t in token_stats])),
            "frac_top1": sum(1 for t in token_stats if t["top1_match"]) / len(token_stats),
            "n_output_tokens": len(token_stats),
        }
    else:
        summary = {}

    return {
        "token_stats": token_stats,
        "trajectory_summary": summary,
        "hidden_states": hidden_states,
        "logits": logits,
    }


def score_candidate_value_tokens(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    candidate_value: str,
    output_text: str | None = None,
) -> dict[str, Any]:
    """Teacher-forced score for candidate value token sequence at claim site."""
    tok = tokenize_ccer_messages(messages, tokenizer, output_text=output_text)
    candidate_ids = tokenizer.encode(str(candidate_value), add_special_tokens=False)
    if not candidate_ids:
        return {"status": "error", "error": "empty_candidate_value"}

    prefix_text = tok["output_text"]
    claim = str(candidate_value)
    if claim not in prefix_text:
        pass

    full_ids = list(tok["full_ids"])
    prompt_count = int(tok["prompt_token_count"])
    output_ids = list(tok["output_ids"])

    value_pos = -1
    decoded_out = tokenizer.decode(output_ids, skip_special_tokens=False)
    idx = decoded_out.lower().find(candidate_value.lower())
    if idx >= 0:
        prefix_out_ids = tokenizer.encode(decoded_out[:idx], add_special_tokens=False)
        value_pos = len(prefix_out_ids)

    if value_pos < 0:
        return {
            "status": "error",
            "error": "candidate_value_not_in_output",
            "candidate_value": candidate_value,
            "candidate_token_ids": candidate_ids,
        }

    score_ids = full_ids[: prompt_count + value_pos] + candidate_ids
    import torch

    model_device = next(model.parameters()).device
    input_tensor = torch.tensor([score_ids], device=model_device)
    with torch.no_grad():
        outputs = model(input_tensor, output_hidden_states=False, use_cache=False)
    logits = outputs.logits[0].cpu().float().numpy()

    token_stats: list[dict[str, Any]] = []
    start = prompt_count + value_pos
    for i, tid in enumerate(candidate_ids):
        pos = start + i - 1
        if pos < 0:
            continue
        logit_vec = logits[pos]
        log_probs = logit_vec - np.max(logit_vec)
        probs = np.exp(log_probs)
        probs /= probs.sum()
        p = float(probs[tid])
        token_stats.append(
            {
                "token_id": int(tid),
                "log_prob": math.log(max(p, 1e-45)),
                "nll": -math.log(max(p, 1e-45)),
                "prob": p,
                "top1_match": int(np.argmax(logit_vec)) == int(tid),
            }
        )

    if not token_stats:
        return {"status": "error", "error": "no_scorable_tokens", "candidate_value": candidate_value}

    return {
        "status": "ok",
        "candidate_value": candidate_value,
        "candidate_token_ids": candidate_ids,
        "prefix_token_count": start,
        "target_token_count": len(candidate_ids),
        "mean_nll": float(np.mean([t["nll"] for t in token_stats])),
        "sum_log_prob": float(np.sum([t["log_prob"] for t in token_stats])),
        "frac_top1": sum(1 for t in token_stats if t["top1_match"]) / len(token_stats),
        "token_stats": token_stats,
    }


def extract_activations(
    model: Any,
    tokenizer: Any,
    *,
    messages: list[dict[str, str]],
    trajectory: dict[str, Any],
    output_text: str | None = None,
    layer_indices: list[int] | None = None,
    claim_anchor_mode: ClaimAnchorMode = "metadata_first",
) -> dict[str, Any]:
    tok = tokenize_ccer_messages(messages, tokenizer, output_text=output_text)
    n_layers = int(getattr(model.config, "num_hidden_layers", 0) or getattr(model.config, "n_layer", 0) or 0)
    layers = layer_indices or sparse_layer_indices(n_layers)

    fwd = teacher_force_on_ids(
        model,
        tokenizer,
        tok["full_ids"],
        prompt_token_count=tok["prompt_token_count"],
        output_ids=tok["output_ids"],
    )
    pos_info = resolve_positions(
        trajectory=trajectory,
        serialized_text=tok["serialized_text"],
        offset_mapping=tok["offset_mapping"],
        prompt_token_count=tok["prompt_token_count"],
        full_token_count=len(tok["full_ids"]),
        claim_anchor_mode=claim_anchor_mode,
    )
    positions = pos_info["positions"]
    hidden_states = fwd["hidden_states"]

    vectors: dict[str, dict[str, np.ndarray]] = {}
    for pos_name in POSITION_NAMES:
        idx = positions.get(pos_name)
        if idx is None:
            continue
        vectors[pos_name] = {}
        for li in layers:
            if li + 1 >= len(hidden_states):
                continue
            h = hidden_states[li + 1][0, int(idx)].detach().cpu().float().numpy()
            vectors[pos_name][f"layer_{li}"] = h.astype(np.float16)

    return {
        "tokenization": {
            "prompt_token_count": tok["prompt_token_count"],
            "full_token_count": len(tok["full_ids"]),
            "output_token_count": len(tok["output_ids"]),
        },
        "positions": positions,
        "position_errors": pos_info["position_errors"],
        "claim_anchor_mode": pos_info.get("claim_anchor_mode"),
        "claim_anchor_source": pos_info.get("claim_anchor_source"),
        "position_ok": pos_info["position_ok"],
        "layer_indices": layers,
        "vectors": vectors,
        "trajectory_summary": fwd.get("trajectory_summary") or {},
    }


def verify_hook_noop(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    atol: float = 1e-3,
) -> dict[str, Any]:
    """P0 hook sanity: forward with no patch matches baseline logits at prompt_end."""
    tok = tokenize_ccer_messages(messages, tokenizer)
    import torch

    model_device = next(model.parameters()).device
    input_tensor = torch.tensor([tok["full_ids"]], device=model_device)
    with torch.no_grad():
        base = model(input_tensor, output_hidden_states=False, use_cache=False)
    base_logits = base.logits[0, tok["prompt_token_count"] - 1].detach().cpu().float()

    def _noop_hook(_module, _inp, out):
        return out

    layer_idx = max(0, int(getattr(model.config, "num_hidden_layers", 1)) // 2)
    handle = _model_layers(model)[layer_idx].register_forward_hook(_noop_hook)
    try:
        with torch.no_grad():
            hooked = model(input_tensor, output_hidden_states=False, use_cache=False)
        hooked_logits = hooked.logits[0, tok["prompt_token_count"] - 1].detach().cpu().float()
    finally:
        handle.remove()

    max_diff = float((base_logits - hooked_logits).abs().max())
    return {
        "max_logit_diff": max_diff,
        "passed": max_diff <= atol,
        "atol": atol,
        "layer_idx": layer_idx,
    }


def teacher_force_score(
    *,
    messages: list[dict[str, str]],
    candidate_value: str,
    output_text: str | None = None,
    model: Any | None = None,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    if model is None or tokenizer is None:
        model, tokenizer, _ = load_hf_model()
    result = score_candidate_value_tokens(
        model,
        tokenizer,
        messages,
        candidate_value=candidate_value,
        output_text=output_text,
    )
    result["messages_hash"] = None
    return result
