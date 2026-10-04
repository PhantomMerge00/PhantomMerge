"""J-space coordinate intervention (paper formula thin adapter).

Uses official jlens.JacobianLens for transport; implements the published
coordinate-swap algebra h_patched = h + V(σ(c) - c) without reimplementing J_l fitting.
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np
import torch

from jlens.lens import JacobianLens
from jlens.protocol import LensModel


def lens_direction_for_token(
    lens: JacobianLens,
    jlens_model: LensModel,
    layer: int,
    token_id: int,
) -> np.ndarray:
    """Approximate J-lens direction for vocabulary token in residual space at layer."""
    J = lens.jacobians[layer].float()
    hf_model = getattr(jlens_model, "_hf_model", None)
    if hf_model is None:
        raise ValueError("jlens_model must be HFLensModel with _hf_model")
    lm_head = getattr(jlens_model, "_lm_head", None)
    if lm_head is None:
        lm_head = hf_model.lm_head
    w_row = lm_head.weight[token_id].detach().float().cpu()
    direction = J.T @ w_row
    norm = float(torch.linalg.norm(direction))
    if norm < 1e-8:
        return direction.numpy().astype(np.float32)
    return (direction / norm).numpy().astype(np.float32)


def token_id_variants(tokenizer: Any, text: str) -> list[int]:
    """Collect token ids for a value string (full + stripped numeric)."""
    ids: set[int] = set()
    for piece in (text, text.strip(), re.sub(r"[^0-9a-zA-Z.]+", "", text)):
        if not piece:
            continue
        enc = tokenizer.encode(piece, add_special_tokens=False)
        for tid in enc:
            ids.add(int(tid))
        if enc:
            ids.add(int(enc[-1]))
    return sorted(ids)


def coordinate_swap_delta(
    h_source: np.ndarray,
    h_target: np.ndarray,
    V: np.ndarray,
) -> np.ndarray:
    """Cross-instance J-space coordinate transfer: h_target + V(c_source - c_target)."""
    v = np.asarray(V, dtype=np.float64)
    hs = np.asarray(h_source, dtype=np.float64).reshape(-1)
    ht = np.asarray(h_target, dtype=np.float64).reshape(-1)
    if v.ndim != 2 or v.shape[0] != hs.shape[0]:
        raise ValueError(f"V shape {v.shape} incompatible with h dim {hs.shape[0]}")
    coeffs_s, _, _, _ = np.linalg.lstsq(v, hs, rcond=None)
    coeffs_t, _, _, _ = np.linalg.lstsq(v, ht, rcond=None)
    return (v @ (coeffs_s - coeffs_t)).astype(np.float32)


def pairwise_token_swap_delta(
    h: np.ndarray,
    v_source: np.ndarray,
    v_target: np.ndarray,
) -> np.ndarray:
    """Two-direction coordinate swap per paper: h + V(σ(c) - c), V = [v_s, v_t]."""
    vs = np.asarray(v_source, dtype=np.float64).reshape(-1, 1)
    vt = np.asarray(v_target, dtype=np.float64).reshape(-1, 1)
    V = np.concatenate([vs, vt], axis=1)
    h_vec = np.asarray(h, dtype=np.float64).reshape(-1)
    c, _, _, _ = np.linalg.lstsq(V, h_vec, rcond=None)
    if c.size < 2:
        c = np.pad(c, (0, 2 - c.size))
    swapped = np.array([c[1], c[0]])
    return (V @ (swapped - c)).astype(np.float32)


def orthogonalize_columns(U: np.ndarray) -> np.ndarray:
    """QR orthonormalize columns of U."""
    u = np.asarray(U, dtype=np.float32)
    if u.ndim != 2 or u.shape[1] == 0:
        return u
    q, _ = np.linalg.qr(u, mode="reduced")
    return q[:, : u.shape[1]].astype(np.float32)
