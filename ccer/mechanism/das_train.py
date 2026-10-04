"""DAS training via interchange-intervention loss (Geiger et al.; expert_recorrect §3.1)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn.functional as F

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    F = None  # type: ignore
    _HAS_TORCH = False


@dataclass(frozen=True)
class InterchangeTriplet:
    """(base, source, target) activation triple for interchange-intervention loss."""

    h_base: np.ndarray
    h_source: np.ndarray
    h_target: np.ndarray
    kind: str  # cross_pm_to_clean | within_counterfactual | cross_clean_to_pm


def _uniform_vectors(vectors: list[np.ndarray]) -> list[np.ndarray]:
    if not vectors:
        return []
    dim = int(np.asarray(vectors[0]).reshape(-1).shape[0])
    out: list[np.ndarray] = []
    for v in vectors:
        flat = np.asarray(v, dtype=np.float64).reshape(-1)
        if flat.shape[0] == dim:
            out.append(flat)
    return out


def interchange_reconstruct(h_base: np.ndarray, h_source: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Subspace interchange: h_base + U(U^T(h_source - h_base))."""
    delta = h_source - h_base
    proj = u @ (u.T @ delta)
    return h_base + proj


def _pca_init(diff_vectors: list[np.ndarray], rank: int) -> np.ndarray:
    vecs = _uniform_vectors(diff_vectors)
    if not vecs:
        return np.zeros((0, 0), dtype=np.float32)
    x = np.stack(vecs, axis=0)
    x = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    r = min(rank, vt.shape[0])
    q, _ = np.linalg.qr(vt[:r].T)
    return q[:, :r].astype(np.float32)


def collect_interchange_triplets(
    *,
    layer: int,
    position: str,
    track: str = "CEM",
    pool: str = "mechanism_research",
    pm_trajectory_allowlist: set[str] | None = None,
) -> list[InterchangeTriplet]:
    """Build supervised interchange triplets from cross + within activations."""
    from ccer.mechanism.activation_store import activation_path, get_vector, load_activation_npz
    from ccer.mechanism.mechanism_pool import PoolName, build_cem_mechanism_cross_pairs

    pool_name: PoolName = "mechanism_research" if pool == "mechanism_research" else "dev"
    if track.upper() != "CEM":
        from ccer.mechanism.pair_select import select_track_pairs

        selection = select_track_pairs("CAP")
        cross_pairs = selection["pm_clean_cross_pairs"]
        pairs = selection["pairs"]
        pm_key = "cap_trajectory_id"
        cond_swap = "query_value_swap"
    else:
        selection = build_cem_mechanism_cross_pairs(pool=pool_name)
        cross_pairs = selection["pm_clean_cross_pairs"]
        pairs = selection["cem_pairs"]
        pm_key = "pm_trajectory_id"
        cond_swap = "rival_value_swap"

    triplets: list[InterchangeTriplet] = []
    clean_by_pm: dict[str, str] = {
        str(cp[pm_key]): str(cp["clean_trajectory_id"]) for cp in cross_pairs if cp.get(pm_key)
    }

    audit_index: dict[str, Any] = {}
    if pool == "mechanism_research":
        from ccer.mechanism.line_b_protocol import load_pair_audit_index, pair_quality_score

        audit_index = load_pair_audit_index()

    for cp in cross_pairs:
        pm_tid = str(cp.get(pm_key) or "")
        clean_tid = str(cp.get("clean_trajectory_id") or "")
        if not pm_tid or not clean_tid:
            continue
        if pm_trajectory_allowlist is not None and pm_tid not in pm_trajectory_allowlist:
            continue
        if audit_index and pair_quality_score(audit_index.get(pm_tid), cross_pair_row=cp) < 0.35:
            continue
        pm_path = activation_path(pm_tid, "original")
        clean_path = activation_path(clean_tid, "original")
        if not pm_path.is_file() or not clean_path.is_file():
            continue
        h_pm = get_vector(load_activation_npz(pm_path), position=position, layer=layer)
        h_clean = get_vector(load_activation_npz(clean_path), position=position, layer=layer)
        if h_pm is None or h_clean is None:
            continue
        h_pm = np.asarray(h_pm, dtype=np.float64).reshape(-1)
        h_clean = np.asarray(h_clean, dtype=np.float64).reshape(-1)
        triplets.append(InterchangeTriplet(h_pm, h_clean, h_clean, "cross_pm_to_clean"))
        triplets.append(InterchangeTriplet(h_clean, h_pm, h_pm, "cross_clean_to_pm"))

    for pair in pairs:
        tid = str(pair["trajectory_id"])
        if pm_trajectory_allowlist is not None and tid not in pm_trajectory_allowlist:
            continue
        orig_path = activation_path(tid, "original")
        swap_path = activation_path(tid, cond_swap)
        if not orig_path.is_file() or not swap_path.is_file():
            continue
        h_orig = get_vector(load_activation_npz(orig_path), position=position, layer=layer)
        h_swap = get_vector(load_activation_npz(swap_path), position=position, layer=layer)
        if h_orig is None or h_swap is None:
            continue
        h_orig = np.asarray(h_orig, dtype=np.float64).reshape(-1)
        h_swap = np.asarray(h_swap, dtype=np.float64).reshape(-1)
        clean_tid = clean_by_pm.get(tid)
        if clean_tid:
            h_clean = get_vector(
                load_activation_npz(activation_path(clean_tid, "original")),
                position=position,
                layer=layer,
            )
            if h_clean is not None:
                h_clean = np.asarray(h_clean, dtype=np.float64).reshape(-1)
                triplets.append(
                    InterchangeTriplet(h_orig, h_clean, h_swap, "within_counterfactual")
                )
        triplets.append(InterchangeTriplet(h_orig, h_swap, h_swap, "within_counterfactual"))

    return triplets


def _fit_das_torch(
    triplets: list[InterchangeTriplet],
    *,
    rank: int,
    seed: int,
    steps: int,
    lr: float,
    init_u: np.ndarray | None,
    log_every: int = 25,
) -> dict[str, Any]:
    if not triplets:
        return {"error": "no_triplets", "rank": rank}
    dim = int(triplets[0].h_base.shape[0])
    if dim < rank + 2:
        return {"error": "insufficient_dim", "dim": dim, "rank": rank}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    bases = torch.tensor(np.stack([t.h_base for t in triplets]), dtype=dtype, device=device)
    sources = torch.tensor(np.stack([t.h_source for t in triplets]), dtype=dtype, device=device)
    targets = torch.tensor(np.stack([t.h_target for t in triplets]), dtype=dtype, device=device)
    # Up-weight owner-binding cross triplets vs within/cohort noise (Line B fix #1 / #3).
    weights = torch.ones(len(triplets), dtype=dtype, device=device)
    for i, t in enumerate(triplets):
        if t.kind == "cross_pm_to_clean":
            weights[i] = 3.0
        elif t.kind == "cross_clean_to_pm":
            weights[i] = 2.0
        elif t.kind == "within_counterfactual":
            weights[i] = 1.5

    if init_u is not None and init_u.size:
        u0 = torch.tensor(init_u[:, :rank], dtype=dtype, device=device)
    else:
        diff_vecs = [t.h_target - t.h_base for t in triplets]
        u0 = torch.tensor(_pca_init(diff_vecs, rank), dtype=dtype, device=device)
    if u0.shape[1] < rank:
        pad = torch.randn(dim, rank - u0.shape[1], dtype=dtype, device=device) * 0.01
        u0 = torch.cat([u0, pad], dim=1)

    raw = torch.nn.Parameter(u0.clone())
    opt = torch.optim.Adam([raw], lr=lr)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    loss_history: list[float] = []
    best_loss = float("inf")
    best_u = u0.detach().clone()
    log_every = max(1, int(log_every))
    print(
        f"[line_b] das_train rank={rank} triplets={len(triplets)} steps={steps} device={device}",
        flush=True,
    )

    for step in range(steps):
        q, _ = torch.linalg.qr(raw, mode="reduced")
        u = q[:, :rank]
        delta = sources - bases
        proj = (delta @ u) @ u.T
        pred = bases + proj
        recon = F.mse_loss(pred, targets, reduction="none").mean(dim=1)
        # Root #1: behavioral margin — pred should be closer to target than base (flip proxy).
        behavioral_terms: list[torch.Tensor] = []
        for i, t in enumerate(triplets):
            if t.kind not in ("cross_pm_to_clean", "cross_clean_to_pm", "within_counterfactual"):
                continue
            base_err = F.mse_loss(bases[i], targets[i], reduction="mean")
            pred_err = F.mse_loss(pred[i], targets[i], reduction="mean")
            behavioral_terms.append(torch.clamp(pred_err - base_err + 0.02, min=0.0))
        behavioral_loss = (
            torch.stack(behavioral_terms).mean()
            if behavioral_terms
            else torch.tensor(0.0, dtype=dtype, device=device)
        )
        # Behavioral alignment: cross_pm_to_clean deltas should lie in subspace (owner-binding axis).
        align_terms: list[torch.Tensor] = []
        for i, t in enumerate(triplets):
            if t.kind != "cross_pm_to_clean":
                continue
            d_i = delta[i]
            d_norm = torch.linalg.norm(d_i)
            if float(d_norm.item()) < 1e-8:
                continue
            p_i = proj[i]
            p_norm = torch.linalg.norm(p_i)
            if float(p_norm.item()) < 1e-8:
                align_terms.append(torch.tensor(1.0, dtype=dtype, device=device))
            else:
                align_terms.append(1.0 - (p_i @ d_i) / (p_norm * d_norm + 1e-8))
        align_loss = (
            torch.stack(align_terms).mean()
            if align_terms
            else torch.tensor(0.0, dtype=dtype, device=device)
        )
        loss = (weights * recon).mean() + 0.35 * align_loss + 0.5 * behavioral_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        lv = float(loss.item())
        loss_history.append(lv)
        if lv < best_loss:
            best_loss = lv
            with torch.no_grad():
                q_best, _ = torch.linalg.qr(raw, mode="reduced")
                best_u = q_best[:, :rank].detach().clone()
        step_n = step + 1
        if step_n == 1 or step_n % log_every == 0 or step_n == steps:
            pct = 100.0 * step_n / steps
            bar_filled = int(20 * step_n / steps)
            bar = "#" * bar_filled + "-" * (20 - bar_filled)
            print(
                f"[line_b] das_train rank={rank} [{bar}] {step_n}/{steps} ({pct:.0f}%) "
                f"loss={lv:.4f} best={best_loss:.4f}",
                flush=True,
            )

    with torch.no_grad():
        final_q, _ = torch.linalg.qr(raw, mode="reduced")
        u_final = final_q[:, :rank]
        delta = sources - bases
        final_mse = float(F.mse_loss(bases + (delta @ u_final) @ u_final.T, targets).item())

    u_np = best_u.cpu().numpy().astype(np.float32)
    kind_counts: dict[str, int] = {}
    for t in triplets:
        kind_counts[t.kind] = kind_counts.get(t.kind, 0) + 1

    return {
        "U_das": u_np,
        "rank": rank,
        "n_triplets": len(triplets),
        "triplet_kinds": kind_counts,
        "steps": steps,
        "lr": lr,
        "seed": seed,
        "das_method": "interchange_intervention_loss_weighted_owner_align_behavioral_v3",
        "final_interchange_mse": final_mse,
        "best_interchange_mse": best_loss,
        "loss_history_last10": loss_history[-10:],
        "init": "pca" if init_u is None else "pca_provided",
    }


def fit_das_subspace(
    diff_vectors: list[np.ndarray] | None = None,
    *,
    rank: int,
    seed: int = 42,
    steps: int = 500,
    lr: float = 0.02,
    layer: int | None = None,
    position: str | None = None,
    track: str = "CEM",
    pool: str = "mechanism_research",
    triplets: list[InterchangeTriplet] | None = None,
    pm_trajectory_allowlist: set[str] | None = None,
    log_every: int = 25,
) -> dict[str, Any]:
    """
    Train orthonormal subspace U via interchange-intervention reconstruction loss.

    When layer/position are provided, collects triplets from activations automatically.
    Legacy diff_vectors-only call still works but requires layer/position for full DAS.
    """
    if triplets is None and layer is not None and position is not None:
        triplets = collect_interchange_triplets(
            layer=layer,
            position=position,
            track=track,
            pool=pool,
            pm_trajectory_allowlist=pm_trajectory_allowlist,
        )

    if triplets:
        init_u = None
        if diff_vectors:
            init_u = _pca_init(diff_vectors, rank)
        if _HAS_TORCH:
            return _fit_das_torch(
                triplets, rank=rank, seed=seed, steps=steps, lr=lr, init_u=init_u, log_every=log_every
            )
        return _fit_das_numpy_fallback(triplets, rank=rank, init_u=init_u)

    vecs = _uniform_vectors(diff_vectors or [])
    if len(vecs) < rank + 2:
        return {"error": "insufficient_vectors", "n": len(vecs), "rank": rank}
    return {
        "error": "triplets_required_for_das",
        "n_vectors": len(vecs),
        "rank": rank,
        "hint": "Pass layer+position or explicit triplets; diff_vectors alone is insufficient for DAS.",
    }


def _fit_das_numpy_fallback(
    triplets: list[InterchangeTriplet],
    *,
    rank: int,
    init_u: np.ndarray | None,
) -> dict[str, Any]:
    """Gradient-free Procrustes refinement when torch unavailable."""
    u = init_u if init_u is not None else _pca_init(
        [t.h_target - t.h_base for t in triplets], rank
    )
    if u.size == 0:
        return {"error": "pca_init_failed", "rank": rank}
    best_u = u.copy()
    best_mse = float("inf")
    for _ in range(50):
        preds = [interchange_reconstruct(t.h_base, t.h_source, best_u) for t in triplets]
        targets = np.stack([t.h_target for t in triplets])
        mse = float(np.mean(np.sum((np.stack(preds) - targets) ** 2, axis=1)))
        if mse < best_mse:
            best_mse = mse
        grad_vecs = []
        for t, pred in zip(triplets, preds):
            err = pred - t.h_target
            delta = t.h_source - t.h_base
            grad_vecs.append(np.outer(err, delta))
        step = np.mean(grad_vecs, axis=0)
        best_u = best_u - 0.01 * (step @ best_u)
        q, _ = np.linalg.qr(best_u)
        best_u = q[:, :rank].astype(np.float32)
    return {
        "U_das": best_u,
        "rank": rank,
        "n_triplets": len(triplets),
        "das_method": "interchange_intervention_loss_numpy_fallback",
        "final_interchange_mse": best_mse,
        "steps": 50,
    }


def compare_pca_vs_das(
    *,
    u_pca: np.ndarray,
    u_das: np.ndarray,
) -> dict[str, Any]:
    if u_pca.size == 0 or u_das.size == 0:
        return {"comparable": False}
    if u_pca.ndim == 1:
        u_pca = u_pca.reshape(-1, 1)
    if u_das.ndim == 1:
        u_das = u_das.reshape(-1, 1)
    if u_pca.shape[0] != u_das.shape[0]:
        return {"comparable": False, "reason": "dim_mismatch"}
    r = min(u_pca.shape[1], u_das.shape[1])
    a = u_pca[:, :r]
    b = u_das[:, :r]
    sim = float(
        np.mean(
            np.abs(
                np.sum(a * b, axis=0)
                / (np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0) + 1e-8)
            )
        )
    )
    return {"comparable": True, "subspace_alignment_mean_cos": sim, "rank_compared": r}
