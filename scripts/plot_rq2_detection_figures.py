#!/usr/bin/env python3
"""RQ2 detection figures: one PDF+SVG per figure, minimal chrome, paper palette."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

ROOT = Path("${PHANTOM_MERGE_ROOT}")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REG = ROOT / "results/DETECTION_METRICS_REGISTRY.json"
CURVES = ROOT / "results/rq2/curves"
OUT = ROOT / "results/rq2/figures"

# Paper palette (purple–gold–pink; lighter fills for overlays)
C_PRIMARY = "#546CAA"
C_MID = "#878CC3"
C_LIGHT = "#BBB3DA"
C_PINK = "#E9C0CE"
C_PEACH = "#FDD2A8"
C_CORAL = "#EEA386"
C_ROSE = "#BB7D72"
C_PM_FILL = "#878CC3"
C_CB_FILL = "#FDD2A8"
FIG_D_FILL_ALPHA_PM = 0.65
FIG_D_FILL_ALPHA_CB = 0.65

PALETTE_LINES = [C_ROSE, C_CORAL, C_PEACH, C_PINK, C_LIGHT, C_MID, C_PRIMARY]

FONT_TICK = 16.5
FONT_LABEL = 18.0
FONT_LEGEND = 13.5
LINE_W = 1.7
FIG_CURVE = (4.0, 3.85)
DPI = 300


def _style_rc() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.linewidth": 0.9,
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg"):
        fig.savefig(OUT / f"{stem}.{ext}", bbox_inches="tight", pad_inches=0.03, dpi=DPI)
    plt.close(fig)


def _spines_clean(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=FONT_TICK)


def _nudge_xticklabels_down(fig: plt.Figure, ax: plt.Axes, dy_points: float = 10) -> None:
    from matplotlib.transforms import ScaledTranslation

    trans = ScaledTranslation(0, -dy_points / 72.0, fig.dpi_scale_trans)
    for lbl in ax.get_xticklabels():
        lbl.set_transform(lbl.get_transform() + trans)
        lbl.set_clip_on(False)


def _nudge_xticklabels_down(fig: plt.Figure, ax: plt.Axes, dy_points: float = 10) -> None:
    from matplotlib.transforms import ScaledTranslation

    trans = ScaledTranslation(0, -dy_points / 72.0, fig.dpi_scale_trans)
    for lbl in ax.get_xticklabels():
        lbl.set_transform(lbl.get_transform() + trans)
        lbl.set_clip_on(False)


def _blend_purple_gold(t: float) -> tuple[float, float, float]:
    """t in [0,1]: figD PM purple → CB gold."""
    from matplotlib.colors import to_rgb

    a, b = to_rgb(C_PM_FILL), to_rgb(C_CB_FILL)
    t = max(0.0, min(1.0, t))
    return tuple(a[i] * (1 - t) + b[i] * t for i in range(3))


def _legend_rounded(
    ax: plt.Axes,
    *,
    loc: str,
    bbox: tuple[float, float],
    ncol: int = 1,
    in_axes: bool = False,
    data_coords: bool = False,
    compact: bool = False,
) -> None:
    kw: dict = {
        "frameon": True,
        "fancybox": True,
        "fontsize": FONT_LEGEND - 1.0 if compact else FONT_LEGEND,
        "loc": loc,
        "bbox_to_anchor": bbox,
        "ncol": ncol,
        "handlelength": 1.5 if compact else 2.2,
        "handletextpad": 0.35 if compact else 0.6,
        "borderpad": 0.12 if compact else 0.55,
        "labelspacing": 0.15 if compact else 0.32,
    }
    if data_coords:
        kw["bbox_transform"] = ax.transData
        kw["borderaxespad"] = 0.35
    elif in_axes:
        kw["bbox_transform"] = ax.transAxes
        kw["borderaxespad"] = 0.02 if compact else 0.25
    leg = ax.legend(**kw)
    frame = leg.get_frame()
    frame.set_facecolor("white")
    frame.set_edgecolor("#CCCCCC")
    frame.set_alpha(0.96)
    frame.set_linewidth(0.8)
    if compact:
        frame.set_boxstyle("round,pad=0.08,rounding_size=0.5")
    else:
        frame.set_boxstyle("round,pad=0.45,rounding_size=1.2")


def _legend_below(fig: plt.Figure, ax: plt.Axes, ncol: int = 4) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    fig.legend(
        handles,
        labels,
        frameon=False,
        fontsize=FONT_LEGEND,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.06),
        ncol=ncol,
        columnspacing=1.1,
        handletextpad=0.5,
        handlelength=2.0,
    )


def _method_metrics(reg: dict, mid: str) -> dict:
    for m in reg.get("methods") or []:
        if m.get("method_id") == mid:
            return m
    raise KeyError(mid)


# Two-line x labels for readout ablation bars (keeps slanted ticks compact).
_READOUT_XTICK_TWO_LINE: dict[str, str] = {
    "D-L49": "Representational\nreadout",
    "D-JSL": "Functional\nreadout",
    "D-AGR": "AGR\n-value",
    "D-BIND": "AGR\n-slot",
}


def _bar_xtick_label(method_id: str, short_label: str) -> str:
    return _READOUT_XTICK_TWO_LINE.get(method_id, short_label)


def plot_fig_a_auroc(reg: dict) -> None:
    # Main-table baselines + readout ablation variants (word labels); low→high AUROC, purple→gold.
    specs: list[tuple[str, str]] = [
        ("D-RARR-G", "RARR"),
        ("D-FACTSCORE-OBS", "FActScore"),
        ("D-FACTOOL-KBQA", "FacTool"),
        ("D-MBERT-Q", "ModernBERT"),
        ("D-TFIDF", "TF-IDF"),
        ("D-SELFCHECK-NLI", "SelfCheck-NLI"),
        ("D-L49", "Representational readout"),
        ("D-JSL", "Functional readout"),
        ("D-AGR", "AGR-value"),
        ("D-BIND", "AGR-slot"),
    ]
    rows_a: list[tuple[str, str, float, float, float]] = []
    for mid, lab in specs:
        m = _method_metrics(reg, mid)
        boot = m["ranking"]["auroc"]
        rows_a.append((mid, lab, boot["auroc"], boot["ci_low"], boot["ci_high"]))
    rows_a.sort(key=lambda x: x[2])
    n = len(rows_a)
    aurocs = [r[2] for r in rows_a]
    lo = [r[3] for r in rows_a]
    hi = [r[4] for r in rows_a]
    labels = [_bar_xtick_label(r[0], r[1]) for r in rows_a]
    colors = [_blend_purple_gold(i / max(n - 1, 1)) for i in range(n)]
    yerr = np.array([[a - l, h - a] for a, l, h in zip(aurocs, lo, hi)]).T

    fig, ax = plt.subplots(figsize=(11.5, 3.4))
    x = np.arange(len(labels))
    ax.bar(x, aurocs, color=colors, width=0.72, zorder=2, edgecolor="white", linewidth=0.6)
    ax.errorbar(x, aurocs, yerr=yerr, fmt="none", ecolor="#333333", capsize=3, linewidth=1.0, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=FONT_TICK - 2, rotation=28, ha="right")
    ax.set_ylabel("AUROC", fontsize=FONT_LABEL)
    ax.set_ylim(0.45, 1.02)
    _spines_clean(ax)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.96, bottom=0.34)
    _save(fig, "figA1_readout_ablation_auroc")


def plot_fig_a_precision(reg: dict) -> None:
    rm = reg.get("recall_matched_to_probe_at_0.05") or {}
    per = rm.get("per_method") or {}
    variants = [
        ("D-L49", "Representational readout"),
        ("D-JSL", "Functional readout"),
        ("D-AGR", "AGR (value)"),
        ("D-BIND", "AGR (slot)"),
    ]
    rows_p: list[tuple[str, str, float]] = []
    for mid, lab in variants:
        row = per.get(mid) or {}
        rows_p.append((mid, lab, float(row.get("precision") or 0.0)))
    rows_p.sort(key=lambda x: x[2])
    n = len(rows_p)
    precs = [r[2] for r in rows_p]
    labels = [_bar_xtick_label(r[0], r[1]) for r in rows_p]
    colors = [_blend_purple_gold(i / max(n - 1, 1)) for i in range(n)]

    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    x = np.arange(len(labels))
    ax.bar(x, precs, color=colors, width=0.58, zorder=2, edgecolor="white", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=FONT_TICK - 1, rotation=18, ha="right")
    ax.set_ylabel("Precision (matched recall)", fontsize=FONT_LABEL)
    ax.set_ylim(0.55, 0.98)
    _spines_clean(ax)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.96, bottom=0.22)
    _save(fig, "figA2_readout_ablation_precision_matched_recall")


def _load_curve(mid: str) -> tuple[np.ndarray, np.ndarray]:
    path = CURVES / f"{mid}_scores.json"
    data = json.loads(path.read_text())
    recs = data["records"]
    y = np.array([r["gold_label"] for r in recs], dtype=int)
    s = np.array([r["score"] for r in recs], dtype=float)
    return y, s


def _fig_b_line_colors(methods: list[tuple[str, str]]) -> dict[str, str]:
    """Original per-method colors, assignment order reversed (AGR ↔ RARR swap ends)."""
    original: list[str] = []
    for i, (mid, _) in enumerate(methods):
        if mid == "D-AGR-SLOT":
            original.append(C_PRIMARY)
        else:
            original.append(PALETTE_LINES[i % len(PALETTE_LINES)])
    reversed_cols = list(reversed(original))
    return {methods[i][0]: reversed_cols[i] for i in range(len(methods))}


def plot_fig_b_roc() -> None:
    methods = [
        ("D-RARR-G", "RARR"),
        ("D-FACTSCORE-OBS", "FActScore"),
        ("D-FACTOOL-KBQA", "FacTool"),
        ("D-MBERT-Q", "ModernBERT"),
        ("D-TFIDF", "TF-IDF"),
        ("D-SELFCHECK-NLI", "SelfCheck"),
        ("D-AGR-SLOT", "AGR-slot"),
    ]
    line_colors = _fig_b_line_colors(methods)
    fig, ax = plt.subplots(figsize=FIG_CURVE)
    for mid, lab in methods:
        y, s = _load_curve(mid)
        if len(np.unique(y)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y, s)
        col = line_colors[mid]
        z = 4 if mid == "D-AGR-SLOT" else 3
        ax.plot(fpr, tpr, color=col, linewidth=LINE_W, label=lab, zorder=z)
    ax.plot([0, 1], [0, 1], color="#CCCCCC", linewidth=0.9, linestyle="--", zorder=1)
    ax.set_xlabel("False positive rate", fontsize=FONT_LABEL)
    ax.set_ylabel("True positive rate", fontsize=FONT_LABEL)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    _spines_clean(ax)
    # Inside axes: hug lower-right corner (just above x-axis spine).
    _legend_rounded(ax, loc="lower right", bbox=(1.0, 0.028), ncol=1, in_axes=True, compact=True)
    leg = ax.get_legend()
    if leg is not None:
        leg.set_zorder(10)
    fig.subplots_adjust(left=0.18, right=0.98, top=0.98, bottom=0.16)
    _save(fig, "figB_roc_overlay")


def plot_fig_c_pr() -> None:
    methods = [
        ("D-RARR-G", "RARR"),
        ("D-FACTSCORE-OBS", "FActScore"),
        ("D-FACTOOL-KBQA", "FacTool"),
        ("D-MBERT-Q", "ModernBERT"),
        ("D-TFIDF", "TF-IDF"),
        ("D-SELFCHECK-NLI", "SelfCheck-NLI"),
        ("D-AGR-SLOT", "AGR (slot)"),
    ]
    # PR: at recall→1, precision cannot fall below prevalence (156/274≈0.57); not a cropped axis.
    fig, ax = plt.subplots(figsize=FIG_CURVE)
    for i, (mid, lab) in enumerate(methods):
        y, s = _load_curve(mid)
        if y.sum() == 0:
            continue
        prec, rec, _ = precision_recall_curve(y, s)
        order = np.argsort(rec)
        col = C_PRIMARY if mid == "D-AGR-SLOT" else PALETTE_LINES[i % len(PALETTE_LINES)]
        ax.plot(rec[order], prec[order], color=col, linewidth=LINE_W, label=lab, zorder=3)
    ax.set_xlabel("Recall", fontsize=FONT_LABEL)
    ax.set_ylabel("Precision", fontsize=FONT_LABEL)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.0, 1.02)
    _spines_clean(ax)
    ax.legend(frameon=False, fontsize=FONT_LEGEND, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=4)
    fig.subplots_adjust(left=0.16, right=0.98, top=0.98, bottom=0.26)
    _save(fig, "figC_pr_overlay")


def _kde_density(samples: np.ndarray, grid: np.ndarray) -> np.ndarray:
    from scipy.stats import gaussian_kde

    if samples.size < 2:
        return np.zeros_like(grid)
    kde = gaussian_kde(samples)
    return kde(grid)


def plot_fig_d_distribution() -> None:
    y, s = _load_curve("D-AGR-SLOT")
    pm = s[y == 1]
    cb = s[y == 0]
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    lo = min(s.min(), np.percentile(s, 1)) - 0.5
    hi = max(s.max(), np.percentile(s, 99)) + 0.5
    bins = np.linspace(lo, hi, 28)
    grid = np.linspace(lo, hi, 220)
    ax.hist(
        cb,
        bins=bins,
        density=True,
        alpha=FIG_D_FILL_ALPHA_CB,
        color=C_CB_FILL,
        label="CB",
        edgecolor="white",
        linewidth=0.5,
        zorder=2,
    )
    ax.hist(
        pm,
        bins=bins,
        density=True,
        alpha=FIG_D_FILL_ALPHA_PM,
        color=C_PM_FILL,
        label="PM",
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )
    ax.plot(
        grid,
        _kde_density(cb, grid),
        color=C_CB_FILL,
        alpha=1.0,
        solid_capstyle="round",
        linewidth=LINE_W,
        zorder=4,
    )
    ax.plot(
        grid,
        _kde_density(pm, grid),
        color=C_PM_FILL,
        alpha=1.0,
        solid_capstyle="round",
        linewidth=LINE_W,
        zorder=5,
    )
    ax.set_xlabel("AGR-slot score", fontsize=FONT_LABEL)
    ax.set_ylabel("Density", fontsize=FONT_LABEL)
    _spines_clean(ax)
    ax.legend(frameon=False, fontsize=FONT_LEGEND, loc="upper right")
    fig.subplots_adjust(left=0.14, right=0.98, top=0.98, bottom=0.16)
    _save(fig, "figD_agr_score_distribution")


def plot_fig_e_threshold_sweep() -> None:
    path = CURVES / "dev_p_pm_threshold_sweep.json"
    data = json.loads(path.read_text())
    pts = data["points"]
    taus = [p["tau"] for p in pts]
    prec = [p["precision"] for p in pts]
    rec = [p["recall"] for p in pts]
    f1 = [p["f1"] for p in pts]
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    ax.plot(taus, prec, color=C_PM_FILL, linewidth=LINE_W, label="Precision")
    ax.plot(taus, rec, color=C_PINK, linewidth=LINE_W, label="Recall")
    ax.plot(taus, f1, color=C_CB_FILL, linewidth=LINE_W, label="F1")
    ax.axvline(0.05, color="#999999", linewidth=1.0, linestyle="--", zorder=0)
    ax.set_xlabel(r"$\tau$", fontsize=FONT_LABEL)
    ax.set_ylabel("Score", fontsize=FONT_LABEL)
    ax.set_xlim(0, 1)
    ax.set_ylim(0.5, 1.02)
    _spines_clean(ax)
    _legend_rounded(
        ax,
        loc="lower left",
        bbox=(0.0, 0.52),
        ncol=1,
        compact=True,
        data_coords=True,
    )
    leg = ax.get_legend()
    if leg is not None:
        leg.set_zorder(10)
    fig.subplots_adjust(left=0.14, right=0.98, top=0.98, bottom=0.16)
    _save(fig, "figE_threshold_sweep_dev")


_FIGF_METHOD_XTICK: dict[str, str] = {
    "representational_readout": "Representational\nreadout",
    "functional_readout": "Functional\nreadout",
    "agr_value": "AGR-value",
    "agr_slot": "AGR-slot",
}

# Heatmap AUROC anchors (paper palette).
FIG_F_HEAT_VMIN = 0.6
FIG_F_HEAT_VMAX = 1.0
FIG_F_HEAT_STOPS: tuple[tuple[float, str], ...] = (
    (0.6, C_PM_FILL),
    (0.7, C_LIGHT),
    (0.9, C_CB_FILL),
    (1.0, C_CORAL),
)


def _fig_f_heatmap_cmap() -> mpl.colors.LinearSegmentedColormap:
    from matplotlib.colors import to_rgb

    lo, hi = FIG_F_HEAT_VMIN, FIG_F_HEAT_VMAX
    cdict: dict[str, list[tuple[float, float, float]]] = {"red": [], "green": [], "blue": []}
    for v, hex_color in FIG_F_HEAT_STOPS:
        pos = (v - lo) / (hi - lo)
        r, g, b = to_rgb(hex_color)
        for ch, val in zip(("red", "green", "blue"), (r, g, b)):
            cdict[ch].append((pos, val, val))
    return mpl.colors.LinearSegmentedColormap("figf_heat", cdict, N=256)

_SHOPPING_READOUT_IDS: dict[str, str] = {
    "representational_readout": "D-L49",
    "functional_readout": "D-JSL",
    "agr_value": "D-AGR",
    "agr_slot": "D-BIND",
}


def _shopping_readout_auroc(reg: dict) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, mid in _SHOPPING_READOUT_IDS.items():
        m = _method_metrics(reg, mid)
        out[key] = float(m["ranking"]["auroc"]["auroc"])
    return out


def plot_fig_f_heatmap(reg: dict) -> None:
    path = CURVES / "tau3_four_readout_crossdomain.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    method_keys = [m["key"] for m in payload["methods"]]
    tau3 = payload["auroc"]
    shop = _shopping_readout_auroc(reg)
    domain_order = ["shopping", "telecom", "airline", "retail"]
    row_labels = ["Shopping", "Telecom", "Airline", "Retail"]
    col_labels = [_FIGF_METHOD_XTICK.get(k, k) for k in method_keys]
    mat = np.array(
        [
            [
                shop[k] if dom == "shopping" else float(tau3[dom][k])
                for k in method_keys
            ]
            for dom in domain_order
        ],
        dtype=float,
    )
    cmap = _fig_f_heatmap_cmap()
    norm = mpl.colors.Normalize(vmin=FIG_F_HEAT_VMIN, vmax=FIG_F_HEAT_VMAX)
    t = norm(mat)
    rgba = cmap(t)
    rgba[..., 3] = FIG_D_FILL_ALPHA_PM
    fig, ax = plt.subplots(figsize=(5.8, 3.35))
    im = ax.imshow(rgba, aspect="auto")
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(
        col_labels,
        rotation=28,
        ha="center",
        va="top",
        rotation_mode="anchor",
        fontsize=FONT_TICK - 2,
    )
    ax.tick_params(axis="x", pad=8, length=2)
    _nudge_xticklabels_down(fig, ax, dy_points=14)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=FONT_TICK)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            ax.text(
                j,
                i,
                f"{v:.2f}",
                ha="center",
                va="center",
                color="#000000",
                fontsize=FONT_TICK - 1,
            )
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("AUROC", fontsize=FONT_LABEL - 1)
    cbar.ax.tick_params(labelsize=FONT_TICK - 2)
    _spines_clean(ax)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.subplots_adjust(left=0.16, right=0.88, top=0.96, bottom=0.42)
    _save(fig, "figF_crossdomain_diagnostic_heatmap")


def main() -> int:
    _style_rc()
    reg = json.loads(REG.read_text(encoding="utf-8"))
    plot_fig_a_auroc(reg)
    plot_fig_b_roc()
    plot_fig_d_distribution()
    plot_fig_e_threshold_sweep()
    plot_fig_f_heatmap(reg)
    print(f"Wrote figures under {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
