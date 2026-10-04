#!/usr/bin/env python3
"""RQ3 mitigation figures: PM-rate vs retention tradeoffs (paper palette, RQ2 chrome)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("${PHANTOM_MERGE_ROOT}")
E2E_SHOP = ROOT / "results/e2e_mitigation/method_comparison_summary.json"
PROBE_SHOP = ROOT / "results/prism_l_plus/method_comparison_summary.json"
DETECTOR = ROOT / "results/agr/DETECTOR_COMPARISON_SUMMARY.json"
ANCHOR = ROOT / "results/rq3/ANCHOR_BASELINE_MATRIX.json"
TAU3_E2E = ROOT / "results/tau3/mitigation/e2e"
TAU3_PROBE = ROOT / "results/tau3/mitigation/probe_gated"
OUT = ROOT / "results/rq3/figures"

C_PRIMARY = "#546CAA"
C_MID = "#878CC3"
C_LIGHT = "#BBB3DA"
C_PINK = "#E9C0CE"
C_PEACH = "#FDD2A8"
C_CORAL = "#EEA386"
C_ROSE = "#BB7D72"
C_ERROR_GRAY = "#9A9A9A"

FONT_TICK = 16.5
FONT_LABEL = 18.0
FONT_LEGEND = 13.5
FONT_ANNOT = 13.5
FONT_TICK_FIGC_Y = 10.0
FONT_TICK_TRADE = 22.0
FONT_LABEL_TRADE = 24.0
FONT_SUPX_TRADE = 26.7
FONT_LEGEND_TRADE = 24.0
MARKER_SIZE = 118
DPI = 300
FILL_ALPHA = 0.65

# Axis (i) Native-Pipeline Comparison — §4.1 E2E (method_id → paper label)
E2E_METHODS: list[tuple[str, str]] = [
    ("b1_rarr_e2e", "RARR"),
    ("b2_cove_e2e", "CoVe"),
    ("b3_copy_e2e", "Copy"),
    ("track_k_e2e", "FAD"),
    ("prism_l_plus_b_e2e", "FAC+RARR"),
    ("prism_l_plus_a_e2e", "FAC"),
]

# Axis (ii) Shared-Gate Mitigation — §4.2 probe-gated
PROBE_METHODS: list[tuple[str, str]] = [
    ("b1_rarr_official", "RARR"),
    ("b2_cove_official", "CoVe"),
    ("b3_copy", "Copy"),
    ("prism_l_plus_a", "FAC"),
    ("track_k", "FAD"),
    ("prism_l_plus_b", "FAC+RARR"),
]

# Swapped vs naive lit/ours pairing: RARR↔FAC hues, CoVe↔FAD, Copy↔FAC+RARR
METHOD_COLOR: dict[str, str] = {
    "b1_rarr_e2e": C_PINK,
    "b2_cove_e2e": C_MID,
    "b3_copy_e2e": C_PRIMARY,
    "track_k_e2e": C_CORAL,
    "prism_l_plus_a_e2e": C_ROSE,
    "prism_l_plus_b_e2e": C_PEACH,
    "b1_rarr_official": C_PINK,
    "b2_cove_official": C_MID,
    "b3_copy": C_PRIMARY,
    "track_k": C_CORAL,
    "prism_l_plus_a": C_ROSE,
    "prism_l_plus_b": C_PEACH,
}

STAR_METHODS = frozenset(
    {"track_k", "track_k_e2e", "prism_l_plus_a", "prism_l_plus_a_e2e"}
)

# Legend colors keyed by paper label (fig B matches fig A).
COLOR_BY_LABEL: dict[str, str] = {lab: METHOD_COLOR[mid] for mid, lab in E2E_METHODS}
SCATTER_S_SQUARE = 1035
SCATTER_S_STAR = 1890

# 1×4 panels (retail re-enabled for layout review)
DOMAIN_PANELS: list[tuple[str, str]] = [
    ("shopping", "Shopping"),
    ("telecom", "Telecom"),
    ("airline", "Airline"),
    ("retail", "Retail"),
]

LEGEND_MARKER_SIZE = 26
LEGEND_MARKER_SIZE_STAR = 45
MARKER_MARGIN_FRAC = 0.38
JITTER_RADIUS_FRAC = 0.155
EDGE_MARKER_PAD_FRAC = 0.14
# Extra data-axis pad from scatter area (points² → ~diameter in axis fraction).
MARKER_LIMIT_PAD_FRAC = 0.055 * (SCATTER_S_STAR / 840.0) ** 0.5

DETECTOR_ARMS: list[tuple[str, str]] = [
    ("bindsurprise_prism_l_plus_a", "AGR-slot"),
    ("agr_rho_prism_l_plus_a", "AGR-value"),
    ("agr_probe_prism_l_plus_a", "Representational"),
]

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
        fig.savefig(OUT / f"{stem}.{ext}", bbox_inches="tight", pad_inches=0.02, dpi=DPI)
    plt.close(fig)


def _spines_clean(ax: plt.Axes, *, tick: float | None = None) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=tick if tick is not None else FONT_TICK)


def _method_color(method_id: str) -> str:
    return METHOD_COLOR[method_id]


def _method_marker(method_id: str) -> str:
    return "*" if method_id in STAR_METHODS else "s"


def _legend_tie_rank(method_id: str) -> int:
    """When harm ties on Shopping e2e: FAD before FAC (pre-swap ability order)."""
    if method_id in ("track_k", "track_k_e2e"):
        return 0
    if method_id in ("prism_l_plus_a", "prism_l_plus_a_e2e"):
        return 1
    return 2


def _swap_fac_fad_legend(specs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Paper fig A/B: FAD immediately left of FAC in the legend."""
    labs = [lab for _, lab in specs]
    if "FAC" not in labs or "FAD" not in labs:
        return specs
    i_fac = labs.index("FAC")
    i_fad = labs.index("FAD")
    out = list(specs)
    out[i_fac], out[i_fad] = out[i_fad], out[i_fac]
    return out


def _legend_specs_e2e_shopping() -> list[tuple[str, str]]:
    """Canonical fig A legend: Shopping harm sort (e2e) then FAC↔FAD swap."""
    mmap = _methods_map_for_domain("shopping", "e2e")
    ranked: list[tuple[float, str, str]] = []
    for mid, lab in E2E_METHODS:
        pm, cb_ret, _, _ = _metrics(mmap[mid])
        harm = pm + (1.0 - cb_ret)
        ranked.append((harm, mid, lab))
    ranked.sort(key=lambda t: (-t[0], _legend_tie_rank(t[1]), t[1]))
    ordered = [(mid, lab) for _, mid, lab in ranked]
    return _swap_fac_fad_legend(ordered)


_CANONICAL_TRADEOFF_LEGEND: list[tuple[str, str]] | None = None


def _canonical_tradeoff_legend() -> list[tuple[str, str]]:
    global _CANONICAL_TRADEOFF_LEGEND
    if _CANONICAL_TRADEOFF_LEGEND is None:
        _CANONICAL_TRADEOFF_LEGEND = _legend_specs_e2e_shopping()
    return _CANONICAL_TRADEOFF_LEGEND


def _legend_specs_for_tradeoff(specs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Fig A defines legend order; fig B reuses the same label order and colors."""
    by_label = {lab: mid for mid, lab in specs}
    return [(by_label[lab], lab) for _, lab in _canonical_tradeoff_legend() if lab in by_label]


def _metrics(block: dict[str, Any]) -> tuple[float, float, float | None, float | None]:
    ac = block.get("audit_conservative") or block.get("audit_optimistic") or {}
    pm = float(ac.get("gated_pm_rate", 0.0))
    cb_ret = float(ac.get("cb_retention_rate", 0.0))
    lo = ac.get("boot_ci95_lo")
    hi = ac.get("boot_ci95_hi")
    return pm, cb_ret, (float(lo) if lo is not None else None), (float(hi) if hi is not None else None)


def _audit_block(block: dict[str, Any]) -> dict[str, Any]:
    return block.get("audit_conservative") or block.get("audit_optimistic") or {}


def _one_minus_pm_ci(ac: dict[str, Any]) -> tuple[float | None, float | None]:
    lo, hi = ac.get("boot_ci95_lo"), ac.get("boot_ci95_hi")
    if lo is None or hi is None:
        return None, None
    return 1.0 - float(hi), 1.0 - float(lo)


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    denom = 1.0 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = (z / denom) * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return float(max(0.0, center - margin)), float(min(1.0, center + margin))


def _lookup_pm_boot_ci(pm_rate: float) -> tuple[float | None, float | None]:
    """Trajectory bootstrap CI for gated PM (B=500) from mitigation summaries."""
    for path in (E2E_SHOP, PROBE_SHOP):
        if not path.is_file():
            continue
        blob = json.loads(path.read_text())
        methods = (blob.get("results_by_tau") or {}).get("aggressive", {}).get("methods")
        if not methods:
            methods = blob.get("methods") or {}
        for block in methods.values():
            ac = _audit_block(block)
            if abs(float(ac.get("gated_pm_rate", -1.0)) - pm_rate) < 1e-9:
                lo, hi = ac.get("boot_ci95_lo"), ac.get("boot_ci95_hi")
                if lo is not None and hi is not None:
                    return float(lo), float(hi)
    return None, None


def _cb_retention_ci(ac: dict[str, Any]) -> tuple[float | None, float | None]:
    k = ac.get("cb_claims_retained")
    n = ac.get("cb_claims_total_baseline")
    if k is None or n is None:
        return None, None
    return _wilson_ci(int(k), int(n))


def _errorbar_caps(
    ax: plt.Axes,
    xpos: np.ndarray,
    vals: list[float],
    lo: list[float | None],
    hi: list[float | None],
) -> None:
    yerr_lo: list[float] = []
    yerr_hi: list[float] = []
    for v, lo_i, hi_i in zip(vals, lo, hi):
        if lo_i is None or hi_i is None:
            yerr_lo.append(0.0)
            yerr_hi.append(0.0)
        else:
            yerr_lo.append(max(0.0, v - lo_i))
            yerr_hi.append(max(0.0, hi_i - v))
    yerr = np.array([yerr_lo, yerr_hi])
    ax.errorbar(
        xpos,
        vals,
        yerr=yerr,
        fmt="none",
        ecolor=C_ERROR_GRAY,
        elinewidth=0.95,
        capsize=3.2,
        capthick=0.95,
        zorder=4,
    )


def _load_shopping_e2e() -> dict[str, dict[str, Any]]:
    data = json.loads(E2E_SHOP.read_text())
    return data["results_by_tau"]["aggressive"]["methods"]


def _load_shopping_probe() -> dict[str, dict[str, Any]]:
    data = json.loads(PROBE_SHOP.read_text())
    return data["results_by_tau"]["aggressive"]["methods"]


def _load_tau3_methods(path: Path, mode: str) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text())
    rbt = data["results_by_tau"]
    if mode == "e2e":
        key = "indomain_agr_slot" if "indomain_agr_slot" in rbt else next(iter(rbt))
    else:
        key = "probe_gated" if "probe_gated" in rbt else next(iter(rbt))
    return rbt[key]["methods"]


def _collect_points(
    methods_map: dict[str, dict[str, Any]],
    specs: list[tuple[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mid, label in specs:
        block = methods_map[mid]
        pm, cb_ret, lo, hi = _metrics(block)
        rows.append(
            {
                "id": mid,
                "label": label,
                "pm": pm,
                "cb_ret": cb_ret,
                "cb_drop": 1.0 - cb_ret,
                "pm_lo": lo,
                "pm_hi": hi,
                "color": COLOR_BY_LABEL[label],
                "marker": _method_marker(mid),
            }
        )
    return rows


def _methods_map_for_domain(domain_key: str, mode: str) -> dict[str, dict[str, Any]]:
    if domain_key == "shopping":
        return _load_shopping_e2e() if mode == "e2e" else _load_shopping_probe()
    base = TAU3_E2E if mode == "e2e" else TAU3_PROBE
    return _load_tau3_methods(base / domain_key / "method_comparison_summary.json", mode)


def _collect_by_domain(specs: list[tuple[str, str]], mode: str) -> list[list[dict[str, Any]]]:
    panels: list[list[dict[str, Any]]] = []
    for dom_key, dom_label in DOMAIN_PANELS:
        mmap = _methods_map_for_domain(dom_key, mode)
        pts = _collect_points(mmap, specs)
        for row in pts:
            row["domain"] = dom_key
            row["domain_label"] = dom_label
        panels.append(pts)
    return panels


def _pad_axis(lo: float, hi: float, *, min_span: float) -> tuple[float, float]:
    if hi < lo:
        lo, hi = hi, lo
    if hi - lo < 1e-9:
        mid = (lo + hi) / 2.0
        lo, hi = mid - min_span / 2.0, mid + min_span / 2.0
    span = hi - lo
    pad = max(span * 0.2, min_span * 0.25)
    return max(0.0, lo - pad), min(1.0, hi + pad)


def _jitter_overlaps(points: list[dict[str, Any]], xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
    """Spread identical (cb_drop, pm) tuples in a small ring (data coords)."""
    from collections import defaultdict

    groups: dict[tuple[float, float], list[dict[str, Any]]] = defaultdict(list)
    for p in points:
        key = (round(p["cb_drop"], 5), round(p["pm"], 5))
        groups[key].append(p)
    xspan = max(xlim[1] - xlim[0], 1e-6)
    yspan = max(ylim[1] - ylim[0], 1e-6)
    radius_x = JITTER_RADIUS_FRAC * xspan
    radius_y = JITTER_RADIUS_FRAC * yspan
    for group in groups.values():
        n = len(group)
        if n == 1:
            p = group[0]
            p["plot_x"], p["plot_y"] = p["cb_drop"], p["pm"]
            continue
        ring = 1.0 + 0.28 * max(0, n - 1)
        for k, p in enumerate(group):
            ang = (2.0 * np.pi * k) / n
            p["plot_x"] = p["cb_drop"] + ring * radius_x * np.cos(ang)
            p["plot_y"] = p["pm"] + ring * radius_y * np.sin(ang)


def _axis_limits_panel(points: list[dict[str, Any]]) -> tuple[tuple[float, float], tuple[float, float]]:
    xs = [p["cb_drop"] for p in points]
    ys = [p["pm"] for p in points]
    for p in points:
        if p.get("pm_lo") is not None:
            ys.append(float(p["pm_lo"]))
        if p.get("pm_hi") is not None:
            ys.append(float(p["pm_hi"]))
    xlim = _pad_axis(min(xs), max(xs), min_span=0.02)
    ylim = _pad_axis(min(ys), max(ys), min_span=0.02)
    return xlim, ylim


def _limits_for_display(points: list[dict[str, Any]]) -> tuple[tuple[float, float], tuple[float, float]]:
    """Limits that include jitter + extra margin so large markers are not clipped."""
    base_x, base_y = _axis_limits_panel(points)
    _jitter_overlaps(points, base_x, base_y)
    xs = [float(p.get("plot_x", p["cb_drop"])) for p in points]
    ys = [float(p.get("plot_y", p["pm"])) for p in points]
    xlim = _pad_axis(min(xs), max(xs), min_span=0.02)
    ylim = _pad_axis(min(ys), max(ys), min_span=0.02)
    xspan = xlim[1] - xlim[0]
    yspan = ylim[1] - ylim[0]
    xbuf = max(xspan * MARKER_MARGIN_FRAC, xspan * EDGE_MARKER_PAD_FRAC)
    ybuf = max(yspan * MARKER_MARGIN_FRAC, yspan * EDGE_MARKER_PAD_FRAC)
    lo_x, hi_x = xlim[0] - xbuf, xlim[1] + xbuf
    lo_y, hi_y = ylim[0] - ybuf, ylim[1] + ybuf
    if min(xs) <= xlim[0] + 1e-5:
        lo_x -= xspan * EDGE_MARKER_PAD_FRAC
    if max(xs) >= xlim[1] - 1e-5:
        hi_x += xspan * EDGE_MARKER_PAD_FRAC
    if min(ys) <= ylim[0] + 1e-5:
        lo_y -= yspan * EDGE_MARKER_PAD_FRAC
    if max(ys) >= ylim[1] - 1e-5:
        hi_y += yspan * EDGE_MARKER_PAD_FRAC
    mpad_x = xspan * MARKER_LIMIT_PAD_FRAC
    mpad_y = yspan * MARKER_LIMIT_PAD_FRAC
    lo_x -= mpad_x
    hi_x += mpad_x
    lo_y -= mpad_y
    hi_y += mpad_y
    return (max(-0.06, lo_x), min(1.10, hi_x)), (max(-0.04, lo_y), min(1.18, hi_y))


def _align_tradeoff_figure_left(fig: plt.Figure, ax_left: plt.Axes) -> None:
    """Snap figure subplot area so the canvas left edge sits at the PM-rate ylabel."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    parts = [ax_left.yaxis.label.get_window_extent(renderer)]
    for t in ax_left.get_yticklabels():
        parts.append(t.get_window_extent(renderer))
    bbox = mpl.transforms.Bbox.union(parts).transformed(fig.transFigure.inverted())
    fig.subplots_adjust(left=max(0.035, bbox.x0 - 0.006))


def _tradeoff_legend_handles(legend_specs: list[tuple[str, str]]) -> list[mpl.lines.Line2D]:
    handles: list[mpl.lines.Line2D] = []
    for mid, lab in legend_specs:
        mk = _method_marker(mid)
        handles.append(
            mpl.lines.Line2D(
                [0],
                [0],
                linestyle="none",
                marker=mk,
                color="w",
                markerfacecolor=COLOR_BY_LABEL[lab],
                markersize=LEGEND_MARKER_SIZE_STAR if mk == "*" else LEGEND_MARKER_SIZE,
                alpha=FILL_ALPHA,
                markeredgecolor="white",
                markeredgewidth=0.55,
                label=lab,
            )
        )
    return handles


def _style_tradeoff_ax(ax: plt.Axes, *, domain_label: str, show_ylabel: bool) -> None:
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5, min_n_ticks=3))
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5, min_n_ticks=3))
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1.0, decimals=0))
    _spines_clean(ax, tick=FONT_TICK_TRADE)
    ax.set_title(
        domain_label,
        fontsize=FONT_LABEL_TRADE,
        loc="center",
        transform=ax.transAxes,
        y=0.94,
        va="top",
        pad=0,
    )
    ax.set_xlabel("")
    if show_ylabel:
        ax.set_ylabel("PM rate", fontsize=FONT_LABEL_TRADE)
    else:
        ax.set_ylabel("")


def _plot_four_domain_tradeoff(stem: str, specs: list[tuple[str, str]], mode: str) -> None:
    panels = _collect_by_domain(specs, mode)
    n = len(DOMAIN_PANELS)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 5.6), sharex=False, sharey=False)
    if n == 1:
        axes = [axes]
    for i, (ax, pts, (_, dom_label)) in enumerate(zip(axes, panels, DOMAIN_PANELS)):
        xlim, ylim = _limits_for_display(pts)
        squares = [p for p in pts if p["marker"] != "*"]
        stars = [p for p in pts if p["marker"] == "*"]
        for p in squares:
            ax.scatter(
                p.get("plot_x", p["cb_drop"]),
                p.get("plot_y", p["pm"]),
                c=p["color"],
                marker=p["marker"],
                s=SCATTER_S_SQUARE,
                alpha=FILL_ALPHA,
                edgecolors="white",
                linewidths=0.85,
                zorder=2,
                clip_on=True,
            )
        for p in stars:
            ax.scatter(
                p.get("plot_x", p["cb_drop"]),
                p.get("plot_y", p["pm"]),
                c=p["color"],
                marker="*",
                s=SCATTER_S_STAR,
                alpha=FILL_ALPHA,
                edgecolors="white",
                linewidths=0.85,
                zorder=6,
                clip_on=True,
            )
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.margins(x=0.01, y=0.02)
        _style_tradeoff_ax(ax, domain_label=dom_label, show_ylabel=(i == 0))

    legend_specs = _legend_specs_for_tradeoff(specs)
    method_handles = _tradeoff_legend_handles(legend_specs)
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.25, top=0.90, wspace=0.26)
    _align_tradeoff_figure_left(fig, axes[0])
    fig.canvas.draw()
    fig.legend(
        handles=method_handles,
        loc="lower left",
        bbox_to_anchor=(0.04, 0.004, 0.94, 0.065),
        mode="expand",
        ncol=6,
        fontsize=FONT_LEGEND_TRADE,
        frameon=False,
        borderaxespad=0.0,
        columnspacing=0.55,
        handletextpad=0.38,
        handlelength=1.1,
    )
    fig.canvas.draw()
    pos0 = axes[0].get_position()
    posn = axes[-1].get_position()
    x_center = 0.5 * (pos0.x0 + posn.x1)
    axes_bottom = min(ax.get_position().y0 for ax in axes)
    legend_top = 0.004 + 0.065
    xlab_y = 0.5 * (legend_top + axes_bottom) - 0.012
    fig.text(
        x_center,
        xlab_y,
        r"$1 - \mathrm{CB\ retention}$",
        ha="center",
        va="center",
        fontsize=FONT_SUPX_TRADE,
    )
    _save(fig, stem)


def plot_fig_a_e2e_four_domains() -> None:
    _plot_four_domain_tradeoff("figA_mitigation_e2e_four_domains", E2E_METHODS, "e2e")


def plot_fig_b_shared_gate_four_domains() -> None:
    _plot_four_domain_tradeoff("figB_mitigation_shared_gate_four_domains", PROBE_METHODS, "probe")


def plot_fig_c_detector_ablation() -> None:
    data = json.loads(DETECTOR.read_text())
    arms = data["e2e_rewrite"]["arms"]
    pm_rates: list[float] = []
    retentions: list[float] = []
    pm_ci_lo: list[float | None] = []
    pm_ci_hi: list[float | None] = []
    ret_ci_lo: list[float | None] = []
    ret_ci_hi: list[float | None] = []
    labels: list[str] = []
    for arm_id, disp in DETECTOR_ARMS:
        block = arms[arm_id]
        ac = _audit_block(block)
        pm, ret, _, _ = _metrics(block)
        one_pm = 1.0 - pm
        pm_rates.append(one_pm)
        retentions.append(ret)
        ac_pm = dict(ac)
        if ac_pm.get("boot_ci95_lo") is None:
            blo, bhi = _lookup_pm_boot_ci(pm)
            if blo is not None:
                ac_pm["boot_ci95_lo"] = blo
                ac_pm["boot_ci95_hi"] = bhi
        lo_pm, hi_pm = _one_minus_pm_ci(ac_pm)
        pm_ci_lo.append(lo_pm)
        pm_ci_hi.append(hi_pm)
        lo_cb, hi_cb = _cb_retention_ci(ac)
        ret_ci_lo.append(lo_cb)
        ret_ci_hi.append(hi_cb)
        labels.append(disp)
    x = np.arange(len(labels))
    w = 0.36
    fig, ax1 = plt.subplots(figsize=(8.2, 4.2))
    b1 = ax1.bar(
        x - w / 2,
        pm_rates,
        width=w,
        color=C_PRIMARY,
        alpha=FILL_ALPHA,
        edgecolor="white",
        linewidth=0.6,
        label=r"$1 - \mathrm{PM\ rate}$",
        zorder=2,
    )
    ax1.set_ylabel(r"$1 - \mathrm{PM\ rate}$", fontsize=FONT_LABEL)
    ax1.set_ylim(0, max(pm_rates) * 1.35 + 0.02)
    ax1.yaxis.set_major_formatter(
        mpl.ticker.FuncFormatter(lambda y, _: f"{int(round(y * 100))}%")
    )
    ax2 = ax1.twinx()
    b2 = ax2.bar(
        x + w / 2,
        retentions,
        width=w,
        color=C_PEACH,
        alpha=FILL_ALPHA,
        edgecolor="white",
        linewidth=0.6,
        label="CB retention",
        zorder=2,
    )
    ax2.set_ylabel("CB retention", fontsize=FONT_LABEL)
    ax2.set_ylim(0, max(retentions) * 1.25 + 0.05)
    ax2.yaxis.set_major_formatter(
        mpl.ticker.FuncFormatter(lambda y, _: f"{int(round(y * 100))}%")
    )
    ax2.spines["top"].set_visible(False)
    _spines_clean(ax1, tick=FONT_TICK_FIGC_Y)
    ax1.tick_params(axis="y", labelsize=FONT_TICK_FIGC_Y)
    ax2.tick_params(axis="y", labelsize=FONT_TICK_FIGC_Y)
    ax1.set_xlim(-0.55, len(labels) - 1 + 0.55)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=FONT_LABEL, ha="center")
    ax1.tick_params(axis="x", labelsize=FONT_LABEL, pad=10)
    _errorbar_caps(ax1, x - w / 2, pm_rates, pm_ci_lo, pm_ci_hi)
    _errorbar_caps(ax2, x + w / 2, retentions, ret_ci_lo, ret_ci_hi)
    for bar, val in zip(b1, pm_rates):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.004,
            f"{int(round(val * 100))}%",
            ha="center",
            va="bottom",
            fontsize=FONT_ANNOT,
        )
    for bar, val in zip(b2, retentions):
        y_pad = 0.007 if val >= 0.70 else 0.02
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + y_pad,
            f"{int(round(val * 100))}%",
            ha="center",
            va="bottom",
            fontsize=FONT_ANNOT,
        )
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    leg = ax1.legend(
        h1 + h2,
        l1 + l2,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.97),
        ncol=2,
        fontsize=FONT_LEGEND - 1,
        frameon=True,
        handlelength=1.8,
        columnspacing=1.4,
        borderpad=0.35,
    )
    leg.get_frame().set_boxstyle("round,pad=0.35,rounding_size=0.8")
    fig.subplots_adjust(left=0.12, right=0.88, top=0.96, bottom=0.20)
    _save(fig, "figC_detector_head_ablation")


def plot_fig_d_anchor() -> None:
    """Anchor verifiers on Shopping held-out test (= Line-A / agr_split test, n=274 claims)."""
    data = json.loads(ANCHOR.read_text())
    llm = data["line_a_test_llm"]
    probe = data["methods"][-1]
    specs = [
        ("B1_anchor_only", "Anchor-only LLM", C_PRIMARY),
        ("B2_source_agnostic", "Source-agnostic LLM", C_ROSE),
        ("cite_or_drop_anchor", "Cite-or-drop LLM", C_PEACH),
        ("NLI_deberta_anchor", "DeBERTa-NLI", C_CORAL),
    ]
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    for key, label, col in specs:
        row = llm[key]
        x, y = float(row["cb_false_drop_rate"]), float(row["pm_recall"])
        ax.scatter(x, y, s=MARKER_SIZE, c=col, alpha=FILL_ALPHA, edgecolors="white", linewidths=0.6, zorder=3)
        ax.annotate(label, (x, y), xytext=(6, 5), textcoords="offset points", fontsize=FONT_ANNOT - 0.5)
    px, py = float(probe["cb_false_drop"]), float(probe["pm_recall"])
    ax.scatter(px, py, s=MARKER_SIZE, c=C_MID, alpha=FILL_ALPHA, edgecolors="white", linewidths=0.6, zorder=3)
    ax.annotate(
        "AGR-slot gate @τ=0.05",
        (px, py),
        xytext=(6, 5),
        textcoords="offset points",
        fontsize=FONT_ANNOT - 0.5,
    )
    ax.set_xlabel("CB false-drop rate", fontsize=FONT_LABEL)
    ax.set_ylabel("PM recall", fontsize=FONT_LABEL)
    ax.set_title("Shopping held-out test (n=274 claims)", fontsize=FONT_LABEL - 1.5, pad=8)
    ax.set_xlim(-0.02, 0.52)
    ax.set_ylim(0.32, 1.05)
    _spines_clean(ax)
    fig.subplots_adjust(left=0.14, right=0.97, top=0.92, bottom=0.14)
    _save(fig, "figD_anchor_validation_recall_falsedrop")


def main() -> None:
    _style_rc()
    plot_fig_a_e2e_four_domains()
    plot_fig_b_shared_gate_four_domains()
    plot_fig_c_detector_ablation()
    plot_fig_d_anchor()
    print(f"Wrote figures under {OUT}")


if __name__ == "__main__":
    main()
