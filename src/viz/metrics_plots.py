"""Metric-based figures: distributions, scatter, bar charts, Pareto.

v2 changes (figures v2 pass):
  f01 — explicit mean lines + CI spans + full-vanishing annotation
  f02 — jitter to fix overplotting, region count labels, outside legend
  f03 — split DFR / ASR panels, 95% CI error bars
  f04 — corrected axis labels, CI error bars on x, no misleading annotation
"""
from __future__ import annotations

from typing import Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.eval.bootstrap import bootstrap_ci
from src.viz.style import ATTACK_LABELS, ATTACK_ORDER, PALETTE


# ---------------------------------------------------------------------------
# f01: per-frame DFR distribution
# ---------------------------------------------------------------------------


def per_frame_dfr_distribution(
    per_image_data: dict[str, list[dict]],
    metric: str = "dfr_prop",
) -> plt.Figure:
    """Violin + strip plot of per-frame DFR with explicit mean lines and CI spans.

    v2: mean line drawn explicitly (not median); 95% CI shown as a vertical
    span; LATENT full-vanishing frames annotated.
    """
    attacks = [a for a in ATTACK_ORDER if a in per_image_data]
    all_vals: list[list[float]] = []
    for atk in attacks:
        frames = [f for f in per_image_data[atk] if f.get("n_clean", 0) > 0]
        if metric == "dfr_prop":
            vals = [1.0 - f["n_adv"] / max(f["n_clean"], 1) for f in frames]
        else:
            vals = [1.0 if f["n_adv"] == 0 else 0.0 for f in frames]
        all_vals.append(vals)

    fig, ax = plt.subplots(figsize=(7, 5))
    positions = list(range(1, len(attacks) + 1))

    parts = ax.violinplot(
        all_vals, positions=positions, showmedians=False,
        showextrema=True, widths=0.6,
    )
    for pc, atk in zip(parts["bodies"], attacks):
        pc.set_facecolor(PALETTE.get(atk, "#999999"))
        pc.set_alpha(0.45)
    for part_name in ("cbars", "cmins", "cmaxes"):
        parts[part_name].set_edgecolor("#666666")
        parts[part_name].set_linewidth(0.8)

    rng = np.random.default_rng(0)
    for pos, (vals, atk) in enumerate(zip(all_vals, attacks), start=1):
        color = PALETTE.get(atk, "#999999")

        # Strip plot (jittered)
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(
            [pos + j for j in jitter], vals,
            color=color, s=16, alpha=0.8, zorder=3,
        )

        # Explicit mean line
        mean_v = float(np.mean(vals))
        ax.hlines(mean_v, pos - 0.28, pos + 0.28, colors=color, linewidths=2.5, zorder=5)

        # Bootstrap CI span
        if len(vals) > 1:
            _, ci_lo, ci_hi = bootstrap_ci(np.array(vals), n_boot=1000, seed=42)
            ax.vlines(pos, ci_lo, ci_hi, colors=color, linewidths=1.2,
                      linestyles="-", alpha=0.55, zorder=4)

        # Annotate full-vanishing cluster (DFR=1) for LATENT
        if atk == "latent" and metric == "dfr_prop":
            n_full = sum(1 for v in vals if abs(v - 1.0) < 1e-9)
            if n_full > 0:
                ax.annotate(
                    f"{n_full} frames:\nfull vanishing",
                    xy=(pos + 0.15, 1.0),
                    xytext=(pos + 0.55, 0.92),
                    fontsize=7.5, color=color,
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                   lw=0.8, mutation_scale=8),
                    ha="left",
                )

    ax.axhline(0.0, color="#888888", linewidth=0.8, linestyle="--", alpha=0.6)
    # Shade FP inflation region
    y_lo, y_hi = ax.get_ylim()
    ax.axhspan(min(y_lo, -0.3), 0.0, alpha=0.05, color="red")
    ax.text(
        0.98, 0.02, "FP inflation region",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=7.5, color="#cc3333", alpha=0.75,
    )
    ax.set_xticks(positions)
    ax.set_xticklabels([ATTACK_LABELS.get(a, a.upper()) for a in attacks])
    ax.set_ylabel("Per-frame DFR (unclipped)" if metric == "dfr_prop" else "DFR_binary (0/1)")
    ax.set_title("Per-frame Detection Failure Rate distribution\n"
                 "(horizontal bars = mean; vertical lines = 95% CI)")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# f02: n_clean vs n_adv scatter
# ---------------------------------------------------------------------------


def n_clean_vs_n_adv_scatter(
    per_image_data: dict[str, list[dict]],
) -> plt.Figure:
    """Scatter n_clean vs n_adv with jitter to reduce overplotting.

    v2: seed=42 jitter, alpha=0.6, s=60, legend outside axes, per-attack
    region count annotations.
    """
    attacks = [a for a in ATTACK_ORDER if a in per_image_data]
    rng = np.random.default_rng(42)

    all_n = []
    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    for atk in attacks:
        xs = np.array([f["n_clean"] for f in per_image_data[atk]], dtype=float)
        ys = np.array([f["n_adv"]   for f in per_image_data[atk]], dtype=float)
        all_n += list(xs) + list(ys)
        jx = rng.uniform(-0.15, 0.15, size=len(xs))
        jy = rng.uniform(-0.15, 0.15, size=len(ys))
        ax.scatter(
            xs + jx, ys + jy,
            label=ATTACK_LABELS.get(atk, atk.upper()),
            color=PALETTE.get(atk, "#999"), s=60, alpha=0.6, zorder=3,
        )

        # Count annotations per attack
        n_below = int(np.sum(ys < xs))
        n_on    = int(np.sum(ys == xs))
        n_above = int(np.sum(ys > xs))
        atk_lbl = ATTACK_LABELS.get(atk, atk.upper())
        # Print inline; full breakdown shown in table below figure
        ax.text(
            0.02, 0.98 - attacks.index(atk) * 0.10,
            f"{atk_lbl}: ↓{n_below} ={n_on} ↑{n_above}",
            transform=ax.transAxes, fontsize=7.5,
            color=PALETTE.get(atk, "#999"), va="top",
        )

    lim = max(all_n) + 1.5 if all_n else 12
    ax.plot([0, lim], [0, lim], "--", color="#555", linewidth=0.9, alpha=0.6, zorder=2)
    ax.fill_between([0, lim], [0, lim], [lim, lim], alpha=0.05, color="red")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)

    ax.set_xlabel("n_clean (detections in clean frame)")
    ax.set_ylabel("n_adv (detections after attack)")
    ax.set_title("n_clean vs n_adv per frame\n(↑ FP inflation | ↓ suppression | = unchanged)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# f03: metric bar chart with CI error bars, split DFR / ASR panels
# ---------------------------------------------------------------------------


def metric_bar_chart(
    summary_data: dict[str, dict[str, Any]],
    metrics: list[tuple[str, str]] | None = None,
) -> plt.Figure:
    """Split bar chart: DFR panel (left) + ASR panel (right) with 95% CI bars.

    v2: two subplots for DFR/ASR groups; error bars with capsize=4;
    zero reference line on DFR panel; DFR_loose shown faded as legacy.
    """
    attacks = [a for a in ATTACK_ORDER if a in summary_data]
    n_atk = len(attacks)
    bar_w = 0.65 / n_atk

    # DFR metrics: (value_key, ci_lo_key, ci_hi_key, label)
    dfr_group = [
        ("dfr_prop", "DFR_prop_lo", "DFR_prop_hi", "DFR prop"),
        ("dfr_bin",  "DFR_bin_lo",  "DFR_bin_hi",  "DFR binary"),
        ("DFR_loose", None, None, "DFR loose (legacy)"),
    ]
    asr_group = [
        ("ASR_strict", "ASR_strict_lo", "ASR_strict_hi", "ASR strict"),
        ("ASR_loose",  None, None, "ASR loose"),
    ]

    fig, (ax_dfr, ax_asr) = plt.subplots(1, 2, figsize=(11, 5))

    def _plot_group(ax, group, y_lo, y_hi, title):
        x = np.arange(len(group))
        for i, atk in enumerate(attacks):
            d = summary_data[atk]
            color = PALETTE.get(atk, "#999")
            label = ATTACK_LABELS.get(atk, atk.upper())
            offsets = x + (i - n_atk / 2 + 0.5) * bar_w
            for col, (val_key, lo_key, hi_key, _) in enumerate(group):
                v = d.get(val_key)
                if v is None:
                    continue
                v = float(v)
                legacy = "legacy" in _.lower()
                alpha = 0.35 if legacy else 0.85
                ax.bar(offsets[col], v, width=bar_w, color=color,
                       alpha=alpha, edgecolor="white", linewidth=0.4,
                       label=label if col == 0 else "_")
                if lo_key and hi_key:
                    lo = d.get(lo_key)
                    hi = d.get(hi_key)
                    if lo is not None and hi is not None:
                        yerr = np.array([[v - float(lo)], [float(hi) - v]])
                        ax.errorbar(offsets[col], v, yerr=yerr, fmt="none",
                                    color=color, capsize=4, linewidth=1.2, zorder=5)

        ax.set_xticks(x)
        ax.set_xticklabels([lbl for _, _, _, lbl in group], rotation=10, ha="right")
        ax.set_ylim(y_lo, y_hi)
        ax.set_title(title)
        ax.set_ylabel("Value")

    _plot_group(ax_dfr, dfr_group, -0.15, 0.45,
                "DFR metrics (left panel, 95% CI bars)")
    ax_dfr.axhline(0, color="#888888", linewidth=0.8)
    ax_dfr.text(0.02, 0.04, "FGSM DFR < 0 (FP inflation)",
                transform=ax_dfr.transAxes, fontsize=7.5, color="#888888")

    _plot_group(ax_asr, asr_group, 0.0, 1.05,
                "ASR metrics (right panel, 95% CI bars)")

    # Shared legend (from DFR panel)
    handles, labels = ax_dfr.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=n_atk,
               bbox_to_anchor=(0.5, 1.01), fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# ---------------------------------------------------------------------------
# f04: stealth vs effectiveness (corrected labels + CI bars)
# ---------------------------------------------------------------------------


def stealth_vs_effectiveness_preview(
    summary_data: dict[str, dict[str, Any]],
    x_key: str = "dfr_prop",
    y_key: str = "PSNR_mask",
    x_lo_key: str = "DFR_prop_lo",
    x_hi_key: str = "DFR_prop_hi",
) -> plt.Figure:
    """Scatter: DFR prop (effectiveness) vs PSNR_mask (stealth).

    v2: corrected axis labels, no misleading annotation near LATENT,
    'ideal' annotation in top-right, 95% CI error bars on x-axis, s=200.
    """
    attacks = [a for a in ATTACK_ORDER if a in summary_data]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    for atk in attacks:
        d = summary_data[atk]
        xv = d.get(x_key)
        yv = d.get(y_key)
        if xv is None or yv is None:
            continue
        xv, yv = float(xv), float(yv)
        color = PALETTE.get(atk, "#999")

        # CI on x
        x_lo = d.get(x_lo_key)
        x_hi = d.get(x_hi_key)
        if x_lo is not None and x_hi is not None:
            ax.errorbar(
                xv, yv,
                xerr=[[xv - float(x_lo)], [float(x_hi) - xv]],
                fmt="none", color=color, capsize=5, linewidth=1.4, zorder=3,
            )

        ax.scatter(xv, yv, color=color, s=200, zorder=4)

        # Auto-offset label to avoid overlap
        dx = {"latent": 0.005, "pgd": -0.002, "fgsm": 0.004}.get(atk, 0.004)
        dy = {"latent": 0.5,   "pgd": -0.8,   "fgsm": 0.5}.get(atk, 0.5)
        ax.annotate(
            ATTACK_LABELS.get(atk, atk.upper()),
            (xv, yv),
            xytext=(xv + dx, yv + dy),
            fontsize=11, color=color, fontweight="bold",
        )

    ax.set_xlabel("DFR_strict_proportional  (effectiveness →)", fontsize=11)
    ax.set_ylabel("PSNR_mask in dB  (← noisier  |  ↑ stealthier)", fontsize=11)
    ax.set_title("Stealth–effectiveness trade-off")

    # Top-right ideal annotation
    ax.annotate(
        "^ ideal:\nhigh effectiveness\n+ high stealth",
        xy=(0.96, 0.96), xycoords="axes fraction",
        ha="right", va="top", fontsize=8.5, color="#444444",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.75),
    )

    fig.tight_layout()
    return fig
