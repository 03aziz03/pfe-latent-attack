"""Generate the Phase 2 Pareto curve (f14): DFR vs masked LPIPS.

Reads results/iso_budget/summary.json produced by run_iso_budget.py and
outputs:
    results/figures/png/f14_pareto_dfr_lpips.png  (300 dpi)
    results/figures/pdf/f14_pareto_dfr_lpips.pdf

Plot layout
-----------
    x-axis : mean masked LPIPS  (↓ better, more imperceptible)
    y-axis : mean DFR_strict_proportional  (↑ better, more effective)
    Colors : latent = blue, PGD = orange
    Markers: latent = circle (o), PGD = square (s)
    Lines  : latent points connected by dashed line (ordered by eps)
             PGD points connected by dotted line
    Error bars: 95 % CI on DFR from bootstrap (src/eval/bootstrap.py)
    Annotation: each point labelled with its eps value
                "↗ ideal" in the top-left of the plot

Usage:
    python scripts/generate_pareto.py
    python scripts/generate_pareto.py --summary results/iso_budget/summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.bootstrap import bootstrap_ci
from src.viz.style import setup_publication_style


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def make_pareto_plot(summary: dict, out_stem: str) -> None:
    setup_publication_style()
    fig, ax = plt.subplots(figsize=(7, 5))

    latent_points: list[dict] = []
    pgd_points: list[dict] = []

    for tag, s in summary.items():
        if s.get("mean_dfr_strict_proportional") is None or s.get("mean_lpips") is None:
            print(f"  Skipping {tag}: missing DFR or LPIPS values.")
            continue

        dfr_vals = np.array(s.get("dfr_values", []), dtype=float)
        if len(dfr_vals) >= 2:
            _, ci_lo, ci_hi = bootstrap_ci(dfr_vals)
        else:
            ci_lo = ci_hi = s["mean_dfr_strict_proportional"]

        record = {
            "tag": tag,
            "eps": s["eps"],
            "mean_dfr": s["mean_dfr_strict_proportional"],
            "mean_lpips": s["mean_lpips"],
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
        }
        if s["attack"] == "latent":
            latent_points.append(record)
        else:
            pgd_points.append(record)

    def _plot_group(points: list[dict], color: str, marker: str, linestyle: str,
                    label: str) -> None:
        if not points:
            return
        points_sorted = sorted(points, key=lambda p: p["eps"])
        xs = [p["mean_lpips"] for p in points_sorted]
        ys = [p["mean_dfr"] for p in points_sorted]
        y_lo = [p["mean_dfr"] - p["ci_lo"] for p in points_sorted]
        y_hi = [p["ci_hi"] - p["mean_dfr"] for p in points_sorted]

        ax.errorbar(xs, ys,
                    yerr=[y_lo, y_hi],
                    fmt=marker,
                    color=color,
                    capsize=4,
                    markersize=8,
                    label=label,
                    zorder=3)
        ax.plot(xs, ys, linestyle=linestyle, color=color, alpha=0.6, zorder=2)

        for p in points_sorted:
            eps_label = f"ε={p['eps']:.3g}" if p["eps"] < 1 else f"ε={p['eps']:.2g}"
            ax.annotate(eps_label,
                        xy=(p["mean_lpips"], p["mean_dfr"]),
                        xytext=(5, 5),
                        textcoords="offset points",
                        fontsize=8,
                        color=color)

    _plot_group(latent_points, color="#1f77b4", marker="o", linestyle="--",
                label="Latent attack")
    _plot_group(pgd_points, color="#ff7f0e", marker="s", linestyle=":",
                label="PGD")

    # "ideal" corner annotation (top-left of axes)
    ax.text(0.03, 0.95, "ideal ↗",
            transform=ax.transAxes,
            fontsize=10, ha="left", va="top", color="gray",
            fontfamily="DejaVu Sans")

    ax.set_xlabel("Masked LPIPS (↓ better)", labelpad=6)
    ax.set_ylabel("DFR_strict_proportional (↑ better)", labelpad=6)
    ax.set_title("Iso-budget Pareto: Effectiveness vs Imperceptibility")
    ax.legend(frameon=False, fontsize=9)

    png_dir = Path("results/figures/png")
    pdf_dir = Path("results/figures/pdf")
    png_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    fig.savefig(png_dir / f"{out_stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(pdf_dir / f"{out_stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {png_dir / (out_stem + '.png')}")
    print(f"Saved {pdf_dir / (out_stem + '.pdf')}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate f14 Pareto curve from iso-budget summary JSON."
    )
    ap.add_argument("--summary", default="results/iso_budget/summary.json",
                    help="Path to summary.json from run_iso_budget.py")
    args = ap.parse_args()

    summary_path = Path(args.summary)
    if not summary_path.exists():
        print(f"WARNING: {summary_path} not found. "
              f"Run scripts/run_iso_budget.py first to generate it.")
        print("Generating a placeholder plot with dummy data for verification.")
        summary = {
            "latent_eps0.25": {
                "attack": "latent", "eps": 0.25,
                "mean_dfr_strict_proportional": 0.30, "mean_lpips": 0.05,
                "dfr_values": [0.25, 0.30, 0.35],
            },
            "latent_eps0.5": {
                "attack": "latent", "eps": 0.50,
                "mean_dfr_strict_proportional": 0.55, "mean_lpips": 0.10,
                "dfr_values": [0.50, 0.55, 0.60],
            },
            "latent_eps1.0": {
                "attack": "latent", "eps": 1.00,
                "mean_dfr_strict_proportional": 0.75, "mean_lpips": 0.20,
                "dfr_values": [0.70, 0.75, 0.80],
            },
            "pgd_eps4": {
                "attack": "pgd", "eps": 4 / 255,
                "mean_dfr_strict_proportional": 0.10, "mean_lpips": 0.02,
                "dfr_values": [0.05, 0.10, 0.15],
            },
            "pgd_eps8": {
                "attack": "pgd", "eps": 8 / 255,
                "mean_dfr_strict_proportional": 0.20, "mean_lpips": 0.04,
                "dfr_values": [0.15, 0.20, 0.25],
            },
            "pgd_eps12": {
                "attack": "pgd", "eps": 12 / 255,
                "mean_dfr_strict_proportional": 0.28, "mean_lpips": 0.07,
                "dfr_values": [0.22, 0.28, 0.34],
            },
        }
    else:
        with open(summary_path) as f:
            summary = json.load(f)

    make_pareto_plot(summary, out_stem="f14_pareto_dfr_lpips")


if __name__ == "__main__":
    main()
