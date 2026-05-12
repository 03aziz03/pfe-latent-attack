"""Generate ablation comparison tables and figures.

Loads results from:
    results/phase1/      (Phase 1 baseline: masked-L2 + original VAE)
    results/ablation/    (Option 1: LPIPS + original VAE)
                         (Option 2: Option 1 + bilateral post-processing)

Produces:
    results/figures/
        ablation_dfr_bar.png       DFR bar chart across all configs + eps
        ablation_lpips_box.png     LPIPS distribution box plots
        ablation_pareto.png        DFR vs LPIPS Pareto scatter
        ablation_table.md          Markdown table for the report
        ablation_table.csv         CSV for further analysis

Usage
-----
    python scripts/ablation_table.py \\
        --phase1_dir  results/iso_budget \\
        --ablation_dir results/ablation  \\
        --output_dir  results/figures
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

LATENT_EPS = [0.25, 0.50, 1.00]
PGD_LABELS = {"pgd_eps4": 4/255, "pgd_eps8": 8/255, "pgd_eps12": 12/255}


def load_json(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def aggregate(records: list[dict]) -> dict:
    dfr_vals   = [r["dfr"]   for r in records if r.get("n_clean", 1) > 0]
    lpips_vals = [r["lpips"] for r in records
                  if r.get("lpips") is not None and r.get("n_clean", 1) > 0]
    n = len(dfr_vals)
    if n == 0:
        return {"mean_dfr": None, "se_dfr": None,
                "mean_lpips": None, "se_lpips": None,
                "dfr_pos": 0, "dfr_neg": 0, "n": 0,
                "dfr_vals": [], "lpips_vals": []}

    def se(v):
        if len(v) < 2: return 0.0
        m = sum(v) / len(v)
        return math.sqrt(sum((x - m)**2 for x in v) / (len(v)-1) / len(v))

    return {
        "mean_dfr":   sum(dfr_vals) / n,
        "se_dfr":     se(dfr_vals),
        "mean_lpips": sum(lpips_vals) / len(lpips_vals) if lpips_vals else None,
        "se_lpips":   se(lpips_vals),
        "dfr_pos":    sum(1 for v in dfr_vals if v > 0),
        "dfr_neg":    sum(1 for v in dfr_vals if v < 0),
        "n":          n,
        "dfr_vals":   dfr_vals,
        "lpips_vals": lpips_vals,
    }


def collect_all(phase1_dir: Path, ablation_dir: Path) -> dict[str, dict]:
    """Return a flat dict of {label: aggregate_stats}."""
    data: dict[str, dict] = {}

    # Phase 1 latent (masked-L2)
    for eps in LATENT_EPS:
        tag = f"latent_eps{eps}"
        p   = phase1_dir / f"{tag}.json"
        if p.exists():
            recs = load_json(p)
            agg  = aggregate(recs)
            agg["dfr_raw"]   = [r["dfr"]   for r in recs]
            agg["lpips_raw"] = [r.get("lpips") for r in recs]
            data[f"Phase1_L2 eps={eps}"] = agg

    # Option 1 latent (LPIPS, no postproc)
    for eps in LATENT_EPS:
        tag = f"option1_latent_eps{eps}"
        p   = ablation_dir / f"{tag}.json"
        if p.exists():
            recs = load_json(p)
            agg  = aggregate(recs)
            agg["dfr_raw"]   = [r["dfr"]   for r in recs]
            agg["lpips_raw"] = [r.get("lpips") for r in recs]
            data[f"Opt1_LPIPS eps={eps}"] = agg

    # Option 2 latent (LPIPS + bilateral)
    for eps in LATENT_EPS:
        tag = f"option2_latent_eps{eps}"
        p   = ablation_dir / f"{tag}.json"
        if p.exists():
            recs = load_json(p)
            agg  = aggregate(recs)
            agg["dfr_raw"]   = [r["dfr"]   for r in recs]
            agg["lpips_raw"] = [r.get("lpips") for r in recs]
            data[f"Opt2_Bilateral eps={eps}"] = agg

    # PGD baselines
    for tag, eps in PGD_LABELS.items():
        for src in [phase1_dir, ablation_dir]:
            p = src / f"{tag}.json"
            if p.exists():
                recs = load_json(p)
                agg  = aggregate(recs)
                agg["dfr_raw"]   = [r["dfr"]   for r in recs]
                agg["lpips_raw"] = [r.get("lpips") for r in recs]
                data[f"PGD {round(eps*255)}/255"] = agg
                break   # use first found

    return data


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

COLORS = {
    "Phase1": "#4878d0",
    "Opt1":   "#ee854a",
    "Opt2":   "#6acc65",
    "PGD":    "#d65f5f",
}

def _color(label: str) -> str:
    if "Phase1" in label:  return COLORS["Phase1"]
    if "Opt1"   in label:  return COLORS["Opt1"]
    if "Opt2"   in label:  return COLORS["Opt2"]
    if "PGD"    in label:  return COLORS["PGD"]
    return "#999999"


def plot_dfr_bar(data: dict, out_path: Path) -> None:
    labels = list(data.keys())
    means  = [data[l]["mean_dfr"] or 0 for l in labels]
    errors = [data[l]["se_dfr"]   or 0 for l in labels]
    colors = [_color(l) for l in labels]

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.9), 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=errors, capsize=4,
                  color=colors, alpha=0.85, edgecolor="black", linewidth=0.6)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Mean DFR (proportional)")
    ax.set_title("DFR Comparison — Phase 1 vs Option 1 vs Option 2 vs PGD")

    # legend
    patches = [
        mpatches.Patch(color=COLORS["Phase1"], label="Phase 1 (masked-L2)"),
        mpatches.Patch(color=COLORS["Opt1"],   label="Option 1 (LPIPS)"),
        mpatches.Patch(color=COLORS["Opt2"],   label="Option 2 (LPIPS + bilateral)"),
        mpatches.Patch(color=COLORS["PGD"],    label="PGD pixel baseline"),
    ]
    ax.legend(handles=patches, fontsize=8, loc="upper left")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  DFR bar chart → {out_path}")


def plot_lpips_box(data: dict, out_path: Path) -> None:
    labels = list(data.keys())
    vals   = [data[l]["lpips_vals"] for l in labels]
    # Drop empty series
    pairs  = [(l, v) for l, v in zip(labels, vals) if v]
    if not pairs:
        print("  No LPIPS data — skipping box plot.")
        return
    labels, vals = zip(*pairs)
    colors = [_color(l) for l in labels]

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.9), 5))
    bps = ax.boxplot(vals, patch_artist=True, notch=False, showfliers=True)
    for patch, c in zip(bps["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Masked LPIPS (AlexNet)")
    ax.set_title("Perceptual Distortion — LPIPS Distribution per Config")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  LPIPS box plot  → {out_path}")


def plot_pareto(data: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))

    for label, stats in data.items():
        dfr  = stats.get("mean_dfr")
        lpips = stats.get("mean_lpips")
        if dfr is None or lpips is None:
            continue
        se_dfr   = stats.get("se_dfr", 0) or 0
        se_lpips = stats.get("se_lpips", 0) or 0
        color = _color(label)
        ax.errorbar(lpips, dfr,
                    xerr=se_lpips, yerr=se_dfr,
                    fmt="o", color=color, markersize=7,
                    capsize=3, linewidth=1.2, alpha=0.85)
        ax.annotate(label, (lpips, dfr),
                    textcoords="offset points", xytext=(5, 3),
                    fontsize=6.5, color=color)

    ax.axhline(0, color="gray", linewidth=0.7, linestyle="--")
    ax.set_xlabel("Mean Masked LPIPS ↓ (lower = more imperceptible)")
    ax.set_ylabel("Mean DFR ↑ (higher = more effective)")
    ax.set_title("Pareto Front — Attack Effectiveness vs Perceptual Quality")

    patches = [
        mpatches.Patch(color=COLORS["Phase1"], label="Phase 1 (masked-L2)"),
        mpatches.Patch(color=COLORS["Opt1"],   label="Option 1 (LPIPS)"),
        mpatches.Patch(color=COLORS["Opt2"],   label="Option 2 (LPIPS + bilateral)"),
        mpatches.Patch(color=COLORS["PGD"],    label="PGD pixel baseline"),
    ]
    ax.legend(handles=patches, fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Pareto scatter  → {out_path}")


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def write_markdown(data: dict, out_path: Path) -> None:
    lines = [
        "# Ablation Study — DFR and LPIPS Comparison\n",
        "| Config | Mean DFR | ±SE | Mean LPIPS | ±SE | Pos frames | Neg frames |",
        "|--------|----------|-----|------------|-----|-----------|-----------|",
    ]
    for label, s in data.items():
        dfr_s    = f"{s['mean_dfr']:+.4f}"   if s["mean_dfr"]   is not None else "N/A"
        se_dfr_s = f"{s['se_dfr']:.4f}"      if s["se_dfr"]     is not None else "N/A"
        lp_s     = f"{s['mean_lpips']:.4f}"  if s["mean_lpips"] is not None else "N/A"
        se_lp_s  = f"{s['se_lpips']:.4f}"    if s["se_lpips"]   is not None else "N/A"
        lines.append(
            f"| {label} | {dfr_s} | {se_dfr_s} | {lp_s} | {se_lp_s} "
            f"| {s['dfr_pos']} | {s['dfr_neg']} |"
        )
    out_path.write_text("\n".join(lines) + "\n")
    print(f"  Markdown table  → {out_path}")


def write_csv(data: dict, out_path: Path) -> None:
    fieldnames = ["config", "mean_dfr", "se_dfr", "mean_lpips",
                  "se_lpips", "dfr_pos", "dfr_neg", "n_frames"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for label, s in data.items():
            w.writerow({
                "config":     label,
                "mean_dfr":   round(s["mean_dfr"],   6) if s["mean_dfr"]   is not None else "",
                "se_dfr":     round(s["se_dfr"],     6) if s["se_dfr"]     is not None else "",
                "mean_lpips": round(s["mean_lpips"], 6) if s["mean_lpips"] is not None else "",
                "se_lpips":   round(s["se_lpips"],   6) if s["se_lpips"]   is not None else "",
                "dfr_pos":    s["dfr_pos"],
                "dfr_neg":    s["dfr_neg"],
                "n_frames":   s["n"],
            })
    print(f"  CSV table       → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate ablation tables and figures."
    )
    ap.add_argument("--phase1_dir",   default="results/iso_budget",
                    help="Directory with Phase 1 JSON results")
    ap.add_argument("--ablation_dir", default="results/ablation",
                    help="Directory with Option 1 + Option 2 JSON results")
    ap.add_argument("--output_dir",   default="results/figures",
                    help="Where to save figures and tables")
    args = ap.parse_args()

    phase1_dir   = Path(args.phase1_dir)
    ablation_dir = Path(args.ablation_dir)
    out_dir      = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading results …")
    data = collect_all(phase1_dir, ablation_dir)
    if not data:
        print("ERROR: No result JSON files found. "
              "Run run_iso_budget.py and run_ablation_sweep.py first.")
        return

    print(f"Found {len(data)} configs: {list(data.keys())}\n")

    print("Generating figures …")
    plot_dfr_bar(data,   out_dir / "ablation_dfr_bar.png")
    plot_lpips_box(data, out_dir / "ablation_lpips_box.png")
    plot_pareto(data,    out_dir / "ablation_pareto.png")

    print("Writing tables …")
    write_markdown(data, out_dir / "ablation_table.md")
    write_csv(data,      out_dir / "ablation_table.csv")

    # Console summary
    print("\n" + "="*70)
    print(f"{'Config':<30} {'DFR':>8} {'±SE':>7} {'LPIPS':>8}")
    print("-"*70)
    for label, s in data.items():
        dfr_s = f"{s['mean_dfr']:+.4f}" if s["mean_dfr"] is not None else "  N/A"
        se_s  = f"{s['se_dfr']:.4f}"    if s["se_dfr"]   is not None else " N/A"
        lp_s  = f"{s['mean_lpips']:.4f}" if s["mean_lpips"] is not None else "  N/A"
        print(f"{label:<30} {dfr_s:>8} {se_s:>7} {lp_s:>8}")
    print("="*70)


if __name__ == "__main__":
    main()
