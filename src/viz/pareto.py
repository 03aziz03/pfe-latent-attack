"""Pareto-frontier visualization for attack trade-off analysis."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np

from src.viz.style import ATTACK_LABELS, PALETTE, save_figure


def _is_dominated(run: dict, others: list[dict], x_key: str, y_key: str) -> bool:
    """Return True if *run* is dominated (worse on both axes) by any other run."""
    xv = run[x_key]
    yv = run[y_key]
    for o in others:
        if o is run:
            continue
        if o[x_key] >= xv and o[y_key] >= yv and (o[x_key] > xv or o[y_key] > yv):
            return True
    return False


def plot_pareto(
    runs: list[dict[str, Any]],
    x_key: str,
    y_key: str,
    save_path: Path | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    title: str = "Stealth vs Effectiveness",
) -> plt.Figure:
    """Scatter plot with Pareto-optimal runs highlighted.

    Args:
        runs:      Each dict must have *x_key*, *y_key*, and optionally *name*
                   (string label) and *attack* (key into PALETTE/ATTACK_LABELS).
        x_key:     Metric name for the x-axis (higher = more desirable assumed).
        y_key:     Metric name for the y-axis (higher = more desirable assumed).
        save_path: If given, save PNG + PDF at this path stem (without extension).
        x_label:   Override x-axis label (default: *x_key*).
        y_label:   Override y-axis label (default: *y_key*).
        title:     Figure title.

    Returns:
        Matplotlib Figure (closed when save_path is provided).
    """
    if not runs:
        raise ValueError("runs list is empty")

    dominated = [_is_dominated(r, runs, x_key, y_key) for r in runs]

    fig, ax = plt.subplots(figsize=(6, 5))

    for run, dom in zip(runs, dominated):
        atk = run.get("attack", "")
        color = PALETTE.get(atk, "#333333")
        label_str = run.get("name", ATTACK_LABELS.get(atk, atk.upper()))
        marker = "o" if not dom else "x"
        alpha = 1.0 if not dom else 0.4
        ax.scatter(
            run[x_key],
            run[y_key],
            color=color,
            marker=marker,
            s=80,
            alpha=alpha,
            zorder=3,
        )
        ax.annotate(
            label_str,
            (run[x_key], run[y_key]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=9,
            color=color,
        )

    # Draw Pareto frontier
    pareto_runs = [r for r, d in zip(runs, dominated) if not d]
    if len(pareto_runs) >= 2:
        pareto_sorted = sorted(pareto_runs, key=lambda r: r[x_key])
        px = [r[x_key] for r in pareto_sorted]
        py = [r[y_key] for r in pareto_sorted]
        ax.plot(px, py, "--", color="#333333", linewidth=1.0, alpha=0.6, zorder=2)

    ax.set_xlabel(x_label if x_label else x_key, fontsize=11)
    ax.set_ylabel(y_label if y_label else y_key, fontsize=11)
    ax.set_title(title, fontsize=12)

    non_pareto_patch = plt.Line2D([0], [0], marker="x", color="#777777", linestyle="None",
                                  markersize=7, label="Dominated")
    pareto_patch = plt.Line2D([0], [0], marker="o", color="#333333", linestyle="None",
                              markersize=7, label="Pareto-optimal")
    ax.legend(handles=[pareto_patch, non_pareto_patch], loc="best")

    fig.tight_layout()

    if save_path is not None:
        save_figure(fig, Path(save_path))

    return fig
