"""Pareto-front analysis and plotting for attack/stealth trade-off curves."""
from __future__ import annotations

from pathlib import Path

import numpy as np


# Keys where higher is better (attack effectiveness).
_ATTACK_KEYS: frozenset[str] = frozenset(
    {"dfr", "asr", "map_drop", "map_drop_50", "map_drop_coco", "conf_drop"}
)


def _dominates(a: dict, b: dict) -> bool:
    """Return True if run *a* dominates run *b*.

    Dominance: *a* is ≥ *b* on every shared metric and strictly > on at least
    one. Direction: higher is better for attack keys; lower is better for
    stealth keys (LPIPS, masked_l2, psnr_mask is reversed → lower PSNR means
    more visible → higher is actually better for stealth, so we treat PSNR as
    an attack-side metric too).
    """
    m_a = a["metrics"]
    m_b = b["metrics"]
    shared = set(m_a) & set(m_b)
    if not shared:
        return False

    strictly_better = False
    for k in shared:
        va, vb = float(m_a[k]), float(m_b[k])
        if k in _ATTACK_KEYS or k.lower() in {"psnr_mask", "psnr"}:
            # higher is better
            if va < vb:
                return False
            if va > vb:
                strictly_better = True
        else:
            # lower is better (stealth: lpips, masked_l2)
            if va > vb:
                return False
            if va < vb:
                strictly_better = True

    return strictly_better


def build_pareto(runs: list[dict]) -> list[dict]:
    """Annotate runs with Pareto dominance flags and sort by primary stealth metric.

    Args:
        runs: List of run dicts, each with keys:
            - 'name'    (str): attack name / variant label.
            - 'budget'  (float): perturbation budget (e.g. ε_z or ε_pixel).
            - 'metrics' (dict): metric name → float value.

    Returns:
        Same list (copies) sorted by stealth metric ascending (less distortion
        first). Each dict gains a 'dominated' (bool) field.
    """
    annotated = [dict(run, dominated=False) for run in runs]

    for i in range(len(annotated)):
        for j in range(len(annotated)):
            if i != j and _dominates(annotated[j], annotated[i]):
                annotated[i]["dominated"] = True
                break

    stealth_key: str | None = None
    for candidate in ("masked_l2", "LPIPS", "lpips", "l2"):
        if any(candidate in r["metrics"] for r in annotated):
            stealth_key = candidate
            break

    if stealth_key:
        annotated.sort(key=lambda r: r["metrics"].get(stealth_key, float("inf")))

    return annotated


def plot_pareto(
    runs: list[dict],
    x_key: str = "LPIPS",
    y_key: str = "DFR",
    save_path: str | Path | None = None,
) -> None:
    """Scatter plot of runs in the (x_key, y_key) metric space.

    Pareto-optimal (non-dominated) runs are shown with filled markers;
    dominated runs use hollow markers. Runs with the same 'name' are
    connected by a dashed line sorted by x_key.

    Args:
        runs:      List of run dicts (same format as build_pareto input).
        x_key:     Metric to plot on the x-axis (stealth metric).
        y_key:     Metric to plot on the y-axis (attack metric).
        save_path: If provided, saves both PNG (150 dpi) and PDF at this path
                   (extension is replaced). The parent directory is created
                   automatically.
    """
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    annotated = build_pareto(runs)
    by_name: dict[str, list[dict]] = {}
    for r in annotated:
        by_name.setdefault(r.get("name", "unknown"), []).append(r)

    fig, ax = plt.subplots(figsize=(6, 5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for idx, (name, group) in enumerate(by_name.items()):
        color = colors[idx % len(colors)]
        xs = [r["metrics"].get(x_key, float("nan")) for r in group]
        ys = [r["metrics"].get(y_key, float("nan")) for r in group]
        dominated = [r.get("dominated", True) for r in group]

        front_x = [x for x, d in zip(xs, dominated) if not d]
        front_y = [y for y, d in zip(ys, dominated) if not d]
        dom_x = [x for x, d in zip(xs, dominated) if d]
        dom_y = [y for y, d in zip(ys, dominated) if d]

        if front_x:
            ax.scatter(front_x, front_y, color=color, marker="o", s=80, label=name, zorder=3)
        if dom_x:
            ax.scatter(
                dom_x, dom_y, color=color, marker="o", s=80,
                facecolors="none", edgecolors=color, zorder=3,
                label=name if not front_x else None,
            )

        paired = sorted(
            [(x, y) for x, y in zip(xs, ys) if np.isfinite(x) and np.isfinite(y)],
            key=lambda t: t[0],
        )
        if len(paired) > 1:
            ax.plot([p[0] for p in paired], [p[1] for p in paired],
                    color=color, lw=1, ls="--", alpha=0.6)

    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.set_title(f"Pareto front: {x_key} vs {y_key}")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
        fig.savefig(save_path.with_suffix(".pdf"), bbox_inches="tight")

    plt.close(fig)
