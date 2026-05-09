"""Publication-style matplotlib configuration and shared colour palette."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PALETTE: dict[str, str] = {
    "latent": "#2166AC",
    "pgd": "#D6604D",
    "fgsm": "#4DAC26",
    "clean": "#888888",
}

ATTACK_LABELS: dict[str, str] = {
    "latent": "LATENT",
    "pgd": "PGD",
    "fgsm": "FGSM",
    "clean": "Clean",
}

ATTACK_ORDER: list[str] = ["latent", "pgd", "fgsm"]


def setup_publication_style() -> None:
    """Apply rcParams for publication-quality figures (serif, tick-in, no box)."""
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "xtick.minor.size": 2,
            "ytick.minor.size": 2,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "legend.frameon": False,
            "figure.dpi": 100,
            "savefig.dpi": 300,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def save_figure(fig: "plt.Figure", path: Path, dpi: int = 300) -> None:
    """Save *fig* as both PNG (300 dpi) and PDF (vector), then close it.

    Parent directories are created automatically.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
