"""Perturbation heatmaps and difference grids for adversarial images.

v2 changes:
  perturbation_heatmap — 2x2 grid (clean + 3 attacks), shared vmax colorbar
  difference_grid      — per-attack calibrated amplification factor
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.viz.style import ATTACK_LABELS, ATTACK_ORDER


def _load_rgb(path: Path, target_size: tuple[int, int] | None = None) -> np.ndarray:
    """Load image as float32 RGB in [0, 1]; resize to (W, H) if target_size given."""
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    if target_size is not None:
        bgr = cv2.resize(bgr, target_size, interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def perturbation_heatmap(
    clean_path: Path,
    adv_paths: dict[str, Path],
    amplify: float = 5.0,
    mask: np.ndarray | None = None,
) -> plt.Figure:
    """2x2 grid: clean reference (top-left) + one heatmap per attack.

    All heatmaps share a single vmax = max |adv - clean| across all three
    attacks, giving a fair absolute comparison of perturbation magnitude.

    Layout (rows × cols):
        [clean]   [LATENT heatmap]
        [PGD hm]  [FGSM heatmap]

    Args:
        clean_path: Path to the clean image.
        adv_paths:  Dict mapping attack name -> adversarial image path.
        amplify:    Unused (kept for signature compatibility); shared vmax
                    is set automatically from data.
        mask:       Optional (H, W) boolean array; pixels outside = 0.

    Returns:
        Matplotlib Figure (10x8 inches).
    """
    clean_raw = _load_rgb(clean_path)
    h, w = clean_raw.shape[:2]

    attacks_present = [a for a in ATTACK_ORDER if a in adv_paths]

    # Compute differences and find shared vmax
    diffs: dict[str, np.ndarray] = {}
    for atk in attacks_present:
        adv = _load_rgb(adv_paths[atk], target_size=(w, h))
        diff = np.abs(adv - clean_raw).mean(axis=2)  # (H, W)
        if mask is not None:
            diff = diff * mask.astype(np.float32)
        diffs[atk] = diff

    shared_vmax = max(d.max() for d in diffs.values()) if diffs else 1.0

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Top-left: clean reference
    axes[0, 0].imshow(clean_raw)
    axes[0, 0].set_title("Clean reference", fontsize=10)
    axes[0, 0].axis("off")

    # Attack positions in 2×2 grid (excluding top-left)
    grid_pos = [(0, 1), (1, 0), (1, 1)]
    im_last = None
    for pos, atk in zip(grid_pos, attacks_present):
        r, c = pos
        diff = diffs[atk]
        im_last = axes[r, c].imshow(diff, cmap="inferno", vmin=0.0, vmax=shared_vmax)
        lbl = ATTACK_LABELS.get(atk, atk.upper())
        axes[r, c].set_title(f"{lbl}  |adv − clean| per-channel mean", fontsize=10)
        axes[r, c].axis("off")

    # Hide unused panels
    for pos in grid_pos[len(attacks_present):]:
        axes[pos[0], pos[1]].set_visible(False)

    # Single shared colorbar
    if im_last is not None:
        fig.colorbar(im_last, ax=axes.ravel().tolist(), shrink=0.55,
                     pad=0.02, label="Mean per-channel |Δ| (shared vmax)")

    fig.suptitle(
        "Perturbation magnitude heatmaps\n"
        "(shared vmax = max across all attacks — enables direct magnitude comparison)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 0.88, 1])
    return fig


def difference_grid(
    clean_path: Path,
    adv_paths: dict[str, Path],
    amplify: float = 10.0,
) -> plt.Figure:
    """Grid: adv frames (row 0) and calibrated amplified differences (row 1).

    v2: per-attack amplification clipped to [1, 100] so all three diffs
    reach roughly the same dynamic range (target max ≈ 50/max_diff).
    Actual amp_factor annotated in each title.

    Args:
        clean_path: Path to the clean image.
        adv_paths:  Dict mapping attack name -> adversarial image path.
        amplify:    Ignored in v2 (kept for backward compatibility); amp is
                    computed per-attack automatically.

    Returns:
        Matplotlib Figure.
    """
    clean_raw = _load_rgb(clean_path)
    h, w = clean_raw.shape[:2]

    attacks = [a for a in ATTACK_ORDER if a in adv_paths]
    if not attacks:
        attacks = list(adv_paths.keys())

    ncols = max(len(attacks), 1)
    fig, axes = plt.subplots(2, ncols, figsize=(4 * ncols, 8))
    if ncols == 1:
        axes = axes.reshape(2, 1)

    for col, atk in enumerate(attacks):
        adv = _load_rgb(adv_paths[atk], target_size=(w, h))
        diff = np.abs(adv - clean_raw)
        max_diff = float(diff.max())

        amp = float(np.clip(50.0 / max_diff if max_diff > 1e-8 else 1.0, 1.0, 100.0))
        diff_rgb = np.clip(diff * amp, 0.0, 1.0)

        axes[0, col].imshow(adv)
        axes[0, col].set_title(ATTACK_LABELS.get(atk, atk.upper()), fontsize=10)
        axes[0, col].axis("off")

        axes[1, col].imshow(diff_rgb)
        axes[1, col].set_title(f"Diff x{amp:.1f}  (max={max_diff:.4f})", fontsize=10)
        axes[1, col].axis("off")

    stem = Path(clean_path).stem
    fig.suptitle(
        f"Adversarial frames and amplified differences — {stem}\n"
        "Amplification calibrated per attack so peak ≈ same brightness.",
        fontsize=11,
    )
    fig.tight_layout()
    return fig
