"""Qualitative image grids for multi-attack / multi-frame comparison.

v2 changes:
  attack_comparison_grid — draws boxes in every cell, uses 4 specific frames,
  DFR annotation per row, consistent 640×640 image size across columns.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.eval.metrics import FrameDetections
from src.viz.detection_overlay import draw_detections
from src.viz.style import ATTACK_LABELS, ATTACK_ORDER, PALETTE


def _hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


def _load_bgr_lb(path: Path, size: tuple[int, int] = (640, 640)) -> Optional[np.ndarray]:
    """Load and letterbox-resize image to *size* (W, H). Returns None if missing."""
    bgr = cv2.imread(str(path))
    if bgr is None:
        return None
    if bgr.shape[:2] != (size[1], size[0]):
        from src.viz.letterbox import letterbox_image  # noqa: PLC0415
        bgr, _, _ = letterbox_image(bgr, target=(size[1], size[0]))
    return bgr


def _find_img(directory: Path, stem: str) -> Optional[Path]:
    for ext in (".jpg", ".jpeg", ".png", ".bmp"):
        p = directory / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def qualitative_grid(
    images: list[np.ndarray],
    titles: list[str],
    n_cols: int = 4,
    img_size: tuple[int, int] | None = None,
    suptitle: str = "",
) -> plt.Figure:
    """Arrange a list of RGB images in a grid with per-cell titles."""
    if len(images) != len(titles):
        raise ValueError(f"len(images)={len(images)} != len(titles)={len(titles)}")
    n = len(images)
    if n == 0:
        raise ValueError("images list is empty")

    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3.0 * n_rows))
    axes_flat = np.array(axes).flatten()

    for idx, (img, title) in enumerate(zip(images, titles)):
        ax = axes_flat[idx]
        if img_size is not None:
            img = cv2.resize(img, img_size, interpolation=cv2.INTER_LINEAR)
        ax.imshow(img)
        ax.set_title(title, fontsize=8)
        ax.axis("off")

    for idx in range(n, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    if suptitle:
        fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    return fig


def attack_comparison_grid(
    clean_dir: Path,
    adv_dirs: dict[str, Path],
    stems: list[str],
    clean_dets: dict[str, FrameDetections] | None = None,
    adv_dets_dict: dict[str, dict[str, FrameDetections]] | None = None,
    per_frame_metrics: dict[str, dict[str, dict]] | None = None,
    img_size: tuple[int, int] = (640, 640),
    class_names: dict[int, str] | None = None,
) -> plt.Figure:
    """Qualitative comparison grid: rows = frames, columns = clean + attacks.

    v2: draws detection boxes in every cell; uses per-frame DFR annotation
    below each row; letterboxes clean images so all cells have *img_size*.

    Args:
        clean_dir:          Directory containing clean images.
        adv_dirs:           Dict attack_name -> adversarial image directory.
        stems:              List of image stems (e.g. ["img00001", "img00020"]).
        clean_dets:         Dict stem -> FrameDetections for clean.
        adv_dets_dict:      Dict attack -> Dict stem -> FrameDetections.
        per_frame_metrics:  Dict attack -> Dict stem -> dict with 'n_clean',
                            'n_adv', 'dfr_prop'. Used for row annotations.
        img_size:           Target (W, H) for all cells.
        class_names:        Optional int->str class label mapping.

    Returns:
        Matplotlib Figure.
    """
    attacks = [a for a in ATTACK_ORDER if a in adv_dirs]
    col_labels = ["Clean"] + [ATTACK_LABELS.get(a, a.upper()) for a in attacks]
    n_rows = len(stems)
    n_cols = len(col_labels)

    clean_color = _hex_to_bgr(PALETTE["clean"])

    # Extra height for row annotation text
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.8 * n_cols, 2.7 * n_rows),
        gridspec_kw={"hspace": 0.35, "wspace": 0.05},
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for row, stem in enumerate(stems):
        # -- Clean column --
        clean_path = _find_img(clean_dir, stem)
        ax_c = axes[row, 0]
        if clean_path is not None:
            bgr = _load_bgr_lb(clean_path, size=img_size)
            if bgr is not None and clean_dets and stem in clean_dets:
                bgr = draw_detections(bgr, clean_dets[stem], color=clean_color,
                                      show_score=False, class_names=class_names)
            if bgr is not None:
                ax_c.imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        ax_c.axis("off")
        if row == 0:
            ax_c.set_title("Clean", fontsize=10, fontweight="bold")
        ax_c.set_ylabel(stem, fontsize=10, rotation=0, labelpad=50,
                        ha="right", va="center")

        # -- Attack columns --
        for col, atk in enumerate(attacks, start=1):
            adv_path = _find_img(adv_dirs[atk], stem)
            ax = axes[row, col]
            if adv_path is not None:
                bgr = _load_bgr_lb(adv_path, size=img_size)
                if bgr is None:
                    bgr = np.full((*img_size[::-1], 3), 114, dtype=np.uint8)
                if adv_dets_dict and atk in adv_dets_dict:
                    fd = adv_dets_dict[atk].get(stem)
                    if fd is not None and len(fd.boxes) > 0:
                        atk_color = _hex_to_bgr(PALETTE.get(atk, "#999999"))
                        bgr = draw_detections(bgr, fd, color=atk_color,
                                              show_score=False, class_names=class_names)
                    elif fd is not None and len(fd.boxes) == 0:
                        # Darken and add text for zero-detection frames
                        bgr = (bgr.astype(np.float32) * 0.4).clip(0, 255).astype(np.uint8)
                        hh, ww = bgr.shape[:2]
                        cv2.putText(bgr, "n_adv=0", (ww // 2 - 45, hh // 2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                ax.imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            ax.axis("off")
            if row == 0:
                ax.set_title(ATTACK_LABELS.get(atk, atk.upper()), fontsize=10, fontweight="bold")

        # -- Per-row DFR annotation --
        if per_frame_metrics:
            ref_clean = (clean_dets or {}).get(stem)
            n_clean = len(ref_clean.boxes) if ref_clean else "?"
            parts = [f"Clean: {n_clean}"]
            for atk in attacks:
                pf = per_frame_metrics.get(atk, {}).get(stem, {})
                n_adv = pf.get("n_adv", "?")
                dfr = pf.get("dfr_prop")
                dfr_str = f"{dfr:+.2f}" if dfr is not None else "?"
                parts.append(f"{ATTACK_LABELS.get(atk, atk)}: {n_adv} (DFR={dfr_str})")
            ann = "  |  ".join(parts)
            # Place below the row using figure-space y (approximate)
            # Use the last column's axis for the annotation
            axes[row, -1].annotate(
                ann, xy=(1.0, -0.12), xycoords="axes fraction",
                ha="right", va="top", fontsize=6.5, color="#333333",
            )

    fig.suptitle("Qualitative comparison: clean vs adversarial frames with detections",
                 fontsize=11, y=1.01)
    return fig
