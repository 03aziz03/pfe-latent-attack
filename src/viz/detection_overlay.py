"""Draw detection boxes on images and produce clean-vs-adversarial overlays.

v2 changes:
  overlay_clean_vs_adv — draws clean boxes on the clean panel (green);
  handles n_adv=0 with a text annotation; adds a footer count line;
  letterboxes the clean image to 640x640 so both panels are the same size.
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
from src.viz.style import PALETTE, ATTACK_LABELS


def _hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


def draw_detections(
    img: np.ndarray,
    dets: FrameDetections,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    show_score: bool = True,
    class_names: dict[int, str] | None = None,
) -> np.ndarray:
    """Draw xyxy bounding boxes on a BGR uint8 image (returns a copy).

    Args:
        img:         HxWx3 BGR uint8 image.
        dets:        FrameDetections to draw.
        color:       BGR colour tuple.
        thickness:   Line thickness in pixels.
        show_score:  If True, overlay confidence score above each box.
        class_names: Optional int->str mapping for class labels.

    Returns:
        New array with boxes drawn; original is unchanged.
    """
    out = img.copy()
    for i in range(len(dets.boxes)):
        x1, y1, x2, y2 = [int(v) for v in dets.boxes[i].tolist()]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        if show_score and len(dets.scores) > i:
            cls_id = int(dets.classes[i]) if len(dets.classes) > i else -1
            cls_str = class_names.get(cls_id, str(cls_id)) if class_names else str(cls_id)
            label_str = f"{cls_str}:{dets.scores[i].item():.2f}"
            cv2.putText(
                out, label_str,
                (x1, max(y1 - 4, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1, cv2.LINE_AA,
            )
    return out


def _letterbox_bgr(img: np.ndarray, target: tuple[int, int] = (640, 640)) -> np.ndarray:
    """Return a letterboxed copy of *img* (gray padding, value=114)."""
    from src.viz.letterbox import letterbox_image  # noqa: PLC0415
    out, _, _ = letterbox_image(img, target=target)
    return out


def overlay_clean_vs_adv(
    clean_path: Path,
    adv_path: Path,
    clean_dets: Optional[FrameDetections] = None,
    adv_dets: Optional[FrameDetections] = None,
    attack_name: str = "adv",
    title: str = "",
    class_names: dict[int, str] | None = None,
) -> plt.Figure:
    """Side-by-side figure: clean frame (left) vs adversarial frame (right).

    v2: clean boxes are drawn on the left panel (green); adversarial boxes on
    the right panel (attack colour).  The clean image is letterboxed to 640x640
    so both panels have the same pixel dimensions.  A footer line below the
    figure reports the count: "Clean: N | ATTACK: M".

    Args:
        clean_path:  Path to the clean image (any size; will be letterboxed).
        adv_path:    Path to the adversarial image (640x640 letterboxed).
        clean_dets:  Detections on the clean image in 640x640 coordinates.
        adv_dets:    Detections on the adversarial image in 640x640 coordinates.
        attack_name: Key into PALETTE / ATTACK_LABELS.
        title:       Overall figure title.
        class_names: Optional dict for class label display.

    Returns:
        Matplotlib Figure (not yet saved).
    """
    clean_bgr = cv2.imread(str(clean_path))
    adv_bgr   = cv2.imread(str(adv_path))
    if clean_bgr is None:
        raise FileNotFoundError(f"Cannot load: {clean_path}")
    if adv_bgr is None:
        raise FileNotFoundError(f"Cannot load: {adv_path}")

    # Letterbox clean to 640×640 so boxes (in 640×640 space) align correctly
    if clean_bgr.shape[:2] != (640, 640):
        clean_bgr = _letterbox_bgr(clean_bgr, target=(640, 640))

    clean_color = _hex_to_bgr(PALETTE["clean"])
    adv_color   = _hex_to_bgr(PALETTE.get(attack_name, PALETTE["latent"]))

    # Draw clean boxes on left panel
    if clean_dets is not None and len(clean_dets.boxes) > 0:
        clean_draw = draw_detections(clean_bgr, clean_dets, color=clean_color,
                                     class_names=class_names)
    else:
        clean_draw = clean_bgr

    # Right panel: handle empty adv detections
    n_adv = len(adv_dets.boxes) if adv_dets is not None else 0
    if adv_dets is not None and n_adv == 0:
        adv_draw = adv_bgr.copy()
        hh, ww = adv_draw.shape[:2]
        # Dark overlay + text for "no detections" case
        overlay = adv_draw.astype(np.float32) * 0.4
        adv_draw = overlay.clip(0, 255).astype(np.uint8)
        for line_idx, text in enumerate([
            "No adversarial detections",
            "(binary-DFR success)",
        ]):
            cv2.putText(
                adv_draw, text,
                (ww // 2 - 140, hh // 2 - 15 + line_idx * 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 1, cv2.LINE_AA,
            )
    elif adv_dets is not None and n_adv > 0:
        adv_draw = draw_detections(adv_bgr, adv_dets, color=adv_color,
                                   class_names=class_names)
    else:
        adv_draw = adv_bgr

    # Convert BGR → RGB for matplotlib
    clean_rgb = cv2.cvtColor(clean_draw, cv2.COLOR_BGR2RGB)
    adv_rgb   = cv2.cvtColor(adv_draw,   cv2.COLOR_BGR2RGB)

    atk_label = ATTACK_LABELS.get(attack_name, attack_name.upper())
    n_clean = len(clean_dets.boxes) if clean_dets is not None else 0
    footer  = f"Clean: {n_clean} box{'es' if n_clean != 1 else ''} | {atk_label}: {n_adv}"

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    axes[0].imshow(clean_rgb)
    axes[0].set_title("Clean", fontsize=11)
    axes[0].axis("off")
    axes[1].imshow(adv_rgb)
    axes[1].set_title(atk_label, fontsize=11)
    axes[1].axis("off")

    suptitle = title if title else f"Clean vs {atk_label} — {Path(clean_path).stem}"
    fig.suptitle(suptitle, fontsize=12)
    fig.text(0.5, 0.02, footer, ha="center", fontsize=10, color="#333333")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    return fig
