"""Class-level analysis figures using Phase 1.5 per-detection data (f11, f12, f13)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision.ops as tv_ops
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.eval.io import load_detections
from src.eval.metrics import FrameDetections
from src.viz.style import ATTACK_LABELS, ATTACK_ORDER, PALETTE

# DETRAC YOLOv8n class names (confirmed from model.names)
DETRAC_CLASSES: dict[int, str] = {0: "car", 1: "bus", 2: "van", 3: "others"}


# ---------------------------------------------------------------------------
# Matching helper
# ---------------------------------------------------------------------------


def _match_pairs(
    clean: FrameDetections,
    adv: FrameDetections,
    iou_thr: float = 0.5,
) -> list[tuple[int, int, float]]:
    """Greedy IoU + same-class matching. Returns list of (ci, ai, iou)."""
    if len(clean.boxes) == 0 or len(adv.boxes) == 0:
        return []
    iou_mat = tv_ops.box_iou(clean.boxes.float(), adv.boxes.float())
    pairs: list[tuple[int, int, float]] = []
    used: set[int] = set()
    for ci in range(len(clean.boxes)):
        best_j, best_iou = -1, iou_thr - 1e-9
        for ai in range(len(adv.boxes)):
            if ai in used:
                continue
            if int(adv.classes[ai]) != int(clean.classes[ci]):
                continue
            v = float(iou_mat[ci, ai])
            if v > best_iou:
                best_j, best_iou = ai, v
        if best_j >= 0:
            pairs.append((ci, best_j, best_iou))
            used.add(best_j)
    return pairs


# ---------------------------------------------------------------------------
# f11: per-class survival bar chart
# ---------------------------------------------------------------------------


def class_breakdown_chart(
    clean_dets: dict[str, FrameDetections],
    adv_dets_dict: dict[str, dict[str, FrameDetections]],
    class_names: dict[int, str] | None = None,
    iou_thr: float = 0.5,
) -> plt.Figure:
    """Grouped bar chart: how many clean detections of each class survive each attack.

    Only same-class IoU-matched detections count as "survived".  New adversarial
    detections that do not match any clean box (spurious) are shown separately.

    Args:
        clean_dets:    Dict stem -> FrameDetections for clean images.
        adv_dets_dict: Dict attack_name -> (Dict stem -> FrameDetections).
        class_names:   Optional int -> str mapping; defaults to DETRAC_CLASSES.
        iou_thr:       IoU threshold for survival matching.

    Returns:
        Matplotlib Figure.
    """
    if class_names is None:
        class_names = DETRAC_CLASSES

    attacks = [a for a in ATTACK_ORDER if a in adv_dets_dict]

    # Aggregate clean counts per class
    clean_per_class: dict[int, int] = {}
    for fd in clean_dets.values():
        for cls in fd.classes.tolist():
            clean_per_class[cls] = clean_per_class.get(cls, 0) + 1

    # Aggregate survived counts per attack per class
    survived: dict[str, dict[int, int]] = {a: {} for a in attacks}
    spurious: dict[str, dict[int, int]] = {a: {} for a in attacks}
    for atk in attacks:
        for stem, c in clean_dets.items():
            a = adv_dets_dict[atk].get(stem)
            if a is None:
                continue
            matched_ai: set[int] = set()
            for ci, ai, _ in _match_pairs(c, a, iou_thr=iou_thr):
                cls = int(c.classes[ci])
                survived[atk][cls] = survived[atk].get(cls, 0) + 1
                matched_ai.add(ai)
            for ai in range(len(a.boxes)):
                if ai not in matched_ai:
                    cls = int(a.classes[ai])
                    spurious[atk][cls] = spurious[atk].get(cls, 0) + 1

    all_classes = sorted(set(clean_per_class) | {c for s in spurious.values() for c in s})
    labels = [class_names.get(c, f"cls_{c}") for c in all_classes]

    n_cls = len(all_classes)
    n_atk = len(attacks) + 1  # +1 for clean
    bar_w = 0.7 / n_atk
    x = np.arange(n_cls)

    fig, ax = plt.subplots(figsize=(max(7, n_cls * 2.5), 5))

    # Clean baseline bars
    clean_vals = [clean_per_class.get(c, 0) for c in all_classes]
    offset = (0 - n_atk / 2 + 0.5) * bar_w
    ax.bar(x + offset, clean_vals, width=bar_w, color=PALETTE["clean"],
           label="Clean (total)", alpha=0.9, edgecolor="white")

    for i, atk in enumerate(attacks, start=1):
        offset = (i - n_atk / 2 + 0.5) * bar_w
        surv_vals = [survived[atk].get(c, 0) for c in all_classes]
        spur_vals = [spurious[atk].get(c, 0) for c in all_classes]
        color = PALETTE.get(atk, "#999")
        atk_label = ATTACK_LABELS.get(atk, atk.upper())
        ax.bar(x + offset, surv_vals, width=bar_w, color=color,
               label=f"{atk_label} survived", alpha=0.85, edgecolor="white")
        ax.bar(x + offset, spur_vals, width=bar_w, color=color,
               label=f"{atk_label} spurious", alpha=0.35, edgecolor="white",
               bottom=surv_vals, hatch="//")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Detection count")
    ax.set_title(f"Per-class detection survival (IoU >= {iou_thr}, same-class match)")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# f12: per-frame detection count timeseries
# ---------------------------------------------------------------------------


def detection_timeseries(
    per_frame_data: dict[str, list[dict]],
    highlight_stems: list[str] | None = None,
) -> plt.Figure:
    """Line plot: n_clean (gray bars) and n_adv per attack over 50 frames.

    Args:
        per_frame_data:   Dict attack -> list of per-frame dicts with 'stem',
                          'n_clean', 'n_adv'.  Use one attack to get n_clean.
        highlight_stems:  Stems to mark with a star (binary-DFR success frames).

    Returns:
        Matplotlib Figure.
    """
    attacks = [a for a in ATTACK_ORDER if a in per_frame_data]
    if not attacks:
        raise ValueError("per_frame_data has no recognised attack keys")

    # Use first attack to get stem order and n_clean
    ref = per_frame_data[attacks[0]]
    stems = [f["stem"] for f in ref]
    n_clean = np.array([f["n_clean"] for f in ref])
    xs = np.arange(len(stems))

    if highlight_stems is None:
        # Default: latent binary-DFR success (n_adv == 0)
        latent_pf = {f["stem"]: f for f in (per_frame_data.get("latent") or [])}
        highlight_stems = [s for s in stems if latent_pf.get(s, {}).get("n_adv", 1) == 0]

    fig, ax = plt.subplots(figsize=(14, 5))

    # Gray background bars for n_clean
    ax.bar(xs, n_clean, color="#cccccc", alpha=0.7, width=0.8, label="n_clean", zorder=1)

    # Per-attack lines
    for atk in attacks:
        pf_map = {f["stem"]: f for f in per_frame_data[atk]}
        n_adv = np.array([pf_map.get(s, {}).get("n_adv", 0) for s in stems])
        ax.plot(xs, n_adv, "-o", color=PALETTE.get(atk, "#999"),
                markersize=4, linewidth=1.5,
                label=f"n_adv ({ATTACK_LABELS.get(atk, atk)})", zorder=3)

    # Highlight binary-DFR success frames with stars
    if highlight_stems:
        hi_xs = [xs[stems.index(s)] for s in highlight_stems if s in stems]
        hi_ys = [0] * len(hi_xs)
        ax.scatter(hi_xs, hi_ys, marker="*", s=120, color="#E74C3C", zorder=5,
                   label="Binary-DFR success (latent n_adv=0)")
        for hx in hi_xs:
            ax.axvline(hx, color="#E74C3C", alpha=0.15, linewidth=1, zorder=0)

    ax.set_xticks(xs[::5])
    ax.set_xticklabels([stems[i] for i in range(0, len(stems), 5)], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Detection count")
    ax.set_title("Per-frame detection counts: clean (bars) vs adversarial (lines)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# f13: IoU distribution of matched detections
# ---------------------------------------------------------------------------


def iou_distribution(
    clean_dets: dict[str, FrameDetections],
    adv_dets_dict: dict[str, dict[str, FrameDetections]],
    iou_thr: float = 0.3,
    n_bins: int = 14,
) -> plt.Figure:
    """Histogram of IoU values for (clean, adv) matched pairs per attack.

    A match is defined as same class + IoU >= iou_thr (greedy 1-to-1).
    This tests the hypothesis that LATENT pushes surviving boxes to new
    locations (low IoU) while FGSM barely moves them (high IoU).

    Args:
        clean_dets:    Dict stem -> FrameDetections for clean images.
        adv_dets_dict: Dict attack_name -> per-stem detections.
        iou_thr:       Minimum IoU for a match to be collected.
        n_bins:        Number of histogram bins in [iou_thr, 1.0].

    Returns:
        Matplotlib Figure (3 stacked panels, one per attack).
    """
    attacks = [a for a in ATTACK_ORDER if a in adv_dets_dict]
    bins = np.linspace(iou_thr, 1.0, n_bins + 1)

    fig, axes = plt.subplots(len(attacks), 1, figsize=(7, 3.5 * len(attacks)), sharex=True)
    if len(attacks) == 1:
        axes = [axes]

    for ax, atk in zip(axes, attacks):
        iou_vals: list[float] = []
        for stem, c in clean_dets.items():
            a = adv_dets_dict[atk].get(stem)
            if a is None:
                continue
            for _, _, iou_v in _match_pairs(c, a, iou_thr=iou_thr):
                iou_vals.append(iou_v)

        color = PALETTE.get(atk, "#999")
        label = ATTACK_LABELS.get(atk, atk.upper())
        n = len(iou_vals)
        if n > 0:
            ax.hist(iou_vals, bins=bins, color=color, alpha=0.75, edgecolor="white")
            mean_iou = float(np.mean(iou_vals))
            ax.axvline(mean_iou, color=color, linewidth=2, linestyle="--",
                       label=f"mean IoU = {mean_iou:.3f}")
            ax.legend(loc="upper left", fontsize=9)
        ax.set_title(f"{label}  (n_matched = {n})")
        ax.set_ylabel("Count")

    axes[-1].set_xlabel(f"IoU (matched pairs, threshold >= {iou_thr})")
    fig.suptitle(
        "IoU distribution of matched (same-class) detection pairs\n"
        "Low IoU = attack displaced surviving boxes; high IoU = boxes unchanged",
        fontsize=10,
    )
    fig.tight_layout()
    return fig
