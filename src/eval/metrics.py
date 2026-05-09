"""Strict metric definitions matching docs/method.tex §11.

Discrepancies vs. the task description (task desc → method.tex):
  DFR  : task says mean(1 - n_adv/n_clean); method.tex says fraction of frames
         with D_adv = ∅. per_frame_dfr returns the proportional value; the
         method.tex binary DFR corresponds to per_frame_dfr == 1.0 (n_adv == 0).
  ASR  : task says IoU-≥-0.5 instance matching; method.tex §11 says
         C_clean ∩ classes(D_adv) = ∅ (pure class membership, no IoU).
         Implemented per method.tex. The iou_thr parameter is kept for API
         compatibility but has no effect on the result.
  conf_drop: method.tex uses class-level pre-NMS max confidence; implemented
         here as post-NMS per-detection IoU-matched drop (task description).
  mAP_drop: not in method.tex; added per task specification.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torchvision.ops as tv_ops


@dataclass
class FrameDetections:
    """Per-frame detection output (post-NMS).

    Args:
        boxes:   (N, 4) float tensor, xyxy pixel coords.
        scores:  (N,) float tensor, confidence scores in [0, 1].
        classes: (N,) long tensor, class indices.
    """

    boxes: torch.Tensor
    scores: torch.Tensor
    classes: torch.Tensor


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _filter(fd: FrameDetections, conf_thr: float) -> FrameDetections:
    """Return a new FrameDetections keeping only boxes with score >= conf_thr."""
    if conf_thr <= 0.0:
        return fd
    mask = fd.scores >= conf_thr
    return FrameDetections(
        boxes=fd.boxes[mask],
        scores=fd.scores[mask],
        classes=fd.classes[mask],
    )


def _greedy_match(
    clean: FrameDetections,
    adv: FrameDetections,
    iou_thr: float,
) -> list[tuple[int, int]]:
    """Greedy 1-to-1 matching (clean_idx, adv_idx) by IoU within same class.

    Requires IoU >= iou_thr and same class label. Each index used at most once.
    Uses torchvision.ops.box_iou for IoU computation.
    """
    if len(clean.boxes) == 0 or len(adv.boxes) == 0:
        return []

    iou_matrix = tv_ops.box_iou(clean.boxes.float(), adv.boxes.float())  # (Nc, Na)
    pairs: list[tuple[int, int]] = []
    used_adv: set[int] = set()

    for i in range(len(clean.boxes)):
        best_j, best_iou = -1, iou_thr - 1e-9  # match at exactly iou_thr
        for j in range(len(adv.boxes)):
            if j in used_adv:
                continue
            if int(adv.classes[j]) != int(clean.classes[i]):
                continue
            iou_val = float(iou_matrix[i, j])
            if iou_val > best_iou:
                best_j, best_iou = j, iou_val
        if best_j >= 0:
            pairs.append((i, best_j))
            used_adv.add(best_j)

    return pairs


# ---------------------------------------------------------------------------
# Per-frame metrics
# ---------------------------------------------------------------------------


def per_frame_dfr(
    clean: FrameDetections,
    adv: FrameDetections,
    conf_thr: float = 0.25,
) -> float:
    """Proportional detection failure rate for one frame: 1 - n_adv / max(n_clean, 1).

    Note: method.tex §11 defines DFR as the fraction of frames with D_adv = ∅
    (a binary per-frame indicator). This function returns the proportional
    analogue; the method.tex DFR equals the fraction of frames where this
    function returns 1.0 (i.e. n_adv == 0). The aggregate() function computes
    both variants.

    Args:
        clean:    Clean-image detections.
        adv:      Adversarial-image detections.
        conf_thr: Confidence threshold applied to both sets.

    Returns:
        Float in (-∞, 1]. Returns 0.0 when n_clean == 0 (undefined; the
        aggregate function skips such frames for DFR).
    """
    c = _filter(clean, conf_thr)
    a = _filter(adv, conf_thr)
    n_clean = len(c.boxes)
    n_adv = len(a.boxes)
    if n_clean == 0:
        return 0.0
    return 1.0 - n_adv / n_clean


def per_frame_asr(
    clean: FrameDetections,
    adv: FrameDetections,
    iou_thr: float = 0.5,
    conf_thr: float = 0.25,
) -> bool:
    """True iff every originally-detected class is absent from adv detections.

    Follows method.tex §11: C_clean ∩ classes(D_adv) = ∅ (pure class
    membership, no IoU matching). The iou_thr parameter is retained for API
    compatibility but is not used in the computation.

    Discrepancy: the task description defines ASR via IoU-≥-iou_thr instance
    matching, which is more permissive. method.tex is stricter: the class must
    be completely absent, regardless of location.

    Args:
        clean:    Clean-image detections.
        adv:      Adversarial-image detections.
        iou_thr:  Unused (kept for API compatibility).
        conf_thr: Confidence threshold applied to both sets.

    Returns:
        False when n_clean == 0 (undefined; aggregate skips such frames).
    """
    c = _filter(clean, conf_thr)
    a = _filter(adv, conf_thr)
    if len(c.boxes) == 0:
        return False
    clean_classes: set[int] = set(c.classes.tolist())
    adv_classes: set[int] = set(a.classes.tolist())
    return len(clean_classes & adv_classes) == 0


def per_frame_map_drop(
    clean: FrameDetections,
    adv: FrameDetections,
    iou_thr: float = 0.5,
) -> float:
    """mAP drop = 1 - mAP@iou_thr, using clean detections as pseudo-GT.

    Uses torchmetrics.detection.MeanAveragePrecision. A perfect match between
    adv and clean gives drop = 0.0; no adv detections give drop = 1.0.

    Not in method.tex; added per task specification.

    Args:
        clean:   Clean-image detections (treated as ground truth).
        adv:     Adversarial-image detections (treated as predictions).
        iou_thr: IoU threshold for mAP computation (default 0.5).

    Returns:
        Float in [0, 1]. Returns 0.0 when n_clean == 0.

    Raises:
        ImportError: if torchmetrics is not installed.
    """
    from torchmetrics.detection import MeanAveragePrecision  # noqa: PLC0415

    if len(clean.boxes) == 0:
        return 0.0

    empty_boxes = torch.zeros((0, 4), dtype=torch.float32)
    empty_scores = torch.zeros(0, dtype=torch.float32)
    empty_labels = torch.zeros(0, dtype=torch.long)

    pred_boxes = adv.boxes.float() if len(adv.boxes) > 0 else empty_boxes
    pred_scores = adv.scores.float() if len(adv.scores) > 0 else empty_scores
    pred_labels = adv.classes.long() if len(adv.classes) > 0 else empty_labels

    metric = MeanAveragePrecision(iou_thresholds=[iou_thr])
    metric.update(
        preds=[{"boxes": pred_boxes, "scores": pred_scores, "labels": pred_labels}],
        target=[{"boxes": clean.boxes.float(), "labels": clean.classes.long()}],
    )
    result = metric.compute()
    map_val = float(result["map"].item())
    if not math.isfinite(map_val):
        map_val = 0.0
    return max(0.0, 1.0 - map_val)


def per_frame_psnr_mask(
    clean_img: torch.Tensor,
    adv_img: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    """PSNR computed only inside the mask (boolean H × W).

    Assumes images have values in [0, 1] and shape (C, H, W). The mask
    broadcasts over channels. Returns float('inf') when images are identical
    inside the mask (MSE = 0).

    Args:
        clean_img: (C, H, W) float tensor, values in [0, 1].
        adv_img:   (C, H, W) float tensor, values in [0, 1].
        mask:      (H, W) boolean tensor. True = inside masked region.

    Returns:
        PSNR in dB, or +inf if the images are identical inside the mask.
    """
    diff = (clean_img - adv_img) * mask.float()
    n_pixels = float(mask.sum().item()) * clean_img.shape[-3]  # #pixels * C
    if n_pixels < 1.0:
        return float("inf")
    mse = diff.pow(2).sum().item() / n_pixels
    if mse < 1e-12:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def per_frame_masked_l2(
    clean_img: torch.Tensor,
    adv_img: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    """RMS pixel error inside mask, normalised by mask area × 3 channels.

    Matches method.tex §6: ||M ⊙ (x_adv - x)||₂ / √(3 · ||M||₁).
    Here ||M||₁ is the count of True pixels in the mask.

    Args:
        clean_img: (C, H, W) float tensor, values in [0, 1].
        adv_img:   (C, H, W) float tensor, values in [0, 1].
        mask:      (H, W) boolean tensor.

    Returns:
        Non-negative float (zero if identical inside mask).
    """
    diff = (clean_img - adv_img) * mask.float()
    n_mask = float(mask.sum().item())
    n = max(n_mask * 3.0, 1e-8)
    return math.sqrt(diff.pow(2).sum().item() / n)


def per_frame_conf_drop(
    clean: FrameDetections,
    adv: FrameDetections,
    iou_thr: float = 0.5,
) -> float:
    """Mean (score_clean - score_adv) over greedy IoU-matched same-class pairs.

    Returns 0.0 if no pair is matched.

    Discrepancy vs method.tex §11: method.tex defines conf_drop as the
    class-level max pre-NMS confidence drop: mean_c(p_c(x) - p_c(x_adv)).
    This function uses post-NMS per-detection IoU matching (task specification).

    Args:
        clean:   Clean-image detections.
        adv:     Adversarial-image detections.
        iou_thr: Minimum IoU for a valid match.

    Returns:
        Mean confidence drop over matched pairs, or 0.0 if no match.
    """
    pairs = _greedy_match(clean, adv, iou_thr)
    if not pairs:
        return 0.0
    drops = [
        clean.scores[i].item() - adv.scores[j].item()
        for i, j in pairs
    ]
    return float(np.mean(drops))


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate(per_frame: list[dict]) -> dict:
    """Mean of each per-frame metric across frames.

    DFR and ASR are computed only over frames where n_clean > 0 (they are
    undefined for frames with no clean detections). All other metrics use
    every frame where the key is present and the value is finite.

    Inf PSNR values (identical images inside mask) are excluded from the mean;
    if ALL values are inf, the aggregate reports inf.

    Args:
        per_frame: List of dicts, each containing:
            - 'n_clean' (int): number of clean detections (required for DFR/ASR filtering)
            - metric keys: 'dfr', 'asr', 'map_drop', 'psnr_mask', 'masked_l2',
              'conf_drop', etc. (all optional, but must be float-castable).

    Returns:
        Dict mapping metric name → mean value (float).
    """
    if not per_frame:
        return {}

    dfr_asr_keys = {"dfr", "asr"}
    all_keys = {
        k
        for f in per_frame
        for k in f
        if k not in ("frame_id", "n_clean", "skipped")
    }

    result: dict[str, float] = {}
    for key in sorted(all_keys):
        if key in dfr_asr_keys:
            vals = [
                float(f[key])
                for f in per_frame
                if key in f and f.get("n_clean", 0) > 0 and f[key] is not None
            ]
        else:
            vals = [
                float(f[key])
                for f in per_frame
                if key in f and f[key] is not None
            ]

        finite_vals = [v for v in vals if math.isfinite(v)]
        if not vals:
            continue
        if finite_vals:
            result[key] = float(np.mean(finite_vals))
        else:
            result[key] = float("inf")

    return result
