"""Letterbox preprocessing to align clean and adversarial coordinate spaces.

YOLO saves adversarial images at 640×640 letterboxed (gray bars top/bottom
for landscape images). Running YOLO on these returns boxes in 640×640 space.
But running YOLO on the original 960×540 clean images returns boxes in
960×540 space. Computing IoU across these spaces always gives ~0.

This module provides:
  letterbox_image     — resize+pad an image to a target square
  unletterbox_boxes   — inverse transform from letterboxed to original coords

Usage in the evaluation pipeline: pre-letterbox clean images to 640×640
before YOLO inference so all four detection JSON files share the same
coordinate frame.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch


def letterbox_image(
    img: np.ndarray,
    target: tuple[int, int] = (640, 640),
    pad_value: int = 114,
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize *img* preserving aspect ratio, then pad to *target* with gray.

    Matches the preprocessing YOLO applies internally, so images letterboxed
    by this function produce boxes in the *target* coordinate frame when
    passed through YOLO (which skips its own letterboxing when the input is
    already the correct size).

    Args:
        img:       HxWxC BGR uint8 image.
        target:    (target_h, target_w) output dimensions in pixels.
        pad_value: Gray-fill value for padding; YOLO standard is 114.

    Returns:
        Tuple of:
          - letterboxed_img: uint8 array of shape (*target, 3)
          - scale: float, the uniform resize factor (same for both axes)
          - (pad_top, pad_left): pixel offsets of the content area inside
            the padded result; useful for unletterbox_boxes.
    """
    h, w = img.shape[:2]
    target_h, target_w = target
    scale = min(target_h / h, target_w / w)
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_top = (target_h - new_h) // 2
    pad_bottom = target_h - new_h - pad_top
    pad_left = (target_w - new_w) // 2
    pad_right = target_w - new_w - pad_left

    padded = cv2.copyMakeBorder(
        resized,
        pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT,
        value=(pad_value, pad_value, pad_value),
    )
    return padded, float(scale), (int(pad_top), int(pad_left))


def unletterbox_boxes(
    boxes: torch.Tensor,
    scale: float,
    pad: tuple[int, int],
    orig_size: tuple[int, int],
) -> torch.Tensor:
    """Transform xyxy boxes from letterboxed frame back to original image coords.

    Args:
        boxes:     (N, 4) float tensor, xyxy in the letterboxed coordinate frame.
        scale:     Scale factor from letterbox_image (same for x and y).
        pad:       (pad_top, pad_left) from letterbox_image.
        orig_size: (orig_h, orig_w) of the source image before letterboxing.

    Returns:
        (N, 4) float tensor in the original coordinate space, clipped to the
        image boundary.  Empty input returns empty output without error.
    """
    if boxes.numel() == 0:
        return boxes.clone()

    pad_top, pad_left = pad
    orig_h, orig_w = orig_size

    out = boxes.clone().float()
    out[:, 0] = ((out[:, 0] - pad_left) / scale).clamp(0.0, float(orig_w))
    out[:, 1] = ((out[:, 1] - pad_top)  / scale).clamp(0.0, float(orig_h))
    out[:, 2] = ((out[:, 2] - pad_left) / scale).clamp(0.0, float(orig_w))
    out[:, 3] = ((out[:, 3] - pad_top)  / scale).clamp(0.0, float(orig_h))
    return out
