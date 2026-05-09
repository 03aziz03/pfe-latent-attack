"""Unit tests for src/viz/letterbox.py."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.viz.letterbox import letterbox_image, unletterbox_boxes


# ---------------------------------------------------------------------------
# letterbox_image tests
# ---------------------------------------------------------------------------


def test_letterbox_output_shape_landscape():
    """Wide image padded top/bottom to reach target square."""
    img = np.zeros((540, 960, 3), dtype=np.uint8)
    out, scale, (pt, pl) = letterbox_image(img, target=(640, 640))
    assert out.shape == (640, 640, 3)
    assert abs(scale - 640 / 960) < 1e-5, f"scale={scale}"
    assert pl == 0, "no left padding expected for landscape"
    assert pt > 0, "top padding expected for landscape"


def test_letterbox_output_shape_portrait():
    """Tall image padded left/right."""
    img = np.zeros((960, 540, 3), dtype=np.uint8)
    out, scale, (pt, pl) = letterbox_image(img, target=(640, 640))
    assert out.shape == (640, 640, 3)
    assert abs(scale - 640 / 960) < 1e-5
    assert pl > 0
    assert pt == 0


def test_letterbox_output_shape_square():
    """Square image is just resized, no padding."""
    img = np.zeros((960, 960, 3), dtype=np.uint8)
    out, scale, (pt, pl) = letterbox_image(img, target=(640, 640))
    assert out.shape == (640, 640, 3)
    assert pt == 0 and pl == 0


def test_letterbox_pad_value():
    """Gray pad is filled with pad_value."""
    img = np.zeros((540, 960, 3), dtype=np.uint8)
    out, scale, (pt, pl) = letterbox_image(img, target=(640, 640), pad_value=114)
    # Top padding should be all 114
    if pt > 0:
        assert np.all(out[:pt] == 114)
    # Content area should be all 0 (black)
    new_h = int(round(540 * scale))
    assert np.all(out[pt : pt + new_h] == 0)


def test_letterbox_content_preserved():
    """White center region survives letterboxing (content is not discarded)."""
    img = np.full((540, 960, 3), 200, dtype=np.uint8)
    out, scale, (pt, pl) = letterbox_image(img, target=(640, 640), pad_value=114)
    new_h = int(round(540 * scale))
    # Content area should be ~200 (not the pad value)
    content = out[pt : pt + new_h]
    assert content.mean() > 150


# ---------------------------------------------------------------------------
# unletterbox_boxes tests
# ---------------------------------------------------------------------------


def test_unletterbox_round_trip_landscape():
    """Box coords survive letterbox_image → YOLO → unletterbox_boxes within 1 px."""
    orig_size = (540, 960)
    img = np.zeros((540, 960, 3), dtype=np.uint8)
    _, scale, pad = letterbox_image(img, target=(640, 640))

    # Simulate a box in the ORIGINAL image coordinates
    # After letterboxing: x_lb = x_orig * scale + pad_left, etc.
    x1_orig, y1_orig, x2_orig, y2_orig = 100.0, 50.0, 300.0, 150.0
    pt, pl = pad
    x1_lb = x1_orig * scale + pl
    y1_lb = y1_orig * scale + pt
    x2_lb = x2_orig * scale + pl
    y2_lb = y2_orig * scale + pt

    boxes_lb = torch.tensor([[x1_lb, y1_lb, x2_lb, y2_lb]])
    boxes_orig = unletterbox_boxes(boxes_lb, scale, pad, orig_size)

    assert abs(float(boxes_orig[0, 0]) - x1_orig) <= 1.0
    assert abs(float(boxes_orig[0, 1]) - y1_orig) <= 1.0
    assert abs(float(boxes_orig[0, 2]) - x2_orig) <= 1.0
    assert abs(float(boxes_orig[0, 3]) - y2_orig) <= 1.0


def test_unletterbox_clips_to_boundary():
    """Boxes outside orig image are clipped."""
    orig_size = (540, 960)
    boxes_lb = torch.tensor([[0.0, 0.0, 700.0, 700.0]])  # extends beyond 640×640
    scale = 640.0 / 960.0
    pad = (80, 0)  # typical for 960×540 → 640×640
    out = unletterbox_boxes(boxes_lb, scale, pad, orig_size)
    assert float(out[0, 2]) <= 960.0
    assert float(out[0, 3]) <= 540.0


def test_unletterbox_empty():
    """Empty input returns empty output without error."""
    boxes = torch.zeros((0, 4))
    out = unletterbox_boxes(boxes, scale=1.0, pad=(0, 0), orig_size=(640, 640))
    assert out.shape == (0, 4)


def test_unletterbox_identity_for_square():
    """640×640 square image: scale=1, pad=(0,0) → boxes unchanged."""
    orig_size = (640, 640)
    boxes = torch.tensor([[100.0, 50.0, 300.0, 200.0]])
    out = unletterbox_boxes(boxes, scale=1.0, pad=(0, 0), orig_size=orig_size)
    assert torch.allclose(out, boxes)
