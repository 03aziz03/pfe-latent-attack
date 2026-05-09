"""Tests for src/eval/metrics.py — strict metric definitions."""
from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from src.eval.metrics import (
    FrameDetections,
    aggregate,
    per_frame_asr,
    per_frame_conf_drop,
    per_frame_dfr,
    per_frame_map_drop,
    per_frame_masked_l2,
    per_frame_psnr_mask,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def make_fd(
    boxes: list[list[float]],
    scores: list[float],
    classes: list[int],
) -> FrameDetections:
    return FrameDetections(
        boxes=(
            torch.tensor(boxes, dtype=torch.float32)
            if boxes
            else torch.zeros((0, 4), dtype=torch.float32)
        ),
        scores=torch.tensor(scores, dtype=torch.float32),
        classes=torch.tensor(classes, dtype=torch.long),
    )


@pytest.fixture
def two_cars() -> FrameDetections:
    return make_fd([[10, 20, 100, 150], [200, 50, 350, 200]], [0.9, 0.85], [0, 0])


@pytest.fixture
def empty_fd() -> FrameDetections:
    return make_fd([], [], [])


@pytest.fixture
def two_cars_clone(two_cars: FrameDetections) -> FrameDetections:
    return FrameDetections(
        boxes=two_cars.boxes.clone(),
        scores=two_cars.scores.clone(),
        classes=two_cars.classes.clone(),
    )


# ---------------------------------------------------------------------------
# 1. Empty clean — DFR/ASR undefined; aggregate must skip the frame
# ---------------------------------------------------------------------------


def test_empty_clean_dfr_returns_zero(empty_fd: FrameDetections, two_cars: FrameDetections) -> None:
    """per_frame_dfr returns 0.0 when clean is empty (undefined case)."""
    assert per_frame_dfr(empty_fd, two_cars) == pytest.approx(0.0)


def test_empty_clean_asr_returns_false(empty_fd: FrameDetections, two_cars: FrameDetections) -> None:
    """per_frame_asr returns False when clean is empty (undefined case)."""
    assert per_frame_asr(empty_fd, two_cars) is False


def test_aggregate_skips_empty_clean_for_dfr_asr() -> None:
    """aggregate() excludes n_clean==0 frames when computing DFR and ASR."""
    per_frame = [
        {"frame_id": "f0", "n_clean": 0, "dfr": 0.0, "asr": False},  # skip
        {"frame_id": "f1", "n_clean": 3, "dfr": 0.5, "asr": True},
        {"frame_id": "f2", "n_clean": 2, "dfr": 1.0, "asr": True},
    ]
    result = aggregate(per_frame)
    assert result["dfr"] == pytest.approx(0.75)  # mean(0.5, 1.0) — f0 excluded
    assert result["asr"] == pytest.approx(1.0)   # mean(True, True) — f0 excluded


# ---------------------------------------------------------------------------
# 2. Empty adv — full attack success
# ---------------------------------------------------------------------------


def test_empty_adv_dfr_is_one(two_cars: FrameDetections, empty_fd: FrameDetections) -> None:
    assert per_frame_dfr(two_cars, empty_fd) == pytest.approx(1.0)


def test_empty_adv_asr_is_true(two_cars: FrameDetections, empty_fd: FrameDetections) -> None:
    assert per_frame_asr(two_cars, empty_fd) is True


def test_empty_adv_map_drop_is_one(two_cars: FrameDetections, empty_fd: FrameDetections) -> None:
    pytest.importorskip("torchmetrics")
    drop = per_frame_map_drop(two_cars, empty_fd)
    assert drop == pytest.approx(1.0, abs=1e-4)


# ---------------------------------------------------------------------------
# 3. Identical clean / adv — no attack effect
# ---------------------------------------------------------------------------


def test_identical_dfr_is_zero(two_cars: FrameDetections, two_cars_clone: FrameDetections) -> None:
    assert per_frame_dfr(two_cars, two_cars_clone) == pytest.approx(0.0)


def test_identical_asr_is_false(two_cars: FrameDetections, two_cars_clone: FrameDetections) -> None:
    assert per_frame_asr(two_cars, two_cars_clone) is False


def test_identical_conf_drop_is_zero(
    two_cars: FrameDetections, two_cars_clone: FrameDetections
) -> None:
    assert per_frame_conf_drop(two_cars, two_cars_clone) == pytest.approx(0.0)


def test_identical_images_masked_l2_is_zero() -> None:
    img = torch.rand(3, 64, 64)
    mask = torch.ones(64, 64, dtype=torch.bool)
    assert per_frame_masked_l2(img, img.clone(), mask) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 4. Class shift — original class disappears, replaced by a different class
# ---------------------------------------------------------------------------


def test_class_shift_asr_true() -> None:
    """Car (class 0) disappears; truck (class 1) appears at same location → ASR True."""
    clean = make_fd([[50, 50, 200, 200]], [0.9], [0])   # class 0 (car)
    adv = make_fd([[50, 50, 200, 200]], [0.8], [1])     # class 1 (truck)
    assert per_frame_asr(clean, adv) is True


def test_class_shift_dfr_is_zero() -> None:
    """One adv detection survives (different class), so proportional DFR = 0."""
    clean = make_fd([[50, 50, 200, 200]], [0.9], [0])
    adv = make_fd([[50, 50, 200, 200]], [0.8], [1])
    # n_clean=1, n_adv=1 → 1 - 1/1 = 0
    assert per_frame_dfr(clean, adv) == pytest.approx(0.0)


def test_class_shift_conf_drop_is_zero() -> None:
    """No same-class IoU match → conf_drop = 0."""
    clean = make_fd([[50, 50, 200, 200]], [0.9], [0])
    adv = make_fd([[50, 50, 200, 200]], [0.8], [1])
    assert per_frame_conf_drop(clean, adv) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 5. IoU edge case: matched at exactly iou_thr boundary
# ---------------------------------------------------------------------------
#
# Boxes:
#   clean: [0, 0, 2, 1]  area = 2
#   adv:   [0, 0, 1, 1]  area = 1
#   intersection = [0,0,1,1] = 1
#   union = 2 + 1 - 1 = 2
#   IoU = 1 / 2 = 0.5  (exactly)


def test_iou_boundary_match_at_threshold() -> None:
    """IoU == 0.5 should be matched when iou_thr = 0.5."""
    clean = make_fd([[0, 0, 2, 1]], [0.9], [0])
    adv = make_fd([[0, 0, 1, 1]], [0.7], [0])
    drop = per_frame_conf_drop(clean, adv, iou_thr=0.5)
    assert drop == pytest.approx(0.9 - 0.7)


def test_iou_boundary_no_match_above_threshold() -> None:
    """IoU == 0.5 should NOT be matched when iou_thr = 0.6."""
    clean = make_fd([[0, 0, 2, 1]], [0.9], [0])
    adv = make_fd([[0, 0, 1, 1]], [0.7], [0])
    drop = per_frame_conf_drop(clean, adv, iou_thr=0.6)
    assert drop == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 6. PSNR: identical images → +inf; known MSE → exact dB
# ---------------------------------------------------------------------------


def test_psnr_identical_is_inf() -> None:
    img = torch.rand(3, 64, 64)
    mask = torch.ones(64, 64, dtype=torch.bool)
    psnr = per_frame_psnr_mask(img, img.clone(), mask)
    assert math.isinf(psnr) and psnr > 0


def test_psnr_mask_one_pixel_max_error() -> None:
    """clean=0, adv=1 on 1 masked pixel → MSE=1 → PSNR=0 dB."""
    clean_img = torch.zeros(3, 8, 8)
    adv_img = torch.ones(3, 8, 8)
    mask = torch.zeros(8, 8, dtype=torch.bool)
    mask[0, 0] = True  # single pixel
    # diff = 1 on 3 channels; n_pixels = 1*3 = 3; MSE = 3/3 = 1.0; PSNR = 0 dB
    psnr = per_frame_psnr_mask(clean_img, adv_img, mask)
    assert psnr == pytest.approx(0.0, abs=1e-5)


def test_psnr_mask_respects_region() -> None:
    """Perturbation outside the mask should not affect masked PSNR."""
    clean_img = torch.zeros(3, 8, 8)
    adv_img = torch.ones(3, 8, 8)      # all pixels differ
    mask = torch.zeros(8, 8, dtype=torch.bool)
    # mask is empty — no pixels inside → inf (nothing to compare)
    psnr = per_frame_psnr_mask(clean_img, adv_img, mask)
    assert math.isinf(psnr) and psnr > 0


# ---------------------------------------------------------------------------
# 7. Masked L2: normalisation check
# ---------------------------------------------------------------------------


def test_masked_l2_full_mask_unit_diff() -> None:
    """All pixels differ by 1 over a full mask → masked_l2 = 1.0."""
    clean_img = torch.zeros(3, 4, 4)
    adv_img = torch.ones(3, 4, 4)
    mask = torch.ones(4, 4, dtype=torch.bool)
    # sum(diff^2) = 48; n = 16*3 = 48; sqrt(48/48) = 1.0
    assert per_frame_masked_l2(clean_img, adv_img, mask) == pytest.approx(1.0)


def test_masked_l2_partial_mask() -> None:
    """Only masked pixels count; zeros outside mask don't contribute."""
    clean_img = torch.zeros(3, 4, 4)
    adv_img = torch.zeros(3, 4, 4)
    adv_img[:, 0, 0] = 1.0            # perturb only one pixel (3 channels)
    mask = torch.zeros(4, 4, dtype=torch.bool)
    mask[0, 0] = True                  # mask that one pixel
    # sum(diff^2) = 3; n = 1*3 = 3; sqrt(3/3) = 1.0
    assert per_frame_masked_l2(clean_img, adv_img, mask) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 8. aggregate: inf PSNR handled correctly
# ---------------------------------------------------------------------------


def test_aggregate_inf_psnr_finite_mean() -> None:
    """Finite PSNR frames drive the mean; inf frames are excluded."""
    per_frame = [
        {"n_clean": 1, "psnr_mask": float("inf")},
        {"n_clean": 1, "psnr_mask": 30.0},
        {"n_clean": 1, "psnr_mask": 40.0},
    ]
    result = aggregate(per_frame)
    assert result["psnr_mask"] == pytest.approx(35.0)


def test_aggregate_all_inf_psnr() -> None:
    """When all PSNR values are inf, aggregate reports inf."""
    per_frame = [
        {"n_clean": 1, "psnr_mask": float("inf")},
        {"n_clean": 1, "psnr_mask": float("inf")},
    ]
    result = aggregate(per_frame)
    assert math.isinf(result["psnr_mask"])


# ---------------------------------------------------------------------------
# 9. io round-trip using synthetic fixture
# ---------------------------------------------------------------------------


def test_io_round_trip(tmp_path: Path) -> None:
    from src.eval.io import load_detections, save_detections

    fixture = Path(__file__).parent / "fixtures" / "synthetic_dets.json"
    dets = load_detections(fixture)

    # Basic sanity on fixture content
    assert "img00001_clean" in dets
    fd = dets["img00001_clean"]
    assert fd.boxes.shape == (2, 4)
    assert fd.scores.shape == (2,)
    assert fd.classes.shape == (2,)

    # Empty detections are represented as zero-row tensors
    empty = dets["img00001_adv_empty"]
    assert empty.boxes.shape == (0, 4)

    # Round-trip: save and reload
    out = tmp_path / "roundtrip.json"
    save_detections(dets, out)
    dets2 = load_detections(out)
    assert set(dets.keys()) == set(dets2.keys())
    for key in dets:
        assert torch.allclose(dets[key].boxes, dets2[key].boxes)
        assert torch.allclose(dets[key].scores, dets2[key].scores)
        assert torch.equal(dets[key].classes, dets2[key].classes)
