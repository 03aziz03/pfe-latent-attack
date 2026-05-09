"""Tests for MaskedLPIPS.

All tests are skipped gracefully when lpips is not installed.
"""
from __future__ import annotations

import pytest
import torch

lpips = pytest.importorskip("lpips")

from src.losses import MaskedLPIPS  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def loss_fn() -> MaskedLPIPS:
    return MaskedLPIPS(net="alex", device=DEVICE)


def _rand(shape: tuple[int, ...]) -> torch.Tensor:
    return torch.rand(shape, device=DEVICE)


def _full_mask(h: int = 64, w: int = 64) -> torch.Tensor:
    return torch.ones(1, 1, h, w, device=DEVICE)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_construction():
    """MaskedLPIPS constructs without error."""
    fn = MaskedLPIPS(net="alex", device=DEVICE)
    assert fn is not None


def test_forward_returns_scalar(loss_fn: MaskedLPIPS):
    """Forward pass on random inputs returns a 0-d scalar tensor."""
    x_adv = _rand((1, 3, 64, 64))
    x = _rand((1, 3, 64, 64))
    M = _full_mask()
    out = loss_fn(x_adv, x, M)
    assert out.ndim == 0, f"Expected scalar, got shape {out.shape}"
    assert out.dtype in (torch.float32, torch.float64)


def test_identical_images_near_zero(loss_fn: MaskedLPIPS):
    """x_adv == x should yield loss ≈ 0."""
    x = _rand((1, 3, 64, 64))
    M = _full_mask()
    out = loss_fn(x, x, M)
    assert float(out.item()) < 1e-3, f"Expected ~0 for identical images, got {out.item()}"


def test_different_images_positive(loss_fn: MaskedLPIPS):
    """x_adv != x should yield loss > 0."""
    x = _rand((1, 3, 64, 64))
    x_adv = torch.ones(1, 3, 64, 64, device=DEVICE)  # maximally different from typical rand
    M = _full_mask()
    out = loss_fn(x_adv, x, M)
    assert float(out.item()) > 0.0, "Expected positive loss for different images"


def test_zero_mask_near_zero(loss_fn: MaskedLPIPS):
    """All-zero mask → both masked images are zero → loss ≈ 0."""
    x_adv = _rand((1, 3, 64, 64))
    x = _rand((1, 3, 64, 64))
    M = torch.zeros(1, 1, 64, 64, device=DEVICE)
    out = loss_fn(x_adv, x, M)
    # With M=0, both x_adv*M and x*M are zero, so LPIPS(−1, −1) ≈ 0.
    # The mask_frac is clamped to 1e-4, so the result is loss/1e-4 but loss itself ≈ 0.
    assert float(out.item()) < 1e-2, f"Expected ~0 for zero mask, got {out.item()}"
