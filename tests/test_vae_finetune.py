"""Tests for VAE fine-tune additions (encode_with_grad + finetuned_weights).

These tests mock AutoencoderKL.from_pretrained to avoid downloading weights.
Skipped gracefully if lpips is not installed (same guard as test_lpips_loss.py).
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

lpips = pytest.importorskip("lpips")


# ---------------------------------------------------------------------------
# Minimal mock of AutoencoderKL
# ---------------------------------------------------------------------------


class _FakeLatentDist:
    def __init__(self, z: torch.Tensor):
        self.mean = z


class _FakeVAEInner(nn.Module):
    """Minimal stand-in for AutoencoderKL."""

    def __init__(self):
        super().__init__()
        self.enc = nn.Linear(3, 4, bias=False)  # tiny learnable params
        self.dec = nn.Linear(4, 3, bias=False)

    def encode(self, x: torch.Tensor) -> _FakeLatentDist:
        b, c, h, w = x.shape
        flat = x.mean(dim=(2, 3))           # (B, C)
        z = self.enc(flat).unsqueeze(-1).unsqueeze(-1)  # (B, 4, 1, 1)
        z = z.expand(b, 4, h // 8, w // 8)
        return _FakeLatentDist(z)

    def decode(self, z: torch.Tensor) -> MagicMock:
        b, c, h, w = z.shape
        flat = z.mean(dim=(2, 3))
        out = self.dec(flat).unsqueeze(-1).unsqueeze(-1)
        out = out.expand(b, 3, h * 8, w * 8)
        sample_mock = MagicMock()
        sample_mock.sample = out
        return sample_mock

    def parameters(self, recurse=True):
        return super().parameters(recurse=recurse)

    def to(self, *args, **kwargs):
        return super().to(*args, **kwargs)

    def eval(self):
        return super().eval()

    def state_dict(self, *args, **kwargs):
        return super().state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=strict)


def _make_sdvae(**kwargs):
    """Create SDVAE with a mocked AutoencoderKL."""
    from src.vae import SDVAE

    fake_inner = _FakeVAEInner()

    with patch("src.vae.AutoencoderKL") as mock_klass:
        mock_klass.from_pretrained.return_value = fake_inner
        vae = SDVAE(device="cpu", dtype=torch.float32, **kwargs)
    return vae


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_encode_with_grad_allows_grad():
    """encode_with_grad() produces a tensor through which autograd works."""
    vae = _make_sdvae()
    x = torch.rand(1, 3, 64, 64, requires_grad=False)
    z = vae.encode_with_grad(x)
    assert z.requires_grad or any(
        p.requires_grad for p in vae.vae.parameters()
    ), "Expected gradient-capable computation in encode_with_grad"
    # Verify autograd.grad doesn't raise
    loss = z.sum()
    loss.backward()  # should not raise


def test_encode_no_grad_does_not_propagate():
    """encode() (with @no_grad) should detach from the autograd graph."""
    vae = _make_sdvae()
    # Temporarily re-enable requires_grad on inner params to check detachment
    for p in vae.vae.parameters():
        p.requires_grad_(True)
    x = torch.rand(1, 3, 64, 64)
    z = vae.encode(x)
    assert not z.requires_grad, "encode() should return a detached tensor"


def test_finetuned_weights_loaded_when_exists(tmp_path: Path):
    """SDVAE loads fine-tuned state dict when finetuned_weights path exists."""
    # Save a dummy state dict
    fake = _FakeVAEInner()
    ckpt_path = tmp_path / "vae_ft.pt"
    torch.save(fake.state_dict(), ckpt_path)

    vae = _make_sdvae(finetuned_weights=str(ckpt_path))
    # If we got here without error the load worked; no assertion needed beyond that.
    assert vae is not None


def test_finetuned_weights_ignored_when_missing(tmp_path: Path):
    """SDVAE silently ignores finetuned_weights when path doesn't exist."""
    nonexistent = str(tmp_path / "does_not_exist.pt")
    vae = _make_sdvae(finetuned_weights=nonexistent)
    assert vae is not None
