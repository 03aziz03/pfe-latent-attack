"""Bounding-box -> pixel mask -> latent mask.

The pixel mask is the union of detected bounding boxes (1 inside, 0 outside).
The latent mask is obtained by max-pooling the pixel mask with a 8x8 kernel
to match the SD VAE's stride-8 downsampling, then broadcast to 4 channels.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F

from .utils import Detection


def boxes_to_pixel_mask(dets: Sequence[Detection],
                         H: int,
                         W: int,
                         device: str | torch.device = "cpu") -> torch.Tensor:
    """Build a (1, 1, H, W) binary mask from detections.

    Box coordinates are clamped to image bounds and rounded to int pixels.
    """
    M = torch.zeros((1, 1, H, W), device=device)
    for d in dets:
        x1, y1, x2, y2 = d.box
        x1i = max(0, int(round(float(x1))))
        y1i = max(0, int(round(float(y1))))
        x2i = min(W, int(round(float(x2))))
        y2i = min(H, int(round(float(y2))))
        if x2i > x1i and y2i > y1i:
            M[..., y1i:y2i, x1i:x2i] = 1.0
    return M


def pixel_mask_to_latent_mask(M: torch.Tensor,
                               latent_channels: int = 4,
                               stride: int = 8) -> torch.Tensor:
    """Downsample a pixel mask to the latent grid.

    The use of max-pool (rather than average-pool) ensures any latent cell
    whose 8x8 receptive field touches a foreground pixel is marked active,
    so we never miss pieces of small objects.

    Args:
        M: (1, 1, H, W) binary mask.
        latent_channels: number of channels in the VAE latent (4 for SD).
        stride: VAE spatial downsampling factor (8 for SD).

    Returns:
        (1, latent_channels, H/stride, W/stride) binary mask.
    """
    if M.shape[-2] % stride != 0 or M.shape[-1] % stride != 0:
        raise ValueError(
            f"Pixel mask spatial size {M.shape[-2:]} not divisible by "
            f"stride {stride}; resize / letterbox to a multiple of {stride}."
        )
    Mz = F.max_pool2d(M, kernel_size=stride, stride=stride)
    Mz = Mz.expand(-1, latent_channels, -1, -1).contiguous()
    return Mz
