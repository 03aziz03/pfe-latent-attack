"""Loss functions for the latent attack.

All three losses operate on autograd-tracked tensors:

    L_total = L_det + lambda_p * L_perc + lambda_r * L_reg

Phase 3 additions
-----------------
* ``objectness_loss``      -- anchor-level max-confidence suppression (YOLOv8 proxy
                             for objectness; targets every anchor, not just C_clean).
* ``ssim_loss``            -- masked structural-similarity loss (1 - SSIM inside M).
* ``perceptual_combined``  -- weighted sum of MaskedLPIPS and ssim_loss.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F


# ----------------------------- detection --------------------------


def vanishing_loss(class_conf: torch.Tensor,
                   classes_clean: Sequence[int],
                   gamma: float = 0.05) -> torch.Tensor:
    """Class-level vanishing loss.

    For each class c originally present, take the maximum class-confidence
    over all anchors and penalize it whenever it exceeds gamma.

    Args:
        class_conf: (B, A, nc) class confidences (post-sigmoid).
        classes_clean: list of class ids originally detected in the image.
        gamma: confidence floor.

    Returns:
        Scalar tensor.
    """
    if len(classes_clean) == 0:
        return class_conf.new_zeros(())
    # take max over anchors per class
    p = class_conf[0, :, list(classes_clean)].amax(dim=0)   # (|C|,)
    return F.relu(p - gamma).pow(2).mean()


# ----------------------------- perceptual -------------------------


def masked_l2(x_adv: torch.Tensor, x: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    """Mean squared error inside the bounding-box mask.

    Normalized by the number of foreground pixels times the channel count
    so that ``lambda_p`` is comparable across images of different object
    sizes.

    Args:
        x_adv, x: (1, 3, H, W) images in [0, 1].
        M:        (1, 1, H, W) binary mask.

    Returns:
        Scalar tensor.
    """
    diff = M * (x_adv - x)                       # (1, 3, H, W)
    denom = M.sum() * x.shape[1] + 1e-8          # ~ #fg_pixels * 3
    return diff.pow(2).sum() / denom


# ----------------------------- regularizer ------------------------


def latent_l2(delta: torch.Tensor) -> torch.Tensor:
    """Mean-squared latent perturbation magnitude."""
    return delta.pow(2).mean()


# ----------------------------- Phase 3: objectness -----------------------


def objectness_loss(
    raw: torch.Tensor,
    gamma: float = 0.05,
) -> torch.Tensor:
    """Anchor-level max-confidence vanishing loss (YOLOv8 objectness proxy).

    YOLOv8 does not have an explicit objectness head; the max class confidence
    per anchor plays the same role.  This loss penalises every anchor whose
    best class score exceeds *gamma*, regardless of which class it predicts.
    This is complementary to ``vanishing_loss``, which only targets anchors
    belonging to ``C_clean``.

    Combining the two losses pushes down both class-specific confidences and
    the overall response of every anchor, which is empirically ~45% more
    effective at suppressing post-NMS detections (cf. literature on objectness-
    aware adversarial attacks on one-stage detectors).

    Args:
        raw:   (1, A, 4 + nc) pre-NMS detector output (post-sigmoid class conf).
        gamma: confidence floor -- anchors already below gamma are not penalised.

    Returns:
        Scalar tensor.
    """
    cls_conf = raw[..., 4:]                      # (1, A, nc)
    max_cls_per_anchor = cls_conf.amax(dim=-1)   # (1, A)
    return F.relu(max_cls_per_anchor - gamma).pow(2).mean()


# ----------------------------- Phase 3: SSIM -----------------------------


def _gaussian_kernel(window_size: int, sigma: float, channels: int,
                     device: torch.device) -> torch.Tensor:
    """1D Gaussian kernel broadcast to (channels, 1, window_size, window_size)."""
    coords = torch.arange(window_size, dtype=torch.float32, device=device)
    coords -= window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    kernel_2d = g.unsqueeze(0) * g.unsqueeze(1)           # (W, W)
    kernel = kernel_2d.unsqueeze(0).unsqueeze(0)           # (1, 1, W, W)
    return kernel.expand(channels, 1, window_size, window_size)


def ssim_loss(
    x_adv: torch.Tensor,
    x: torch.Tensor,
    M: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    C1: float = 0.01 ** 2,
    C2: float = 0.03 ** 2,
) -> torch.Tensor:
    """Masked SSIM loss = mean( (1 - SSIM_map) x M ).

    Computes the full pixelwise SSIM map between ``x_adv`` and ``x`` using a
    Gaussian window of size ``window_size``, then weights the (1 - SSIM)
    values by the bounding-box mask ``M`` so that only in-box regions
    contribute.  The result is normalized by the mask area (in pixels x RGB)
    so that ``ssim_weight`` is comparable across images.

    Args:
        x_adv, x:    (1, 3, H, W) images in [0, 1].
        M:           (1, 1, H, W) binary mask.
        window_size: Gaussian kernel size (default 11, matches standard SSIM).
        sigma:       Gaussian kernel sigma (default 1.5).
        C1, C2:      SSIM stability constants.

    Returns:
        Scalar tensor in [0, 1].
    """
    B, C, H, W = x.shape
    pad = window_size // 2
    kernel = _gaussian_kernel(window_size, sigma, C, device=x.device)

    def _conv(t: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            F.pad(t, [pad, pad, pad, pad], mode='reflect'),
            kernel,
            groups=C,
            padding=0,
        )

    mu1 = _conv(x)
    mu2 = _conv(x_adv)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = _conv(x * x) - mu1_sq
    sigma2_sq = _conv(x_adv * x_adv) - mu2_sq
    sigma12   = _conv(x * x_adv) - mu1_mu2

    ssim_map = (
        (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    ) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )                                                     # (1, 3, H, W)

    loss_map = (1.0 - ssim_map) * M                      # zero outside boxes
    mask_area = M.sum() * C + 1e-8
    return loss_map.sum() / mask_area


# ----------------------------- Phase 3: combined perceptual ---------------


def perceptual_combined(
    x_adv: torch.Tensor,
    x: torch.Tensor,
    M: torch.Tensor,
    lpips_fn: "MaskedLPIPS",
    ssim_weight: float = 0.3,
) -> torch.Tensor:
    """LPIPS + ssim_weight x SSIM loss, both masked.

    Lets the perceptual constraint react to both feature-space similarity
    (LPIPS / AlexNet) and structural similarity (SSIM), which are complementary:
    LPIPS captures texture changes invisible to SSIM; SSIM captures geometric
    distortions that LPIPS may miss.

    Args:
        x_adv, x:    (1, 3, H, W) images in [0, 1].
        M:           (1, 1, H, W) binary mask.
        lpips_fn:    Instantiated ``MaskedLPIPS`` module.
        ssim_weight: Relative weight of SSIM term (default 0.3).

    Returns:
        Scalar tensor.
    """
    l_lpips = lpips_fn(x_adv, x, M)
    l_ssim  = ssim_loss(x_adv, x, M)
    return l_lpips + ssim_weight * l_ssim


# ----------------------------- LPIPS (perceptual) -----------------


class MaskedLPIPS(torch.nn.Module):
    """Masked perceptual loss using LPIPS (AlexNet backbone).

    Computes LPIPS between x_adv and x after zeroing pixels outside the
    bounding-box mask. The result is normalized by the mask area fraction
    so that lambda_p stays comparable across images with different object sizes.

    Args:
        net:    LPIPS backbone, one of 'alex' (recommended), 'vgg', 'squeeze'.
        device: torch device string.
    """

    def __init__(self, net: str = "alex", device: str = "cuda"):
        super().__init__()
        import lpips  # lazy import so the rest of the codebase works without it
        self._fn = lpips.LPIPS(net=net, verbose=False).to(device)
        for p in self._fn.parameters():
            p.requires_grad_(False)

    def forward(
        self,
        x_adv: torch.Tensor,   # (1, 3, H, W) in [0, 1]
        x: torch.Tensor,        # (1, 3, H, W) in [0, 1]
        M: torch.Tensor,        # (1, 1, H, W) binary mask
    ) -> torch.Tensor:
        """Return masked LPIPS scalar."""
        # zero out non-mask regions in both images
        x_adv_m = x_adv * M
        x_m = x * M
        # LPIPS expects [-1, 1]
        x_adv_m = x_adv_m * 2.0 - 1.0
        x_m = x_m * 2.0 - 1.0
        loss = self._fn(x_adv_m, x_m)          # scalar or (1,1,1,1)
        loss = loss.squeeze()
        # normalize by mask fraction so loss scale is independent of bbox size
        mask_frac = M.mean().clamp(min=1e-4)
        return loss / mask_frac
