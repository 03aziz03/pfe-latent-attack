"""Loss functions for the latent attack.

All three losses operate on autograd-tracked tensors:

    L_total = L_det + lambda_p * L_perc + lambda_r * L_reg
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
