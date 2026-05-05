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
