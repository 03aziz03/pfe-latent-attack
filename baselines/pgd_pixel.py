"""Pixel-space PGD baseline (full-image and mask-restricted variants).

Same vanishing detection loss as the latent attack so that comparisons isolate
the effect of the latent space rather than the loss function.

Usage:
    python baselines/pgd_pixel.py \
        --input data/images \
        --output results/adv_pgd \
        --steps 50 --eps 0.0314 --alpha 0.00392 --mask-restricted
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.detector import YOLOv8Wrapper
from src.losses import vanishing_loss
from src.masks import boxes_to_pixel_mask
from src.data import ImageFolder
from src.utils import load_config, save_image, set_seed


def pgd_attack(detector: YOLOv8Wrapper,
               x: torch.Tensor,
               eps: float = 8 / 255,
               alpha: float = 1 / 255,
               steps: int = 50,
               gamma: float = 0.05,
               conf_thr: float = 0.25,
               iou_thr: float = 0.45,
               mask_restricted: bool = True,
               early_stop: bool = True) -> torch.Tensor:
    """L-inf PGD vanishing attack.

    Returns ``x_adv`` in [0, 1] with ``||x_adv - x||_inf <= eps`` (within mask
    if ``mask_restricted=True``).
    """
    device = x.device
    D_clean = detector.detect_nms(x, conf_thr=conf_thr, iou_thr=iou_thr)
    if len(D_clean) == 0:
        return x.clone()
    C_clean = sorted({d.cls for d in D_clean})

    if mask_restricted:
        M = boxes_to_pixel_mask(D_clean, H=x.shape[-2], W=x.shape[-1], device=device)
    else:
        M = torch.ones((1, 1, x.shape[-2], x.shape[-1]), device=device)

    delta = torch.zeros_like(x, requires_grad=True)
    for t in range(steps):
        x_adv = (x + delta).clamp(0, 1)
        raw = detector.forward_raw(x_adv)
        class_conf = detector.class_confidence(raw)
        L_det = vanishing_loss(class_conf, C_clean, gamma=gamma)

        grad = torch.autograd.grad(L_det, delta)[0]
        with torch.no_grad():
            # gradient descent step: minimize L_det
            delta -= alpha * grad.sign()
            delta.clamp_(-eps, eps)
            delta.mul_(M)
            # ensure x + delta in [0, 1]
            delta.data = (x + delta).clamp(0, 1) - x

            p = class_conf[0, :, C_clean].amax(dim=0).max().item()
        delta.requires_grad_(True)

        if early_stop and p < gamma:
            break

    return (x + delta.detach()).clamp(0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--eps", type=float, default=None)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--mask-restricted", action="store_true", default=None)
    ap.add_argument("--full-image", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["runtime"]["seed"])
    device = cfg["runtime"]["device"]

    eps = args.eps if args.eps is not None else cfg["baselines"]["eps_pixel"]
    alpha = args.alpha if args.alpha is not None else cfg["baselines"]["pgd_alpha"]
    steps = args.steps if args.steps is not None else cfg["baselines"]["pgd_steps"]
    if args.full_image:
        mask_restricted = False
    elif args.mask_restricted:
        mask_restricted = True
    else:
        mask_restricted = cfg["baselines"]["mask_restricted"]

    detector = YOLOv8Wrapper(cfg["detector"]["weights"], device=device)
    loader = ImageFolder(args.input, imgsz=cfg["detector"]["imgsz"])

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    desc = f"PGD (mask={mask_restricted})"
    for stem, x in tqdm(loader, total=len(loader), desc=desc):
        x = x.to(device)
        x_adv = pgd_attack(detector, x,
                             eps=eps, alpha=alpha, steps=steps,
                             gamma=cfg["attack"]["gamma"],
                             conf_thr=cfg["detector"]["conf_thr"],
                             iou_thr=cfg["detector"]["iou_nms"],
                             mask_restricted=mask_restricted)
        save_image(x_adv, out_dir / f"{stem}.png")

    print(f"PGD (mask_restricted={mask_restricted}, eps={eps}, steps={steps}) "
          f"-> {out_dir}")


if __name__ == "__main__":
    main()
