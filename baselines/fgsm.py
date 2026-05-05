"""Pixel-space FGSM baseline.

For fair comparison with our latent attack, FGSM is run with the SAME
detection loss (class-level vanishing loss). Optionally restricted to the
union of bounding boxes via a pixel mask.

Usage:
    python baselines/fgsm.py \
        --input data/images \
        --output results/adv_fgsm \
        --eps 0.0314 \
        --mask-restricted
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

# allow running this file as a script from the repo root
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.detector import YOLOv8Wrapper
from src.losses import vanishing_loss
from src.masks import boxes_to_pixel_mask
from src.data import ImageFolder
from src.utils import load_config, save_image, set_seed


def fgsm_attack(detector: YOLOv8Wrapper,
                x: torch.Tensor,
                eps: float = 8 / 255,
                gamma: float = 0.05,
                conf_thr: float = 0.25,
                iou_thr: float = 0.45,
                mask_restricted: bool = True) -> torch.Tensor:
    """One-step FGSM that increases false negatives.

    Args:
        x: (1, 3, H, W) image in [0, 1].
    Returns:
        x_adv with the same shape, in [0, 1].
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

    x_var = x.clone().detach().requires_grad_(True)
    raw = detector.forward_raw(x_var)
    class_conf = detector.class_confidence(raw)
    L_det = vanishing_loss(class_conf, C_clean, gamma=gamma)
    grad = torch.autograd.grad(L_det, x_var)[0]

    # gradient descent on L_det -> subtract sign(grad)
    perturb = -eps * grad.sign() * M
    x_adv = (x + perturb).clamp(0, 1)
    return x_adv.detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--input", required=True, help="folder of input images")
    ap.add_argument("--output", required=True, help="output folder for adv images")
    ap.add_argument("--eps", type=float, default=None)
    ap.add_argument("--mask-restricted", action="store_true", default=None,
                     help="restrict perturbation to bounding-box pixels")
    ap.add_argument("--full-image", action="store_true",
                     help="opposite of --mask-restricted")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["runtime"]["seed"])
    device = cfg["runtime"]["device"]

    eps = args.eps if args.eps is not None else cfg["baselines"]["eps_pixel"]
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

    for stem, x in tqdm(loader, total=len(loader), desc="FGSM"):
        x = x.to(device)
        x_adv = fgsm_attack(detector, x,
                              eps=eps,
                              gamma=cfg["attack"]["gamma"],
                              conf_thr=cfg["detector"]["conf_thr"],
                              iou_thr=cfg["detector"]["iou_nms"],
                              mask_restricted=mask_restricted)
        save_image(x_adv, out_dir / f"{stem}.png")

    print(f"FGSM (mask_restricted={mask_restricted}, eps={eps}) -> {out_dir}")


if __name__ == "__main__":
    main()
