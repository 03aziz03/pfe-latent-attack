"""Three sanity checks to verify the pipeline before scaling up.

Check 1: zero-perturbation round-trip.
    With delta = 0, the paste-back image should be (numerically) equal to x
    inside the mask up to VAE reconstruction error, and *exactly* equal
    outside the mask.

Check 2: attack actually drops confidences.
    After running the attack, p_c should be < gamma for every class
    originally present.

Check 3: perturbation is mask-localized.
    |x_adv - x| outside the mask should be exactly zero (up to floating
    point) thanks to the paste-back.

Usage:
    python scripts/sanity_check.py --image data/images/your_image.jpg
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.detector import YOLOv8Wrapper
from src.vae import SDVAE
from src.attack import LatentObjectAttack, AttackConfig
from src.masks import boxes_to_pixel_mask, pixel_mask_to_latent_mask
from src.utils import load_config, load_image, save_image, set_seed


def check_paste_back(detector, vae, x, M, Mz):
    """Check 1 + 3: paste-back zeros perturbation outside M, equals x inside up to VAE error."""
    z = vae.encode(x)
    delta = torch.zeros_like(z)
    z_adv = z + Mz * delta
    x_dec = vae.decode(z_adv)
    x_paste = M * x_dec + (1 - M) * x

    diff_outside = (x_paste - x) * (1 - M)
    diff_inside = (x_paste - x) * M
    print(f"  outside mask  |x' - x|_max = {diff_outside.abs().max().item():.3e}  "
          f"(expected ~0 due to paste-back)")
    print(f"  inside  mask  |x' - x|_max = {diff_inside.abs().max().item():.3e}  "
          f"(VAE reconstruction error, ~1e-2 typical)")


def check_attack_drops_confidence(result, cfg):
    """Check 2."""
    final_p_max = result.history["p_max"][-1] if result.history["p_max"] else None
    print(f"  steps_taken = {result.steps_taken}")
    print(f"  initial L_det = {result.history['L_det'][0]:.4f}")
    print(f"  final   L_det = {result.history['L_det'][-1]:.4f}")
    print(f"  final   p_max = {final_p_max}  (target: < gamma = {cfg.gamma})")
    if final_p_max is not None and final_p_max < cfg.gamma:
        print("  PASS - all originally-detected classes vanished")
    else:
        print("  WARN - some classes still above gamma; consider increasing eps_z or num_steps")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--image", required=True)
    ap.add_argument("--output", default="results/sanity")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["runtime"]["seed"])
    device = cfg["runtime"]["device"]

    print("Loading detector and VAE...")
    detector = YOLOv8Wrapper(cfg["detector"]["weights"], device=device)
    vae = SDVAE(cfg["vae"]["model_id"], scale=cfg["vae"]["scale"], device=device)

    print(f"Loading image {args.image}")
    x = load_image(args.image, imgsz=cfg["detector"]["imgsz"]).to(device)

    print("\n[Check 1+3] zero-perturbation paste-back consistency")
    D_clean = detector.detect_nms(x,
                                    conf_thr=cfg["detector"]["conf_thr"],
                                    iou_thr=cfg["detector"]["iou_nms"])
    print(f"  clean detections: {len(D_clean)}")
    if not D_clean:
        print("  no detections -> nothing to attack on this image; "
              "try a different image (e.g. one with people / cars / animals).")
        return
    M = boxes_to_pixel_mask(D_clean, H=x.shape[-2], W=x.shape[-1], device=device)
    Mz = pixel_mask_to_latent_mask(M)
    check_paste_back(detector, vae, x, M, Mz)

    print("\n[Check 2] attack drops class confidences below gamma")
    acfg = AttackConfig(
        eps_z=cfg["attack"]["eps_z"],
        gamma=cfg["attack"]["gamma"],
        lambda_p=cfg["attack"]["lambda_p"],
        lambda_r=cfg["attack"]["lambda_r"],
        lr=cfg["attack"]["lr"],
        num_steps=cfg["attack"]["num_steps"],
        early_stop=cfg["attack"]["early_stop"],
        early_stop_margin=cfg["attack"]["early_stop_margin"],
        conf_thr=cfg["detector"]["conf_thr"],
        iou_nms=cfg["detector"]["iou_nms"],
    )
    attack = LatentObjectAttack(detector, vae, acfg)
    result = attack.attack(x)
    check_attack_drops_confidence(result, acfg)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    save_image(x, out / "clean.png")
    save_image(result.x_adv, out / "adv.png")
    diff = (result.x_adv - x).abs() * 10  # 10x amplification
    save_image(diff.clamp(0, 1), out / "diff_x10.png")
    save_image(M.expand(-1, 3, -1, -1), out / "mask.png")
    print(f"\nWrote sanity outputs to {out}/  (clean.png, adv.png, diff_x10.png, mask.png)")


if __name__ == "__main__":
    main()
