"""Batch latent attack over a folder of images.

For each image:
    1. run YOLOv8 to get D_clean
    2. run LatentObjectAttack
    3. save adversarial image to --output
    4. record per-image stats (steps, final p_max, # original detections)
       to a JSON metadata file alongside the images.

Usage:
    python scripts/run_attack.py --input data/images --output results/adv_latent
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.detector import YOLOv8Wrapper
from src.vae import SDVAE
from src.attack import LatentObjectAttack, AttackConfig
from src.data import ImageFolder
from src.utils import load_config, save_image, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--input", required=True, help="folder of input images")
    ap.add_argument("--output", required=True, help="folder for adversarial images")
    # quick overrides
    ap.add_argument("--eps_z", type=float, default=None)
    ap.add_argument("--num_steps", type=int, default=None)
    ap.add_argument("--lambda_p", type=float, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["runtime"]["seed"])
    device = cfg["runtime"]["device"]

    print("Loading detector and VAE...")
    detector = YOLOv8Wrapper(cfg["detector"]["weights"], device=device)
    vae = SDVAE(cfg["vae"]["model_id"], scale=cfg["vae"]["scale"], device=device)

    acfg = AttackConfig(
        eps_z=args.eps_z if args.eps_z is not None else cfg["attack"]["eps_z"],
        gamma=cfg["attack"]["gamma"],
        lambda_p=args.lambda_p if args.lambda_p is not None else cfg["attack"]["lambda_p"],
        lambda_r=cfg["attack"]["lambda_r"],
        lr=cfg["attack"]["lr"],
        num_steps=args.num_steps if args.num_steps is not None else cfg["attack"]["num_steps"],
        early_stop=cfg["attack"]["early_stop"],
        early_stop_margin=cfg["attack"]["early_stop_margin"],
        conf_thr=cfg["detector"]["conf_thr"],
        iou_nms=cfg["detector"]["iou_nms"],
    )
    attack = LatentObjectAttack(detector, vae, acfg)

    loader = ImageFolder(args.input, imgsz=cfg["detector"]["imgsz"])
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta: list[dict] = []
    t0 = time.time()
    for stem, x in tqdm(loader, total=len(loader), desc="Latent attack"):
        x = x.to(device)
        result = attack.attack(x)
        save_image(result.x_adv, out_dir / f"{stem}.png")
        meta.append({
            "stem": stem,
            "n_clean_detections": len(result.detections_clean),
            "classes_clean": result.classes_clean,
            "steps_taken": result.steps_taken,
            "final_p_max": result.history["p_max"][-1] if result.history["p_max"] else None,
            "final_L": result.history["L"][-1] if result.history["L"] else None,
        })
    elapsed = time.time() - t0

    with open(out_dir / "_attack_meta.json", "w") as f:
        json.dump({
            "config": vars(acfg),
            "elapsed_seconds": elapsed,
            "n_images": len(meta),
            "items": meta,
        }, f, indent=2)
    print(f"Done in {elapsed:.1f}s. Wrote {len(meta)} adversarial images to {out_dir}")


if __name__ == "__main__":
    main()
