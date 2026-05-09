"""Iso-budget sweep: latent attack vs PGD at matched perceptual budgets.

Sweeps eps for the latent attack and PGD on the first N frames of the dev set.
Records DFR_strict_proportional and masked LPIPS for each (attack, eps) config.

Budget grid
-----------
    LATENT_EPS = [0.25, 0.50, 1.00]      # eps_z values
    PGD_EPS    = [4/255, 8/255, 12/255]  # pixel L-inf

Dev set
-------
    data/images_50/img00001.jpg ... img00030.jpg  (--n_frames, default 30)

Output
------
    results/iso_budget/
        latent_eps0.25.json
        latent_eps0.50.json
        latent_eps1.00.json
        pgd_eps4.json
        pgd_eps8.json
        pgd_eps12.json
        summary.json     <- aggregated mean DFR + mean LPIPS per config

Usage:
    python scripts/run_iso_budget.py \\
        --config configs/phase2.yaml \\
        --data data/images_50 \\
        --n_frames 30 \\
        --output results/iso_budget

    Add --resume to skip configs whose output JSON already exists.

Expected runtime on Colab L4: ~4 hours (6 configs × 30 frames × ~80 steps).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.attack import AttackConfig, LatentObjectAttack
from src.detector import YOLOv8Wrapper
from src.eval.metrics import FrameDetections, per_frame_dfr
from src.losses import MaskedLPIPS
from src.masks import boxes_to_pixel_mask
from src.utils import load_config, load_image, set_seed
from src.vae import SDVAE

# ---------------------------------------------------------------------------
# Budget grid
# ---------------------------------------------------------------------------

LATENT_EPS: list[float] = [0.25, 0.50, 1.00]
PGD_EPS: list[float] = [4 / 255, 8 / 255, 12 / 255]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def dets_to_frame(dets: list) -> FrameDetections:
    """Convert list[Detection] (from detect_nms) to FrameDetections."""
    if not dets:
        return FrameDetections(
            boxes=torch.zeros((0, 4), dtype=torch.float32),
            scores=torch.zeros(0, dtype=torch.float32),
            classes=torch.zeros(0, dtype=torch.long),
        )
    boxes = torch.tensor([d.box for d in dets], dtype=torch.float32)
    scores = torch.tensor([d.score for d in dets], dtype=torch.float32)
    classes = torch.tensor([d.cls for d in dets], dtype=torch.long)
    return FrameDetections(boxes=boxes, scores=scores, classes=classes)


def run_pgd(
    x: torch.Tensor,
    detector: YOLOv8Wrapper,
    clean_dets: list,
    eps: float,
    pgd_steps: int,
    pgd_alpha: float,
    gamma: float,
    mask_restricted: bool,
    device: str,
) -> torch.Tensor:
    """Pixel-space PGD attack for object vanishing.

    Returns x_adv (1, 3, H, W) in [0, 1].
    """
    if not clean_dets:
        return x.clone()

    _, _, H, W = x.shape
    C_clean = sorted({d.cls for d in clean_dets})
    M: torch.Tensor | None = None
    if mask_restricted:
        from src.masks import boxes_to_pixel_mask
        M = boxes_to_pixel_mask(clean_dets, H=H, W=W, device=device)

    delta = torch.zeros_like(x, requires_grad=False)

    for _ in range(pgd_steps):
        delta.requires_grad_(True)
        x_adv = (x + delta).clamp(0.0, 1.0)
        if mask_restricted and M is not None:
            x_adv = M * x_adv + (1.0 - M) * x

        raw = detector.forward_raw(x_adv)
        class_conf = detector.class_confidence(raw)

        from src.losses import vanishing_loss
        L = vanishing_loss(class_conf, C_clean, gamma=gamma)
        L.backward()

        with torch.no_grad():
            grad_sign = delta.grad.sign()
            delta = delta + pgd_alpha * grad_sign
            if mask_restricted and M is not None:
                delta = delta * M
            delta = delta.clamp(-eps, eps)

    with torch.no_grad():
        x_adv = (x + delta).clamp(0.0, 1.0)
        if mask_restricted and M is not None:
            x_adv = M * x_adv + (1.0 - M) * x
    return x_adv.detach()


# ---------------------------------------------------------------------------
# Per-config sweep
# ---------------------------------------------------------------------------


def sweep_config(
    attack_type: str,
    eps: float,
    image_paths: list[Path],
    detector: YOLOv8Wrapper,
    vae: SDVAE | None,
    cfg: dict,
    device: str,
    lpips_loss: "MaskedLPIPS | None",
) -> list[dict[str, Any]]:
    """Run one (attack_type, eps) config on all frames; return per-frame records."""
    records: list[dict[str, Any]] = []
    conf_thr: float = cfg["detector"]["conf_thr"]
    iou_nms: float = cfg["detector"]["iou_nms"]
    gamma: float = cfg["attack"]["gamma"]
    _, _, H, W = (1, 3, 640, 640)  # fixed image size

    if attack_type == "latent":
        assert vae is not None
        acfg = AttackConfig(
            eps_z=eps,
            gamma=gamma,
            lambda_p=cfg["attack"]["lambda_p"],
            lambda_r=cfg["attack"]["lambda_r"],
            lr=cfg["attack"]["lr"],
            num_steps=cfg["attack"]["num_steps"],
            early_stop=cfg["attack"]["early_stop"],
            early_stop_margin=cfg["attack"]["early_stop_margin"],
            conf_thr=conf_thr,
            iou_nms=iou_nms,
            use_lpips=cfg["attack"].get("use_lpips", False),
            lpips_net=cfg["attack"].get("lpips_net", "alex"),
        )
        attack = LatentObjectAttack(detector, vae, acfg)

    for img_path in tqdm(image_paths, desc=f"{attack_type} eps={eps:.4f}", leave=False):
        x = load_image(img_path).to(device)

        # Clean detections
        clean_dets = detector.detect_nms(x, conf_thr=conf_thr, iou_thr=iou_nms)
        clean_fd = dets_to_frame(clean_dets)
        n_clean = len(clean_dets)

        if attack_type == "latent":
            result = attack.attack(x)
            x_adv = result.x_adv
        else:  # pgd
            pgd_cfg = cfg.get("baselines", {})
            x_adv = run_pgd(
                x=x,
                detector=detector,
                clean_dets=clean_dets,
                eps=eps,
                pgd_steps=pgd_cfg.get("pgd_steps", 50),
                pgd_alpha=pgd_cfg.get("pgd_alpha", 1 / 255),
                gamma=gamma,
                mask_restricted=pgd_cfg.get("mask_restricted", True),
                device=device,
            )

        # Adversarial detections
        adv_dets = detector.detect_nms(x_adv, conf_thr=conf_thr, iou_thr=iou_nms)
        adv_fd = dets_to_frame(adv_dets)
        n_adv = len(adv_dets)

        dfr = per_frame_dfr(clean_fd, adv_fd, conf_thr=0.0)

        # Masked LPIPS
        lpips_val: float | None = None
        if lpips_loss is not None and n_clean > 0:
            H_img, W_img = x.shape[2], x.shape[3]
            M = boxes_to_pixel_mask(clean_dets, H=H_img, W=W_img, device=device)
            with torch.no_grad():
                lpips_val = float(lpips_loss(x_adv, x, M).item())

        records.append({
            "stem": img_path.stem,
            "eps": eps,
            "dfr": dfr,
            "lpips": lpips_val,
            "n_clean": n_clean,
            "n_adv": n_adv,
        })

    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Iso-budget sweep (latent vs PGD).")
    ap.add_argument("--config", default="configs/phase2.yaml")
    ap.add_argument("--data", default="data/images_50",
                    help="Directory containing .jpg images")
    ap.add_argument("--n_frames", type=int, default=30,
                    help="Number of frames to evaluate (default 30)")
    ap.add_argument("--output", default="results/iso_budget",
                    help="Output directory for per-config JSONs and summary")
    ap.add_argument("--resume", action="store_true",
                    help="Skip configs whose output JSON already exists")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["runtime"]["seed"])
    device = cfg["runtime"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU.")
        device = "cpu"

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect image paths (sorted, first n_frames)
    all_paths = sorted(Path(args.data).glob("*.jpg"))[: args.n_frames]
    if not all_paths:
        raise FileNotFoundError(f"No .jpg images found in {args.data}")
    print(f"Using {len(all_paths)} frames from {args.data}")

    # Load models
    print("Loading detector...")
    detector = YOLOv8Wrapper(cfg["detector"]["weights"], device=device)

    print("Loading VAE...")
    vae_cfg = cfg["vae"]
    vae = SDVAE(
        model_id=vae_cfg["model_id"],
        scale=vae_cfg["scale"],
        device=device,
        finetuned_weights=vae_cfg.get("finetuned_weights"),
    )

    # LPIPS loss (optional)
    lpips_loss: MaskedLPIPS | None = None
    try:
        lpips_loss = MaskedLPIPS(net="alex", device=device)
        print("LPIPS loss enabled.")
    except Exception as e:
        print(f"WARNING: LPIPS not available ({e}). Skipping LPIPS metric.")

    # -----------------------------------------------------------------------
    # Latent attack sweep
    # -----------------------------------------------------------------------
    all_summaries: dict[str, dict] = {}

    for eps in LATENT_EPS:
        tag = f"latent_eps{eps:.2f}".rstrip("0").rstrip(".")
        # keep at most 2 decimal places, strip trailing zeros
        tag = f"latent_eps{eps}"
        out_path = out_dir / f"{tag}.json"

        if args.resume and out_path.exists():
            print(f"[resume] skipping {tag} (file exists)")
            with open(out_path) as f:
                records = json.load(f)
        else:
            print(f"Running latent attack eps_z={eps} ...")
            records = sweep_config(
                attack_type="latent",
                eps=eps,
                image_paths=all_paths,
                detector=detector,
                vae=vae,
                cfg=cfg,
                device=device,
                lpips_loss=lpips_loss,
            )
            with open(out_path, "w") as f:
                json.dump(records, f, indent=2)
            print(f"  Saved {out_path}")

        dfr_vals = [r["dfr"] for r in records if r["n_clean"] > 0]
        lpips_vals = [r["lpips"] for r in records
                      if r["lpips"] is not None and r["n_clean"] > 0]
        all_summaries[tag] = {
            "attack": "latent",
            "eps": eps,
            "mean_dfr_strict_proportional": float(np.mean(dfr_vals)) if dfr_vals else None,
            "mean_lpips": float(np.mean(lpips_vals)) if lpips_vals else None,
            "dfr_values": dfr_vals,
            "n_frames": len(records),
        }

    # -----------------------------------------------------------------------
    # PGD sweep
    # -----------------------------------------------------------------------
    pgd_eps_labels = {4 / 255: "pgd_eps4", 8 / 255: "pgd_eps8", 12 / 255: "pgd_eps12"}

    for eps in PGD_EPS:
        tag = pgd_eps_labels[eps]
        out_path = out_dir / f"{tag}.json"

        if args.resume and out_path.exists():
            print(f"[resume] skipping {tag} (file exists)")
            with open(out_path) as f:
                records = json.load(f)
        else:
            print(f"Running PGD attack eps_pixel={eps:.4f} ({round(eps * 255)}/255) ...")
            records = sweep_config(
                attack_type="pgd",
                eps=eps,
                image_paths=all_paths,
                detector=detector,
                vae=None,
                cfg=cfg,
                device=device,
                lpips_loss=lpips_loss,
            )
            with open(out_path, "w") as f:
                json.dump(records, f, indent=2)
            print(f"  Saved {out_path}")

        dfr_vals = [r["dfr"] for r in records if r["n_clean"] > 0]
        lpips_vals = [r["lpips"] for r in records
                      if r["lpips"] is not None and r["n_clean"] > 0]
        all_summaries[tag] = {
            "attack": "pgd",
            "eps": eps,
            "eps_over_255": round(eps * 255),
            "mean_dfr_strict_proportional": float(np.mean(dfr_vals)) if dfr_vals else None,
            "mean_lpips": float(np.mean(lpips_vals)) if lpips_vals else None,
            "dfr_values": dfr_vals,
            "n_frames": len(records),
        }

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    # Print table
    print(f"\n{'Config':<20} {'DFR':>8} {'LPIPS':>8}")
    print("-" * 40)
    for tag, s in all_summaries.items():
        dfr_str = f"{s['mean_dfr_strict_proportional']:.4f}" if s['mean_dfr_strict_proportional'] is not None else "N/A"
        lpips_str = f"{s['mean_lpips']:.4f}" if s['mean_lpips'] is not None else "N/A"
        print(f"{tag:<20} {dfr_str:>8} {lpips_str:>8}")


if __name__ == "__main__":
    main()
