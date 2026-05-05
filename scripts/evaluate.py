"""Evaluate adversarial images against the clean set.

Computes:

* Detection Failure Rate (DFR):
        per-instance fraction of clean detections that no longer survive.

* Attack Success Rate (ASR):
        fraction of images where at least one originally-detected object
        disappears.

* mAP drop:
        difference in mean detection score across the matched clean set.
        (NOTE: a true COCO mAP requires ground-truth annotations. With a
        plain folder loader we report a *self-mAP-drop* against YOLOv8's
        clean predictions, which is the standard surrogate for adversarial
        evaluation when no GT is available. Switch to a COCO loader to get
        the GT-based mAP.)

* Mean LPIPS_mask / PSNR_mask placeholders:
        we report masked L2 and PSNR (mask region only) instead of LPIPS to
        keep dependencies minimal.

Usage:
    python scripts/evaluate.py \
        --clean data/images \
        --adv   results/adv_latent \
        --out   results/metrics_latent.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.detector import YOLOv8Wrapper
from src.data import ImageFolder
from src.masks import boxes_to_pixel_mask
from src.utils import (
    load_config, set_seed, load_image, iou_xyxy, Detection,
)


def match_detections(clean: list[Detection],
                     adv: list[Detection],
                     iou_thr: float = 0.5) -> int:
    """Greedy 1-to-1 matching by IoU + class. Returns # clean dets matched."""
    if not clean or not adv:
        return 0
    c_box = torch.tensor([d.box for d in clean])
    a_box = torch.tensor([d.box for d in adv])
    ious = iou_xyxy(c_box, a_box)                         # (Nc, Na)
    matched = 0
    used_a = set()
    for i, d in enumerate(clean):
        # candidate j with same class and best IoU
        best_j, best_iou = -1, 0.0
        for j, dj in enumerate(adv):
            if j in used_a or dj.cls != d.cls:
                continue
            if ious[i, j] > best_iou:
                best_j, best_iou = j, float(ious[i, j])
        if best_j >= 0 and best_iou >= iou_thr:
            matched += 1
            used_a.add(best_j)
    return matched


def masked_psnr(x_adv: torch.Tensor, x: torch.Tensor, M: torch.Tensor) -> float:
    diff = M * (x_adv - x)
    n = M.sum() * x.shape[1] + 1e-8
    mse = (diff.pow(2).sum() / n).item()
    if mse <= 1e-12:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--clean", required=True, help="folder of clean images")
    ap.add_argument("--adv", required=True, help="folder of adversarial images")
    ap.add_argument("--out", required=True, help="JSON output path")
    ap.add_argument("--iou-thr", type=float, default=0.5)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["runtime"]["seed"])
    device = cfg["runtime"]["device"]

    detector = YOLOv8Wrapper(cfg["detector"]["weights"], device=device)
    clean_loader = ImageFolder(args.clean, imgsz=cfg["detector"]["imgsz"])
    adv_root = Path(args.adv)

    n_clean_total = 0
    n_kept_total = 0
    n_imgs_with_clean = 0
    n_imgs_attacked = 0       # at least one clean detection disappeared
    sum_clean_score = 0.0
    sum_adv_score = 0.0
    score_count = 0
    psnr_vals: list[float] = []
    masked_l2_vals: list[float] = []

    per_image: list[dict] = []
    for stem, x in tqdm(clean_loader, total=len(clean_loader), desc="Eval"):
        x = x.to(device)
        D_clean = detector.detect_nms(
            x, conf_thr=cfg["detector"]["conf_thr"],
            iou_thr=cfg["detector"]["iou_nms"]
        )

        adv_path = adv_root / f"{stem}.png"
        if not adv_path.exists():
            adv_path = adv_root / f"{stem}.jpg"
        if not adv_path.exists():
            per_image.append({"stem": stem, "skipped": "no adv image"})
            continue
        x_adv = load_image(adv_path, imgsz=cfg["detector"]["imgsz"]).to(device)

        D_adv = detector.detect_nms(
            x_adv, conf_thr=cfg["detector"]["conf_thr"],
            iou_thr=cfg["detector"]["iou_nms"]
        )

        if D_clean:
            n_imgs_with_clean += 1
            kept = match_detections(D_clean, D_adv, iou_thr=args.iou_thr)
            n_clean_total += len(D_clean)
            n_kept_total += kept
            if kept < len(D_clean):
                n_imgs_attacked += 1
            sum_clean_score += sum(d.score for d in D_clean)
            sum_adv_score += sum(d.score for d in D_adv)
            score_count += len(D_clean)

        # masked perceptual quality
        if D_clean:
            M = boxes_to_pixel_mask(D_clean, H=x.shape[-2], W=x.shape[-1], device=device)
            psnr_vals.append(masked_psnr(x_adv, x, M))
            with torch.no_grad():
                diff = M * (x_adv - x)
                denom = M.sum() * x.shape[1] + 1e-8
                masked_l2_vals.append((diff.pow(2).sum() / denom).item())

        per_image.append({
            "stem": stem,
            "n_clean": len(D_clean),
            "n_adv": len(D_adv),
            "n_kept": match_detections(D_clean, D_adv, iou_thr=args.iou_thr) if D_clean else 0,
        })

    # ----- aggregate -----
    dfr = 1.0 - (n_kept_total / n_clean_total) if n_clean_total else 0.0
    asr = (n_imgs_attacked / n_imgs_with_clean) if n_imgs_with_clean else 0.0
    mean_score_drop = (sum_clean_score - sum_adv_score) / score_count if score_count else 0.0

    metrics = {
        "n_images_with_clean_detections": n_imgs_with_clean,
        "n_clean_detections": n_clean_total,
        "n_kept_after_attack": n_kept_total,
        "DFR": dfr,
        "ASR": asr,
        "mean_confidence_drop": mean_score_drop,
        "mean_PSNR_mask_dB": (sum(psnr_vals) / len(psnr_vals)) if psnr_vals else None,
        "mean_masked_L2": (sum(masked_l2_vals) / len(masked_l2_vals)) if masked_l2_vals else None,
        "iou_threshold_for_match": args.iou_thr,
        "n_per_image": len(per_image),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"summary": metrics, "per_image": per_image}, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
