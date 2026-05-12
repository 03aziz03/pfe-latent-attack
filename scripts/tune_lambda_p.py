"""Quick lambda_p tuning on a small frame subset.

Sweeps lambda_p over a log-scale grid on N frames (default 5) using the
Option 1 config (original SD-VAE + LPIPS loss). Prints a DFR / LPIPS
trade-off table to help choose the operating lambda_p before running the
full iso-budget sweep.

Usage
-----
    python scripts/tune_lambda_p.py \\
        --config configs/phase2_option1.yaml \\
        --data   data/images_50 \\
        --n_frames 5 \\
        --eps_z  0.50

Output (stdout):
    lambda_p  | mean_DFR | mean_LPIPS | DFR_pos | DFR_neg
    ----------+----------+------------+---------+--------
    0.0005    |  0.0812  |   0.2341   |   4     |   1
    0.0010    |  0.0735  |   0.1984   |   3     |   1
    ...

Recommendation: pick the largest lambda_p where mean_DFR stays close
to the unconstrained (lambda_p=0) baseline.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.attack import AttackConfig, LatentObjectAttack
from src.detector import YOLOv8Wrapper
from src.eval.metrics import FrameDetections, per_frame_dfr
from src.losses import MaskedLPIPS
from src.masks import boxes_to_pixel_mask
from src.utils import load_config, load_image, set_seed
from src.vae import SDVAE

# Log-scale sweep
LAMBDA_P_GRID = [0.0, 0.0005, 0.001, 0.005, 0.01, 0.05]


def dets_to_frame(dets: list) -> FrameDetections:
    if not dets:
        return FrameDetections(
            boxes=torch.zeros((0, 4), dtype=torch.float32),
            scores=torch.zeros(0, dtype=torch.float32),
            classes=torch.zeros(0, dtype=torch.long),
        )
    boxes   = torch.tensor([d.box   for d in dets], dtype=torch.float32)
    scores  = torch.tensor([d.score for d in dets], dtype=torch.float32)
    classes = torch.tensor([d.cls   for d in dets], dtype=torch.long)
    return FrameDetections(boxes=boxes, scores=scores, classes=classes)


def run_one(
    lp: float,
    image_paths: list[Path],
    detector: YOLOv8Wrapper,
    vae: SDVAE,
    cfg: dict,
    device: str,
    lpips_metric: MaskedLPIPS,
    eps_z: float,
) -> dict:
    """Run attack with given lambda_p on all frames; return aggregated stats."""
    acfg = AttackConfig(
        eps_z=eps_z,
        gamma=cfg["attack"]["gamma"],
        lambda_p=lp,
        lambda_r=cfg["attack"]["lambda_r"],
        lr=cfg["attack"]["lr"],
        num_steps=cfg["attack"]["num_steps"],
        early_stop=cfg["attack"]["early_stop"],
        early_stop_margin=cfg["attack"]["early_stop_margin"],
        conf_thr=cfg["detector"]["conf_thr"],
        iou_nms=cfg["detector"]["iou_nms"],
        use_lpips=(lp > 0),          # disable LPIPS term for lambda_p=0 baseline
        lpips_net=cfg["attack"].get("lpips_net", "alex"),
    )
    attack = LatentObjectAttack(detector, vae, acfg)

    dfr_vals, lpips_vals = [], []

    for img_path in image_paths:
        x = load_image(img_path).to(device)
        clean_dets = detector.detect_nms(
            x, conf_thr=cfg["detector"]["conf_thr"],
            iou_thr=cfg["detector"]["iou_nms"],
        )
        if not clean_dets:
            continue

        result   = attack.attack(x)
        x_adv    = result.x_adv
        adv_dets = detector.detect_nms(
            x_adv, conf_thr=cfg["detector"]["conf_thr"],
            iou_thr=cfg["detector"]["iou_nms"],
        )

        clean_fd = dets_to_frame(clean_dets)
        adv_fd   = dets_to_frame(adv_dets)
        dfr = per_frame_dfr(clean_fd, adv_fd, conf_thr=0.0)
        dfr_vals.append(dfr)

        H, W = x.shape[2], x.shape[3]
        M = boxes_to_pixel_mask(clean_dets, H=H, W=W, device=device)
        with torch.no_grad():
            lp_val = float(lpips_metric(x_adv, x, M).item())
        lpips_vals.append(lp_val)

    n = len(dfr_vals)
    mean_dfr   = sum(dfr_vals)   / n if n else float("nan")
    mean_lpips = sum(lpips_vals) / n if n else float("nan")
    pos = sum(1 for v in dfr_vals if v > 0)
    neg = sum(1 for v in dfr_vals if v < 0)

    return {
        "lambda_p":   lp,
        "mean_dfr":   mean_dfr,
        "mean_lpips": mean_lpips,
        "dfr_pos":    pos,
        "dfr_neg":    neg,
        "n_frames":   n,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Tune lambda_p for Option 1.")
    ap.add_argument("--config",    default="configs/phase2_option1.yaml")
    ap.add_argument("--data",      default="data/images_50")
    ap.add_argument("--n_frames",  type=int,   default=5)
    ap.add_argument("--eps_z",     type=float, default=0.50)
    args = ap.parse_args()

    cfg    = load_config(args.config)
    device = cfg["runtime"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    set_seed(cfg["runtime"]["seed"])

    image_paths = sorted(Path(args.data).glob("*.jpg"))[: args.n_frames]
    if not image_paths:
        raise FileNotFoundError(f"No images in {args.data}")
    print(f"Tuning lambda_p on {len(image_paths)} frames  |  eps_z={args.eps_z}\n")

    print("Loading models …")
    detector = YOLOv8Wrapper(cfg["detector"]["weights"], device=device)
    vae_cfg  = cfg["vae"]
    vae = SDVAE(
        model_id=vae_cfg["model_id"],
        scale=vae_cfg["scale"],
        device=device,
        finetuned_weights=vae_cfg.get("finetuned_weights"),   # None for Option 1
    )
    lpips_metric = MaskedLPIPS(net="alex", device=device)

    # -----------------------------------------------------------------------
    # Sweep
    # -----------------------------------------------------------------------
    results = []
    for lp in LAMBDA_P_GRID:
        print(f"  lambda_p = {lp:.4f} …", end="  ", flush=True)
        row = run_one(lp, image_paths, detector, vae, cfg, device, lpips_metric, args.eps_z)
        results.append(row)
        print(f"DFR={row['mean_dfr']:+.4f}  LPIPS={row['mean_lpips']:.4f}  "
              f"pos={row['dfr_pos']}  neg={row['dfr_neg']}")

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    print("\n" + "="*65)
    print(f"{'lambda_p':>10} | {'mean_DFR':>10} | {'mean_LPIPS':>10} | "
          f"{'pos/total':>10} | {'neg/total':>10}")
    print("-"*65)
    n_fr = results[0]["n_frames"]
    for r in results:
        print(f"{r['lambda_p']:>10.4f} | {r['mean_dfr']:>+10.4f} | "
              f"{r['mean_lpips']:>10.4f} | "
              f"{r['dfr_pos']:>4}/{n_fr:<5} | "
              f"{r['dfr_neg']:>4}/{n_fr:<5}")
    print("="*65)

    # Recommendation: largest lambda_p within 10% of unconstrained DFR
    base_dfr = results[0]["mean_dfr"]   # lambda_p=0
    thresh   = base_dfr * 0.90
    best = results[0]
    for r in results:
        if r["mean_dfr"] >= thresh:
            best = r
    print(f"\nRecommended lambda_p = {best['lambda_p']:.4f}  "
          f"(DFR={best['mean_dfr']:+.4f}, within 10% of unconstrained "
          f"DFR={base_dfr:+.4f})")
    print("Update configs/phase2_option1.yaml  →  lambda_p:", best["lambda_p"])


if __name__ == "__main__":
    main()
