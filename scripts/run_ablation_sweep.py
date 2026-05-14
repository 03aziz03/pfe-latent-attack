"""Ablation sweep — Option 1 (original VAE + LPIPS) and Option 2 (+ bilateral post-processing).

Runs the full latent attack ONCE per frame, then immediately applies bilateral
and guided filtering within the bounding-box mask M, so both Option 1 and
Option 2 metrics are obtained in a single Colab pass without saving raw images.

Output layout
-------------
    results/ablation/
        option1_latent_eps0.25.json     raw attack (Option 1)
        option1_latent_eps0.5.json
        option1_latent_eps1.0.json
        option2_latent_eps0.25.json     bilateral post-processed (Option 2)
        option2_latent_eps0.5.json
        option2_latent_eps1.0.json
        pgd_eps4.json
        pgd_eps8.json
        pgd_eps12.json
        summary.json                    mean DFR + mean LPIPS for every config

JSON record schema (per frame)
-------------------------------
    {
        "stem":    "img00001",
        "eps":     0.5,
        "variant": "option1" | "option2",
        "filter":  "none" | "bilateral" | "guided",
        "dfr":     float,          # (n_clean - n_adv) / n_clean
        "lpips":   float,          # masked LPIPS (AlexNet)
        "n_clean": int,
        "n_adv":   int,
        "steps":   int             # Adam steps taken (early-stop indicator)
    }

Post-processing detail
----------------------
Option 2 applies a masked bilateral filter:
    x_pp = M * bilateralFilter(x_adv) + (1 - M) * x_clean
This smooths high-frequency adversarial artefacts inside bounding boxes while
leaving background pixels untouched. Guided filter is applied if
opencv-contrib is available (ximgproc); otherwise falls back to bilateral.

Usage
-----
    python scripts/run_ablation_sweep.py \\
        --config  configs/phase2_option1.yaml \\
        --data    data/images_50 \\
        --n_frames 30 \\
        --output  results/ablation

    Add --resume to skip configs whose JSON already exists.

Expected runtime on Colab L4: ~3 h (3 eps × 30 frames × 80 steps).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.attack import AttackConfig, LatentObjectAttack
from src.detector import YOLOv8Wrapper
from src.eval.metrics import (
    FrameDetections,
    per_frame_dfr,
    per_frame_asr,
    per_frame_map_drop,
)
from src.losses import MaskedLPIPS
from src.masks import boxes_to_pixel_mask
from src.utils import load_config, load_image, set_seed
from src.vae import SDVAE

# ---------------------------------------------------------------------------
# Budget grids
# ---------------------------------------------------------------------------

LATENT_EPS: list[float] = [0.25, 0.50, 1.00]
PGD_EPS:    list[float] = [4 / 255, 8 / 255, 12 / 255]

# Bilateral filter defaults (tuned for 640×640 surveillance frames)
BILATERAL_D          = 9     # filter diameter
BILATERAL_SIGMA_COL  = 50    # range kernel sigma (colour similarity)
BILATERAL_SIGMA_SPC  = 50    # spatial kernel sigma


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def dets_to_frame(dets: list) -> FrameDetections:
    if not dets:
        return FrameDetections(
            boxes=torch.zeros((0, 4), dtype=torch.float32),
            scores=torch.zeros(0, dtype=torch.float32),
            classes=torch.zeros(0, dtype=torch.long),
        )
    return FrameDetections(
        boxes=torch.tensor([d.box   for d in dets], dtype=torch.float32),
        scores=torch.tensor([d.score for d in dets], dtype=torch.float32),
        classes=torch.tensor([d.cls  for d in dets], dtype=torch.long),
    )


# ---------------------------------------------------------------------------
# Post-processing filters
# ---------------------------------------------------------------------------


def tensor_to_uint8(t: torch.Tensor) -> np.ndarray:
    """(1, 3, H, W) float32 in [0,1] → (H, W, 3) uint8."""
    return (t[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)


def uint8_to_tensor(arr: np.ndarray, device: str) -> torch.Tensor:
    """(H, W, 3) uint8 → (1, 3, H, W) float32 in [0,1]."""
    return torch.from_numpy(arr).float().div(255).permute(2, 0, 1).unsqueeze(0).to(device)


def bilateral_filter_masked(
    x_adv:   torch.Tensor,   # (1, 3, H, W)
    x_clean: torch.Tensor,   # (1, 3, H, W)
    M:       torch.Tensor,   # (1, 1, H, W)  binary mask
    d:       int   = BILATERAL_D,
    sc:      float = BILATERAL_SIGMA_COL,
    ss:      float = BILATERAL_SIGMA_SPC,
) -> torch.Tensor:
    """Apply bilateral filter to x_adv and paste within mask M."""
    device  = x_adv.device
    x_np    = tensor_to_uint8(x_adv)
    x_filt  = cv2.bilateralFilter(x_np, d=d, sigmaColor=sc, sigmaSpace=ss)
    x_filt_t = uint8_to_tensor(x_filt, str(device))
    return (M * x_filt_t + (1 - M) * x_clean).clamp(0, 1)


def guided_filter_masked(
    x_adv:   torch.Tensor,
    x_clean: torch.Tensor,
    M:       torch.Tensor,
    radius:  int   = 8,
    eps:     float = 0.01,
) -> torch.Tensor:
    """Apply guided filter (guide = x_clean) within mask M.

    Falls back to bilateral if opencv-contrib (ximgproc) is unavailable.
    """
    try:
        device   = x_adv.device
        guide_np = tensor_to_uint8(x_clean)
        src_np   = tensor_to_uint8(x_adv)
        x_filt   = cv2.ximgproc.guidedFilter(
            guide=guide_np, src=src_np, radius=radius, eps=eps
        )
        x_filt_t = uint8_to_tensor(x_filt, str(device))
        return (M * x_filt_t + (1 - M) * x_clean).clamp(0, 1)
    except AttributeError:
        # ximgproc not available — fall back to bilateral
        return bilateral_filter_masked(x_adv, x_clean, M)


# ---------------------------------------------------------------------------
# PGD baseline (pixel-space)
# ---------------------------------------------------------------------------


def run_pgd(
    x:              torch.Tensor,
    detector:       YOLOv8Wrapper,
    clean_dets:     list,
    eps:            float,
    pgd_steps:      int,
    pgd_alpha:      float,
    gamma:          float,
    mask_restricted: bool,
    device:         str,
) -> torch.Tensor:
    if not clean_dets:
        return x.clone()

    _, _, H, W = x.shape
    C_clean = sorted({d.cls for d in clean_dets})
    M: torch.Tensor | None = None
    if mask_restricted:
        M = boxes_to_pixel_mask(clean_dets, H=H, W=W, device=device)

    delta = torch.zeros_like(x, requires_grad=False)
    for _ in range(pgd_steps):
        delta.requires_grad_(True)
        x_adv = (x + delta).clamp(0.0, 1.0)
        if mask_restricted and M is not None:
            x_adv = M * x_adv + (1.0 - M) * x

        raw        = detector.forward_raw(x_adv)
        class_conf = detector.class_confidence(raw)

        from src.losses import vanishing_loss
        L = vanishing_loss(class_conf, C_clean, gamma=gamma)
        L.backward()

        with torch.no_grad():
            delta = delta + pgd_alpha * delta.grad.sign()
            if mask_restricted and M is not None:
                delta = delta * M
            delta = delta.clamp(-eps, eps)

    with torch.no_grad():
        x_adv = (x + delta).clamp(0.0, 1.0)
        if mask_restricted and M is not None:
            x_adv = M * x_adv + (1.0 - M) * x
    return x_adv.detach()


# ---------------------------------------------------------------------------
# Core sweep (latent, single eps)
# ---------------------------------------------------------------------------


def sweep_latent(
    eps:          float,
    image_paths:  list[Path],
    detector:     YOLOv8Wrapper,
    vae:          SDVAE,
    cfg:          dict,
    device:       str,
    lpips_loss:   MaskedLPIPS,
) -> tuple[list[dict], list[dict]]:
    """Run the latent attack once; return (option1_records, option2_records)."""
    opt1_records: list[dict[str, Any]] = []
    opt2_records: list[dict[str, Any]] = []

    conf_thr = cfg["detector"]["conf_thr"]
    iou_nms  = cfg["detector"]["iou_nms"]
    gamma    = cfg["attack"]["gamma"]

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

    for img_path in tqdm(image_paths,
                         desc=f"latent eps={eps:.2f}",
                         leave=False):
        x          = load_image(img_path).to(device)
        clean_dets = detector.detect_nms(x, conf_thr=conf_thr, iou_thr=iou_nms)
        n_clean    = len(clean_dets)

        # ---- run attack ----
        result  = attack.attack(x)
        x_adv   = result.x_adv
        M       = result.M          # (1,1,H,W)

        adv_dets = detector.detect_nms(x_adv, conf_thr=conf_thr, iou_thr=iou_nms)
        n_adv    = len(adv_dets)

        clean_fd = dets_to_frame(clean_dets)
        adv_fd1  = dets_to_frame(adv_dets)

        dfr1 = per_frame_dfr(clean_fd, adv_fd1, conf_thr=0.0)
        asr1 = per_frame_asr(clean_fd, adv_fd1, conf_thr=0.0)
        try:
            map_drop1 = per_frame_map_drop(clean_fd, adv_fd1)
        except Exception:
            map_drop1 = None

        H, W = x.shape[2], x.shape[3]
        if n_clean > 0:
            M_pix = boxes_to_pixel_mask(clean_dets, H=H, W=W, device=device)
        else:
            M_pix = torch.zeros(1, 1, H, W, device=device)

        with torch.no_grad():
            lpips1 = float(lpips_loss(x_adv, x, M_pix).item()) if n_clean > 0 else None

        opt1_records.append({
            "stem":     img_path.stem,
            "eps":      eps,
            "variant":  "option1",
            "filter":   "none",
            "dfr":      dfr1,
            "dfr_strict": int(n_adv == 0 and n_clean > 0),
            "asr":      int(asr1),
            "map_drop": map_drop1,
            "lpips":    lpips1,
            "n_clean":  n_clean,
            "n_adv":    n_adv,
            "steps":    result.steps_taken,
        })

        # ---- Option 2a: bilateral filter ----
        x_bil    = bilateral_filter_masked(x_adv, x, M_pix)
        bil_dets = detector.detect_nms(x_bil, conf_thr=conf_thr, iou_thr=iou_nms)
        n_bil    = len(bil_dets)

        adv_fd2  = dets_to_frame(bil_dets)
        dfr2     = per_frame_dfr(clean_fd, adv_fd2, conf_thr=0.0)
        asr2     = per_frame_asr(clean_fd, adv_fd2, conf_thr=0.0)
        try:
            map_drop2 = per_frame_map_drop(clean_fd, adv_fd2)
        except Exception:
            map_drop2 = None

        with torch.no_grad():
            lpips2 = float(lpips_loss(x_bil, x, M_pix).item()) if n_clean > 0 else None

        opt2_records.append({
            "stem":     img_path.stem,
            "eps":      eps,
            "variant":  "option2",
            "filter":   "bilateral",
            "dfr":      dfr2,
            "dfr_strict": int(n_bil == 0 and n_clean > 0),
            "asr":      int(asr2),
            "map_drop": map_drop2,
            "lpips":    lpips2,
            "n_clean":  n_clean,
            "n_adv":    n_bil,
            "steps":    result.steps_taken,
        })

    return opt1_records, opt2_records


# ---------------------------------------------------------------------------
# Core sweep (PGD)
# ---------------------------------------------------------------------------


def sweep_pgd(
    eps:          float,
    image_paths:  list[Path],
    detector:     YOLOv8Wrapper,
    cfg:          dict,
    device:       str,
    lpips_loss:   MaskedLPIPS,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    conf_thr = cfg["detector"]["conf_thr"]
    iou_nms  = cfg["detector"]["iou_nms"]
    pgd_cfg  = cfg.get("baselines", {})
    gamma    = cfg["attack"]["gamma"]

    for img_path in tqdm(image_paths,
                         desc=f"pgd eps={eps:.4f}",
                         leave=False):
        x          = load_image(img_path).to(device)
        clean_dets = detector.detect_nms(x, conf_thr=conf_thr, iou_thr=iou_nms)
        n_clean    = len(clean_dets)

        x_adv    = run_pgd(
            x=x, detector=detector, clean_dets=clean_dets,
            eps=eps,
            pgd_steps=pgd_cfg.get("pgd_steps", 50),
            pgd_alpha=pgd_cfg.get("pgd_alpha", 1 / 255),
            gamma=gamma,
            mask_restricted=pgd_cfg.get("mask_restricted", True),
            device=device,
        )
        adv_dets = detector.detect_nms(x_adv, conf_thr=conf_thr, iou_thr=iou_nms)
        n_adv    = len(adv_dets)

        clean_fd = dets_to_frame(clean_dets)
        adv_fd   = dets_to_frame(adv_dets)

        dfr  = per_frame_dfr(clean_fd, adv_fd, conf_thr=0.0)
        asr  = per_frame_asr(clean_fd, adv_fd, conf_thr=0.0)
        try:
            map_drop = per_frame_map_drop(clean_fd, adv_fd)
        except Exception:
            map_drop = None

        H, W = x.shape[2], x.shape[3]
        M_pix = boxes_to_pixel_mask(clean_dets, H=H, W=W, device=device) if n_clean > 0 \
                else torch.zeros(1, 1, H, W, device=device)
        with torch.no_grad():
            lpips_val = float(lpips_loss(x_adv, x, M_pix).item()) if n_clean > 0 else None

        records.append({
            "stem":     img_path.stem,
            "eps":      eps,
            "variant":  "pgd",
            "filter":   "none",
            "dfr":      dfr,
            "dfr_strict": int(n_adv == 0 and n_clean > 0),
            "asr":      int(asr),
            "map_drop": map_drop,
            "lpips":    lpips_val,
            "n_clean":  n_clean,
            "n_adv":    n_adv,
            "steps":    pgd_cfg.get("pgd_steps", 50),
        })

    return records


# ---------------------------------------------------------------------------
# Aggregate summary
# ---------------------------------------------------------------------------


def summarise(tag: str, records: list[dict], attack: str, eps: float) -> dict:
    import math

    valid = [r for r in records if r["n_clean"] > 0]
    n = len(valid)

    def mean(vals):
        v = [x for x in vals if x is not None]
        return float(sum(v) / len(v)) if v else None

    def se(vals):
        v = [x for x in vals if x is not None]
        if len(v) < 2:
            return 0.0
        m = sum(v) / len(v)
        return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1) / len(v))

    dfr_vals      = [r["dfr"]       for r in valid]
    asr_vals      = [r.get("asr")   for r in valid]
    map_drop_vals = [r.get("map_drop") for r in valid]
    lpips_vals    = [r["lpips"]     for r in valid if r.get("lpips") is not None]

    # DFR strict: fraction of frames where all detections suppressed (n_adv == 0)
    dfr_strict_vals = [r.get("dfr_strict", 0) for r in valid]

    return {
        "tag":            tag,
        "attack":         attack,
        "eps":            eps,
        "n_frames":       len(records),
        "n_valid":        n,
        # --- DFR proportionnel ---
        "mean_dfr":       mean(dfr_vals),
        "se_dfr":         se(dfr_vals),
        "dfr_pos":        sum(1 for v in dfr_vals if v > 0),
        "dfr_neg":        sum(1 for v in dfr_vals if v < 0),
        "dfr_zero":       sum(1 for v in dfr_vals if v == 0),
        # --- DFR strict (fraction frames avec n_adv==0) ---
        "dfr_strict_rate": float(sum(dfr_strict_vals) / n) if n else None,
        "dfr_strict_count": sum(dfr_strict_vals),
        # --- ASR (toutes classes absentes) ---
        "mean_asr":       mean(asr_vals),
        "asr_count":      sum(1 for v in asr_vals if v),
        # --- mAP drop ---
        "mean_map_drop":  mean(map_drop_vals),
        "se_map_drop":    se([v for v in map_drop_vals if v is not None]),
        # --- LPIPS ---
        "mean_lpips":     mean(lpips_vals),
        "se_lpips":       se(lpips_vals),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Option 1 + Option 2 ablation sweep in one Colab pass."
    )
    ap.add_argument("--config",    default="configs/phase2_option1.yaml")
    ap.add_argument("--data",      default="data/images_50")
    ap.add_argument("--n_frames",  type=int, default=30)
    ap.add_argument("--output",    default="results/ablation")
    ap.add_argument("--resume",    action="store_true",
                    help="Skip configs whose output JSON already exists.")
    args = ap.parse_args()

    cfg    = load_config(args.config)
    device = cfg["runtime"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    set_seed(cfg["runtime"]["seed"])

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(Path(args.data).glob("*.jpg"))[: args.n_frames]
    if not image_paths:
        raise FileNotFoundError(f"No .jpg images in {args.data}")
    print(f"Frames: {len(image_paths)}  |  device: {device}")

    print("Loading detector …")
    detector = YOLOv8Wrapper(cfg["detector"]["weights"], device=device)

    print("Loading VAE (original SD-VAE, no fine-tuned weights) …")
    vae_cfg = cfg["vae"]
    vae = SDVAE(
        model_id=vae_cfg["model_id"],
        scale=vae_cfg["scale"],
        device=device,
        finetuned_weights=vae_cfg.get("finetuned_weights"),  # None for Option 1
    )

    print("Loading LPIPS …")
    lpips_loss = MaskedLPIPS(net="alex", device=device)

    all_summaries: dict[str, dict] = {}

    # -----------------------------------------------------------------------
    # Latent sweep — Option 1 + Option 2
    # -----------------------------------------------------------------------
    for eps in LATENT_EPS:
        tag1 = f"option1_latent_eps{eps}"
        tag2 = f"option2_latent_eps{eps}"
        out1 = out_dir / f"{tag1}.json"
        out2 = out_dir / f"{tag2}.json"

        if args.resume and out1.exists() and out2.exists():
            print(f"[resume] skipping eps={eps}")
            with open(out1) as f: opt1_rec = json.load(f)
            with open(out2) as f: opt2_rec = json.load(f)
        else:
            print(f"\nLatent attack  eps_z={eps} …")
            opt1_rec, opt2_rec = sweep_latent(
                eps=eps,
                image_paths=image_paths,
                detector=detector,
                vae=vae,
                cfg=cfg,
                device=device,
                lpips_loss=lpips_loss,
            )
            with open(out1, "w") as f: json.dump(opt1_rec, f, indent=2)
            with open(out2, "w") as f: json.dump(opt2_rec, f, indent=2)
            print(f"  Saved {out1.name}  +  {out2.name}")

        all_summaries[tag1] = summarise(tag1, opt1_rec, "latent_option1", eps)
        all_summaries[tag2] = summarise(tag2, opt2_rec, "latent_option2_bilateral", eps)

    # -----------------------------------------------------------------------
    # PGD baseline sweep
    # -----------------------------------------------------------------------
    pgd_labels = {4/255: "pgd_eps4", 8/255: "pgd_eps8", 12/255: "pgd_eps12"}
    for eps in PGD_EPS:
        tag     = pgd_labels[eps]
        out_pgd = out_dir / f"{tag}.json"

        if args.resume and out_pgd.exists():
            print(f"[resume] skipping {tag}")
            with open(out_pgd) as f: pgd_rec = json.load(f)
        else:
            print(f"\nPGD  eps={eps:.4f} ({round(eps*255)}/255) …")
            pgd_rec = sweep_pgd(
                eps=eps,
                image_paths=image_paths,
                detector=detector,
                cfg=cfg,
                device=device,
                lpips_loss=lpips_loss,
            )
            with open(out_pgd, "w") as f: json.dump(pgd_rec, f, indent=2)
            print(f"  Saved {out_pgd.name}")

        all_summaries[tag] = summarise(tag, pgd_rec, "pgd", eps)

    # -----------------------------------------------------------------------
    # Summary JSON + console table
    # -----------------------------------------------------------------------
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_summaries, f, indent=2)

    print("\n" + "="*95)
    print(f"{'Config':<32} {'DFR':>8} {'±SE':>6} {'DFR_strict':>10} {'ASR':>6} {'mAP_drop':>9} {'LPIPS':>7}")
    print("-"*95)
    for tag, s in all_summaries.items():
        dfr_s    = f"{s['mean_dfr']:+.4f}"        if s.get("mean_dfr")        is not None else "   N/A"
        se_s     = f"{s['se_dfr']:.4f}"            if s.get("se_dfr")          is not None else "  N/A"
        strict_s = f"{s['dfr_strict_rate']:.3f}"   if s.get("dfr_strict_rate") is not None else "   N/A"
        asr_s    = f"{s['mean_asr']:.3f}"          if s.get("mean_asr")        is not None else "  N/A"
        map_s    = f"{s['mean_map_drop']:.4f}"     if s.get("mean_map_drop")   is not None else "    N/A"
        lpips_s  = f"{s['mean_lpips']:.4f}"        if s.get("mean_lpips")      is not None else "   N/A"
        print(f"{tag:<32} {dfr_s:>8} {se_s:>6} {strict_s:>10} {asr_s:>6} {map_s:>9} {lpips_s:>7}")
    print("="*95)
    print(f"\nSummary → {summary_path}")
    print("\nLégende :")
    print("  DFR        = 1 - n_adv/n_clean  (proportionnel, par frame)")
    print("  DFR_strict = fraction frames où n_adv == 0  (suppression totale)")
    print("  ASR        = fraction frames où toutes les classes clean disparaissent")
    print("  mAP_drop   = 1 - mAP@0.5  (clean detections comme pseudo-GT)")


if __name__ == "__main__":
    main()
