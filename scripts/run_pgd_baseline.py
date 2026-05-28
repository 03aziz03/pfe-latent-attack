"""PGD-mask baseline evaluation — same metric pipeline as run_phase3_ablation.py.

Runs PGD-mask (mask_restricted=True, eps=8/255, 50 steps) on a folder of frames
and computes DFR_prop, DFR_strict, ASR, LPIPS — identical to the latent attack
metric pipeline so results are directly comparable in Table 4.18.

Usage:
    python scripts/run_pgd_baseline.py \\
        --data   data/multiclass/MVI_20032 \\
        --n_frames 70 \\
        --output results/multiclass/MVI_20032 \\
        --resume
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.detector import YOLOv8Wrapper
from src.masks import boxes_to_pixel_mask
from src.losses import vanishing_loss
from src.eval.metrics import (FrameDetections, per_frame_dfr, per_frame_asr,
                               per_frame_map_drop)

try:
    import lpips as _lpips_lib
    _LPIPS_OK = True
except ImportError:
    _LPIPS_OK = False

try:
    from torchmetrics.detection import MeanAveragePrecision  # noqa
    _TORCHMETRICS_OK = True
except ImportError:
    _TORCHMETRICS_OK = False


# ── helpers (same as run_phase3_ablation.py) ──────────────────────────────────

def _load_image(path: Path, imgsz: int, device: torch.device) -> torch.Tensor:
    from torchvision.transforms import functional as TF
    from PIL import Image
    img = Image.open(path).convert("RGB").resize((imgsz, imgsz))
    return TF.to_tensor(img).unsqueeze(0).to(device)


def _detections_to_fd(dets, device) -> FrameDetections:
    if not dets:
        return FrameDetections(
            boxes=torch.zeros((0, 4), device=device),
            scores=torch.zeros(0, device=device),
            classes=torch.zeros(0, dtype=torch.long, device=device),
        )
    boxes   = torch.tensor([d.box   for d in dets], dtype=torch.float32, device=device)
    scores  = torch.tensor([d.score for d in dets], dtype=torch.float32, device=device)
    classes = torch.tensor([d.cls   for d in dets], dtype=torch.long,    device=device)
    return FrameDetections(boxes=boxes, scores=scores, classes=classes)


def _compute_lpips(x_adv, x, M, lpips_fn) -> float:
    if lpips_fn is None:
        return float("nan")
    with torch.no_grad():
        x_adv_m = x_adv * M * 2.0 - 1.0
        x_m     = x     * M * 2.0 - 1.0
        val     = lpips_fn(x_adv_m, x_m).squeeze()
        mask_frac = M.mean().clamp(min=1e-4)
        return float((val / mask_frac).item())


def pgd_mask_attack(detector, x, eps, alpha, steps, gamma, conf_thr, iou_nms):
    """PGD-mask: identical to baselines/pgd_pixel.py but returns (x_adv, M, D_clean)."""
    device = x.device
    D_clean = detector.detect_nms(x, conf_thr=conf_thr, iou_thr=iou_nms)
    if len(D_clean) == 0:
        M = torch.zeros((1, 1, x.shape[-2], x.shape[-1]), device=device)
        return x.clone(), M, D_clean, 0

    C_clean = sorted({d.cls for d in D_clean})
    M = boxes_to_pixel_mask(D_clean, H=x.shape[-2], W=x.shape[-1], device=device)

    delta = torch.zeros_like(x, requires_grad=True)
    steps_done = 0
    for t in range(steps):
        x_adv = (x + delta).clamp(0, 1)
        raw = detector.forward_raw(x_adv)
        class_conf = detector.class_confidence(raw)
        L_det = vanishing_loss(class_conf, C_clean, gamma=gamma)

        grad = torch.autograd.grad(L_det, delta)[0]
        with torch.no_grad():
            delta -= alpha * grad.sign()
            delta.clamp_(-eps, eps)
            delta.mul_(M)
            delta.data = (x + delta).clamp(0, 1) - x
            p = class_conf[0, :, C_clean].amax(dim=0).max().item()
        delta.requires_grad_(True)
        steps_done = t + 1
        if p < gamma:
            break

    return (x + delta.detach()).clamp(0, 1), M, D_clean, steps_done


def _summarise(records):
    import math, numpy as np
    if not records:
        return {}
    valid = [r for r in records if r.get("n_clean", 0) > 0]
    if not valid:
        return {"n_frames": len(records), "n_valid": 0}

    dfr_v    = [r["dfr"]        for r in valid]
    strict_v = [r["dfr_strict"] for r in valid]
    asr_v    = [float(r["asr"]) for r in valid]
    lpips_v  = [r["lpips"]      for r in valid if r.get("lpips") and math.isfinite(r["lpips"])]

    def ms(v):
        if not v: return float("nan"), float("nan")
        m = float(np.mean(v))
        se = float(np.std(v, ddof=1) / math.sqrt(len(v))) if len(v) > 1 else 0.0
        return m, se

    m_dfr,   se_dfr   = ms(dfr_v)
    m_asr,   _        = ms(asr_v)
    m_lpips, _        = ms(lpips_v)

    n_valid = len(valid)
    strict_count = sum(strict_v)

    return {
        "n_frames":         len(records),
        "n_valid":          n_valid,
        "mean_dfr":         m_dfr,
        "se_dfr":           se_dfr,
        "dfr_strict_rate":  strict_count / n_valid,
        "dfr_strict_count": strict_count,
        "mean_asr":         m_asr,
        "asr_count":        sum(int(v) for v in asr_v),
        "mean_lpips":       m_lpips,
    }


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data",      required=True)
    ap.add_argument("--n_frames",  type=int, default=70)
    ap.add_argument("--output",    required=True)
    ap.add_argument("--eps",       type=float, default=8/255,   help="PGD L-inf budget")
    ap.add_argument("--alpha",     type=float, default=1/255,   help="PGD step size")
    ap.add_argument("--steps",     type=int,   default=50,      help="PGD steps")
    ap.add_argument("--weights",   default="runs/yolov8n_detrac/best.pt")
    ap.add_argument("--conf_thr",  type=float, default=0.25)
    ap.add_argument("--iou_nms",   type=float, default=0.45)
    ap.add_argument("--gamma",     type=float, default=0.05)
    ap.add_argument("--imgsz",     type=int,   default=640)
    ap.add_argument("--resume",    action="store_true", default=True)
    ap.add_argument("--no_resume", dest="resume", action="store_false")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # output dir: results/multiclass/<seq>/pgd_mask/
    out_dir = Path(args.output) / "pgd_mask"
    out_dir.mkdir(parents=True, exist_ok=True)
    records_path = out_dir / "per_frame.jsonl"

    # load frames
    data_dir = Path(args.data)
    all_frames = sorted(list(data_dir.glob("*.jpg")) + list(data_dir.glob("*.png")))
    frames = all_frames[:args.n_frames]
    print(f"Frames: {len(frames)} of {len(all_frames)} available")

    # resume
    done: set[str] = set()
    records: list[dict] = []
    if args.resume and records_path.exists():
        with open(records_path) as f:
            for line in f:
                r = json.loads(line)
                records.append(r)
                done.add(r["frame_id"])
        print(f"Resuming: {len(done)} frames already done")

    # load models
    print(f"Loading detector: {args.weights}")
    detector = YOLOv8Wrapper(weights=args.weights, device=str(device))

    lpips_fn = None
    if _LPIPS_OK:
        import lpips
        lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(device)
        for p in lpips_fn.parameters():
            p.requires_grad_(False)

    print(f"PGD-mask: eps={args.eps:.4f} alpha={args.alpha:.5f} steps={args.steps}")

    with open(records_path, "a") as fout:
        for img_path in frames:
            stem = img_path.stem
            if stem in done:
                continue

            t0 = time.time()
            x = _load_image(img_path, args.imgsz, device)

            x_adv, M, D_clean, steps_done = pgd_mask_attack(
                detector, x,
                eps=args.eps, alpha=args.alpha, steps=args.steps,
                gamma=args.gamma, conf_thr=args.conf_thr, iou_nms=args.iou_nms,
            )

            D_adv = detector.detect_nms(x_adv, conf_thr=args.conf_thr, iou_thr=args.iou_nms)

            fd_clean = _detections_to_fd(D_clean, device)
            fd_adv   = _detections_to_fd(D_adv,   device)
            n_clean, n_adv = len(D_clean), len(D_adv)

            dfr_val  = per_frame_dfr(fd_clean, fd_adv, conf_thr=0.0)
            asr_val  = per_frame_asr(fd_clean, fd_adv, conf_thr=0.0)
            map_drop = (per_frame_map_drop(fd_clean, fd_adv)
                        if _TORCHMETRICS_OK and n_clean > 0 else None)
            lpips_val = _compute_lpips(x_adv, x, M, lpips_fn)

            rec = {
                "frame_id":  stem,
                "n_clean":   n_clean,
                "n_adv":     n_adv,
                "dfr":       dfr_val,
                "dfr_strict": int(n_adv == 0 and n_clean > 0),
                "asr":       bool(asr_val),
                "map_drop":  map_drop,
                "lpips":     lpips_val,
                "steps":     steps_done,
                "elapsed_s": round(time.time() - t0, 2),
            }
            records.append(rec)
            fout.write(json.dumps(rec) + "\n")
            fout.flush()

            print(f"  [pgd_mask] {stem}  n_clean={n_clean} n_adv={n_adv} "
                  f"dfr={dfr_val:+.3f} strict={rec['dfr_strict']} "
                  f"asr={int(asr_val)} steps={steps_done} ({rec['elapsed_s']:.1f}s)")

    summary = _summarise(records)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== PGD-mask summary ({summary.get('n_frames')} frames) ===")
    print(f"  DFR_prop   : {summary.get('mean_dfr', float('nan')):+.1%} ± {summary.get('se_dfr', 0):.1%}")
    print(f"  DFR_strict : {summary.get('dfr_strict_rate', float('nan')):.1%}")
    print(f"  ASR        : {summary.get('mean_asr', float('nan')):.1%}")
    print(f"  LPIPS      : {summary.get('mean_lpips', float('nan')):.3f}")
    print(f"\nResults → {out_dir}/summary.json")


if __name__ == "__main__":
    main()
