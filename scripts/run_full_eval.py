"""Phase 1.5 — Full evaluation pipeline (letterbox-aware).

Runs YOLOv8 inference on clean and adversarial images in a COMMON coordinate
space (640×640 letterboxed), computes strict metrics, and writes
results/metrics_full.json.

Coordinate-space alignment
--------------------------
Adversarial images are pre-saved at 640×640 letterboxed resolution.  Running
YOLO directly on the original 960×540 clean images returns boxes in 960×540
space, giving IoU≈0 vs adv boxes → mAP_drop trivially 1.0.

Fix: we pre-letterbox every clean image to 640×640 using src.viz.letterbox
and write the result to a temp directory before running YOLO.  All four
detection JSONs end up in the same 640×640 coordinate frame.

Detection JSONs in the OLD coordinate space (max coord > 640) are detected
automatically and regenerated.

Usage
-----
    python scripts/run_full_eval.py           # auto-detect, skip cached
    python scripts/run_full_eval.py --force   # delete all cached dets, re-run
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.bootstrap import bootstrap_ci  # noqa: E402
from src.eval.io import load_detections, save_detections  # noqa: E402
from src.eval.metrics import FrameDetections, per_frame_asr, per_frame_dfr  # noqa: E402
from src.eval.run_detection import run_detection  # noqa: E402
from src.viz.letterbox import letterbox_image  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_PATH = ROOT / "runs" / "yolov8n_detrac" / "best.pt"
CLEAN_DIR = ROOT / "data" / "images"
ADV_DIRS: dict[str, Path] = {
    "latent": ROOT / "results" / "adv_latent",
    "pgd": ROOT / "results" / "adv_pgd",
    "fgsm": ROOT / "results" / "adv_fgsm",
}
RESULTS_DIR = ROOT / "results"
OUT_FULL = RESULTS_DIR / "metrics_full.json"
LB_PARAMS_PATH = RESULTS_DIR / "letterbox_params.json"

CONF_THR: float = 0.25
IOU_NMS: float = 0.45
IMG_SIZE: int = 640
LB_TARGET: tuple[int, int] = (640, 640)
N_BOOT: int = 1000
SEED: int = 42
REFERENCE_N_CLEAN: int = 387
TOLERANCE: float = 0.05

STEMS = [f"img{i:05d}" for i in range(1, 51)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _device() -> str:
    if torch.cuda.is_available():
        dev = "cuda"
        print(f"[INFO] CUDA available — using GPU ({torch.cuda.get_device_name(0)})")
    else:
        dev = "cpu"
        print("[INFO] No CUDA — using CPU")
    return dev


def _is_old_coordinate_space(dets: dict[str, FrameDetections]) -> bool:
    """Return True if any box coordinate exceeds 640 (non-letterboxed space)."""
    for fd in dets.values():
        if len(fd.boxes) > 0 and float(fd.boxes.max()) > 640.5:
            return True
    return False


def _letterbox_stems(
    stems: list[str],
    src_dir: Path,
    tmp_dir: Path,
    target: tuple[int, int],
) -> dict[str, dict]:
    """Letterbox each clean image to tmp_dir. Returns letterbox params per stem."""
    params: dict[str, dict] = {}
    for stem in stems:
        for ext in (".jpg", ".jpeg", ".png", ".bmp"):
            src = src_dir / f"{stem}{ext}"
            if not src.exists():
                continue
            img = cv2.imread(str(src))
            if img is None:
                continue
            orig_h, orig_w = img.shape[:2]
            lb, scale, (pt, pl) = letterbox_image(img, target=target)
            cv2.imwrite(str(tmp_dir / f"{stem}{ext}"), lb)
            params[stem] = {
                "scale": scale,
                "pad_top": pt,
                "pad_left": pl,
                "orig_h": orig_h,
                "orig_w": orig_w,
            }
            break
    return params


def _sanity_check(clean_dets: dict[str, FrameDetections]) -> int:
    n_total = sum(len(fd.boxes) for fd in clean_dets.values())
    lo = int(REFERENCE_N_CLEAN * (1 - TOLERANCE))
    hi = int(REFERENCE_N_CLEAN * (1 + TOLERANCE))
    if not (lo <= n_total <= hi):
        raise RuntimeError(
            f"Sanity check FAILED: expected {REFERENCE_N_CLEAN} ±{TOLERANCE*100:.0f}% "
            f"clean detections ({lo}–{hi}), got {n_total}.\n"
            f"Check model weights and confidence threshold."
        )
    print(
        f"[OK] Sanity check passed: {n_total} clean detections "
        f"(reference {REFERENCE_N_CLEAN}, tol ±{TOLERANCE*100:.0f}%)"
    )
    return n_total


def _compute_map(
    clean_dets: dict[str, FrameDetections],
    adv_dets: dict[str, FrameDetections],
    stems: list[str],
    iou_thresholds: list[float],
) -> float:
    from torchmetrics.detection import MeanAveragePrecision  # noqa: PLC0415

    metric = MeanAveragePrecision(iou_thresholds=iou_thresholds)
    for stem in stems:
        if stem not in clean_dets or stem not in adv_dets:
            continue
        c = clean_dets[stem]
        a = adv_dets[stem]
        metric.update(
            preds=[{
                "boxes": a.boxes.float(),
                "scores": a.scores.float() if len(a.scores) > 0
                          else torch.zeros(0, dtype=torch.float32),
                "labels": a.classes.long(),
            }],
            target=[{
                "boxes": c.boxes.float(),
                "labels": c.classes.long(),
            }],
        )
    result = metric.compute()
    v = float(result["map"].item())
    return max(0.0, v) if np.isfinite(v) else 0.0


def _per_frame_metrics(
    clean_dets: dict[str, FrameDetections],
    adv_dets: dict[str, FrameDetections],
    stems: list[str],
) -> list[dict]:
    rows: list[dict] = []
    for stem in stems:
        if stem not in clean_dets:
            continue
        c = clean_dets[stem]
        a = adv_dets.get(
            stem,
            FrameDetections(
                boxes=torch.zeros((0, 4)),
                scores=torch.zeros(0),
                classes=torch.zeros(0, dtype=torch.long),
            ),
        )
        n_clean = len(c.boxes)
        n_adv = len(a.boxes)
        rows.append({
            "stem": stem,
            "n_clean": n_clean,
            "n_adv": n_adv,
            "dfr_prop": per_frame_dfr(c, a, conf_thr=CONF_THR),
            "dfr_bin": 1.0 if n_adv == 0 else 0.0,
            "asr_strict": bool(per_frame_asr(c, a, conf_thr=CONF_THR)),
        })
    return rows


def _summarise(
    per_frame: list[dict],
    clean_dets: dict[str, FrameDetections],
    adv_dets: dict[str, FrameDetections],
    stems: list[str],
) -> dict:
    valid = [f for f in per_frame if f["n_clean"] > 0]
    n = len(valid)
    if n == 0:
        return {}

    asr_vals = np.array([1.0 if f["asr_strict"] else 0.0 for f in valid])
    dfr_prop_vals = np.array([f["dfr_prop"] for f in valid])
    dfr_bin_vals = np.array([f["dfr_bin"] for f in valid])

    asr_mean, asr_lo, asr_hi = bootstrap_ci(asr_vals, n_boot=N_BOOT, seed=SEED)
    prop_mean, prop_lo, prop_hi = bootstrap_ci(dfr_prop_vals, n_boot=N_BOOT, seed=SEED)
    bin_mean, bin_lo, bin_hi = bootstrap_ci(dfr_bin_vals, n_boot=N_BOOT, seed=SEED)

    print("    Computing mAP@0.5 ...", end=" ", flush=True)
    map_50 = _compute_map(clean_dets, adv_dets, stems, iou_thresholds=[0.5])
    print(f"{map_50:.4f}")

    coco_thrs = [round(t, 2) for t in np.arange(0.5, 1.0, 0.05).tolist()]
    print("    Computing mAP@0.5:0.95 ...", end=" ", flush=True)
    map_5095 = _compute_map(clean_dets, adv_dets, stems, iou_thresholds=coco_thrs)
    print(f"{map_5095:.4f}")

    n_fp = int(np.sum(dfr_prop_vals < 0))

    return {
        "n_frames": n,
        "ASR_strict": float(asr_mean),
        "ASR_strict_lo": float(asr_lo),
        "ASR_strict_hi": float(asr_hi),
        "DFR_prop": float(prop_mean),
        "DFR_prop_lo": float(prop_lo),
        "DFR_prop_hi": float(prop_hi),
        "DFR_bin": float(bin_mean),
        "DFR_bin_lo": float(bin_lo),
        "DFR_bin_hi": float(bin_hi),
        "mAP_50": float(map_50),
        "mAP_drop_50": float(max(0.0, 1.0 - map_50)),
        "mAP_5095": float(map_5095),
        "mAP_drop_5095": float(max(0.0, 1.0 - map_5095)),
        "n_fp_inflation": n_fp,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(force: bool = False) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = _device()

    # ------------------------------------------------------------------ #
    # 1. Clean-image inference (letterboxed to 640×640)
    # ------------------------------------------------------------------ #
    clean_dets_path = RESULTS_DIR / "dets_clean.json"

    need_clean_inference = True
    if clean_dets_path.exists() and not force:
        cached = load_detections(clean_dets_path)
        if _is_old_coordinate_space(cached):
            print(
                "[WARN] dets_clean.json is in the OLD (non-letterboxed) coordinate space "
                "(max box coord > 640). Deleting and regenerating with letterboxed images."
            )
            clean_dets_path.unlink()
            if OUT_FULL.exists():
                OUT_FULL.unlink()
                print("[INFO] Deleted metrics_full.json (invalidated).")
        else:
            print(f"[SKIP] {clean_dets_path} (letterboxed space) — loading cached.")
            clean_dets = cached
            need_clean_inference = False
    elif force:
        for p in [clean_dets_path, OUT_FULL]:
            if p.exists():
                p.unlink()
                print(f"[FORCE] Deleted {p.name}")
        for atk in ADV_DIRS:
            ap = RESULTS_DIR / f"dets_{atk}.json"
            if ap.exists():
                ap.unlink()
                print(f"[FORCE] Deleted dets_{atk}.json")

    if need_clean_inference:
        print(f"\n[1/4] Letterboxing clean images to {LB_TARGET} and running YOLO ...")
        tmp_dir = Path(tempfile.mkdtemp(prefix="clean_lb_"))
        try:
            lb_params = _letterbox_stems(STEMS, CLEAN_DIR, tmp_dir, LB_TARGET)
            # Save letterbox params for downstream use (visualization, etc.)
            with open(LB_PARAMS_PATH, "w", encoding="utf-8") as fh:
                json.dump(lb_params, fh, indent=2)
            clean_dets = run_detection(
                image_dir=tmp_dir,
                model_path=MODEL_PATH,
                output_path=clean_dets_path,
                stems=STEMS,
                conf_thr=CONF_THR,
                iou_nms=IOU_NMS,
                img_size=IMG_SIZE,
                device=device,
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    n_clean_total = _sanity_check(clean_dets)

    # ------------------------------------------------------------------ #
    # 2. Adversarial inference + per-attack metrics
    # ------------------------------------------------------------------ #
    output: dict = {
        "meta": {
            "n_frames": len(STEMS),
            "conf_thr": CONF_THR,
            "iou_nms": IOU_NMS,
            "device": device,
            "letterbox_target": list(LB_TARGET),
            "coordinate_space": "640x640_letterboxed",
            "n_boot": N_BOOT,
            "seed": SEED,
        },
        "clean": {
            "dets_path": str(clean_dets_path.relative_to(ROOT)),
            "n_total": n_clean_total,
        },
    }

    for step, attack in enumerate(list(ADV_DIRS.keys()), start=2):
        adv_dir = ADV_DIRS[attack]
        dets_path = RESULTS_DIR / f"dets_{attack}.json"

        if dets_path.exists() and not force:
            print(f"[SKIP] dets_{attack}.json — loading cached.")
            adv_dets = load_detections(dets_path)
        else:
            print(f"\n[{step}/{len(ADV_DIRS)+1}] Running YOLO on {attack} ({adv_dir}) ...")
            adv_dets = run_detection(
                image_dir=adv_dir,
                model_path=MODEL_PATH,
                output_path=dets_path,
                stems=STEMS,
                conf_thr=CONF_THR,
                iou_nms=IOU_NMS,
                img_size=IMG_SIZE,
                device=device,
            )

        print(f"  Computing strict metrics for {attack.upper()} ...")
        per_frame = _per_frame_metrics(clean_dets, adv_dets, STEMS)
        summary = _summarise(per_frame, clean_dets, adv_dets, STEMS)

        nf = summary.get("n_frames", 0)
        print(
            f"  ASR_strict={summary.get('ASR_strict', 0):.3f} "
            f"[{summary.get('ASR_strict_lo', 0):.3f}, {summary.get('ASR_strict_hi', 0):.3f}]"
            f"  mAP_drop@0.5={summary.get('mAP_drop_50', 0):.3f}"
            f"  n_fp={summary.get('n_fp_inflation', 0)}/{nf}"
        )

        output[attack] = {
            "dets_path": str(dets_path.relative_to(ROOT)),
            "per_frame": per_frame,
            "summary": summary,
        }

    # ------------------------------------------------------------------ #
    # 3. Save metrics_full.json
    # ------------------------------------------------------------------ #
    with open(OUT_FULL, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2)
    print(f"\n[OK] Wrote {OUT_FULL}")

    # ------------------------------------------------------------------ #
    # 4. Refresh metric_comparison.md
    # ------------------------------------------------------------------ #
    print("\n[INFO] Refreshing metric_comparison.md ...")
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "recompute_metrics.py")],
        check=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force", action="store_true",
        help="Delete all cached dets_*.json and metrics_full.json, then re-run everything.",
    )
    args = parser.parse_args()
    main(force=args.force)
