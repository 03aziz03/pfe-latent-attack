"""Phase 3 ablation sweep — incremental improvement benchmarking.

Runs the full attack pipeline for each of the six Phase 3 configs:
  baseline, objectness (3A), momentum (3B), restart (3C), ssim (3D), combined.

For every (config, frame) pair the script computes:
  DFR (proportional), DFR_strict, ASR, mAP_drop, LPIPS, steps_taken

Results are saved to ``results/phase3/<config_name>/`` in the same format as
run_ablation_sweep.py so the Phase 3 Pareto and comparison scripts can reuse
the existing figures/ablation_table.py pipeline.

Usage (Colab)
-------------
    !python scripts/run_phase3_ablation.py \\
        --data   data/images_50 \\
        --n_frames 30 \\
        --output results/phase3 \\
        --configs configs/phase3_baseline.yaml \\
                  configs/phase3_objectness.yaml \\
                  configs/phase3_momentum.yaml \\
                  configs/phase3_restart.yaml \\
                  configs/phase3_ssim.yaml \\
                  configs/phase3_combined.yaml

Alternatively, use --all_phase3 to automatically discover all
configs/phase3_*.yaml files in alphabetical order.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import torch
import yaml

# ── ensure project root is on sys.path ────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.attack import AttackConfig, LatentObjectAttack
from src.detector import YOLOv8Wrapper
from src.vae import SDVAE
from src.eval.metrics import (FrameDetections, per_frame_dfr, per_frame_asr,
                               per_frame_map_drop)

try:
    import lpips as _lpips_lib  # noqa: F401
    _LPIPS_OK = True
except ImportError:
    _LPIPS_OK = False

try:
    from torchmetrics.detection import MeanAveragePrecision as _MAP_CLS  # noqa: F401
    _TORCHMETRICS_OK = True
except ImportError:
    _TORCHMETRICS_OK = False

try:
    import lpips as _lpips_lib2
    _LPIPS_OK2 = True
except ImportError:
    _LPIPS_OK2 = False


# ── helpers ───────────────────────────────────────────────────────────────────


def _load_image(path: Path, imgsz: int, device: torch.device) -> torch.Tensor:
    """Load a JPEG/PNG frame as a (1,3,H,W) float tensor in [0,1]."""
    from torchvision.transforms import functional as TF
    from PIL import Image
    img = Image.open(path).convert("RGB")
    img = img.resize((imgsz, imgsz), Image.BILINEAR)
    return TF.to_tensor(img).unsqueeze(0).to(device)


def _detections_to_fd(dets, device: torch.device) -> FrameDetections:
    """Convert list[Detection] → FrameDetections for metric functions."""
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


def _compute_lpips_value(x_adv: torch.Tensor, x: torch.Tensor,
                          M: torch.Tensor, lpips_fn) -> float:
    """Compute masked LPIPS value (no grad)."""
    if lpips_fn is None:
        return float("nan")
    with torch.no_grad():
        x_adv_m = x_adv * M * 2.0 - 1.0
        x_m     = x     * M * 2.0 - 1.0
        val     = lpips_fn(x_adv_m, x_m).squeeze()
        mask_frac = M.mean().clamp(min=1e-4)
        return float((val / mask_frac).item())


def _build_attack_config(cfg_yaml: dict) -> AttackConfig:
    """Parse the 'attack' block of a YAML config into AttackConfig."""
    atk = cfg_yaml.get("attack", {})
    return AttackConfig(
        eps_z              = float(atk.get("eps_z",              0.50)),
        gamma              = float(atk.get("gamma",              0.05)),
        lambda_p           = float(atk.get("lambda_p",           0.001)),
        lambda_r           = float(atk.get("lambda_r",           1e-3)),
        lr                 = float(atk.get("lr",                 0.01)),
        num_steps          = int(atk.get("num_steps",            80)),
        early_stop         = bool(atk.get("early_stop",          True)),
        early_stop_margin  = float(atk.get("early_stop_margin",  0.0)),
        conf_thr           = float(cfg_yaml.get("detector", {}).get("conf_thr", 0.25)),
        iou_nms            = float(cfg_yaml.get("detector", {}).get("iou_nms",  0.45)),
        use_lpips          = bool(atk.get("use_lpips",           True)),
        lpips_net          = str(atk.get("lpips_net",            "alex")),
        # Phase 3
        use_objectness     = bool(atk.get("use_objectness",      False)),
        obj_weight         = float(atk.get("obj_weight",         0.5)),
        use_momentum       = bool(atk.get("use_momentum",        False)),
        momentum_decay     = float(atk.get("momentum_decay",     0.9)),
        n_restarts         = int(atk.get("n_restarts",           1)),
        restart_noise      = float(atk.get("restart_noise",      0.5)),
        use_ssim           = bool(atk.get("use_ssim",            False)),
        ssim_weight        = float(atk.get("ssim_weight",        0.3)),
    )


def _summarise(records: list[dict]) -> dict:
    """Aggregate per-frame records into a summary dict (mean ± SE)."""
    import math
    import numpy as np

    if not records:
        return {}

    def _mean_se(vals: list[float]) -> tuple[float, float]:
        arr = [v for v in vals if v is not None and math.isfinite(v)]
        if not arr:
            return float("nan"), float("nan")
        m = float(np.mean(arr))
        se = float(np.std(arr, ddof=1) / math.sqrt(len(arr))) if len(arr) > 1 else 0.0
        return m, se

    n_frames = len(records)

    # DFR (proportional) — skip frames with n_clean == 0
    dfr_vals = [r["dfr"] for r in records if r.get("n_clean", 0) > 0]
    # DFR strict — fraction of valid frames with n_adv == 0
    strict_vals = [r.get("dfr_strict", 0) for r in records if r.get("n_clean", 0) > 0]
    # ASR
    asr_vals = [float(r["asr"]) for r in records if r.get("n_clean", 0) > 0]
    # mAP drop
    map_vals = [r["map_drop"] for r in records if r.get("map_drop") is not None]
    # LPIPS
    lpips_vals = [r["lpips"] for r in records
                  if r.get("lpips") is not None and math.isfinite(r["lpips"])]
    # steps
    steps_vals = [r.get("steps", 0) for r in records]

    m_dfr, se_dfr = _mean_se(dfr_vals)
    m_asr, _       = _mean_se(asr_vals)
    m_map, se_map  = _mean_se(map_vals)
    m_lpips, _     = _mean_se(lpips_vals)

    n_valid = len(dfr_vals)
    strict_count = sum(strict_vals)
    strict_rate  = strict_count / n_valid if n_valid > 0 else float("nan")

    return {
        "n_frames":        n_frames,
        "n_valid":         n_valid,
        "mean_dfr":        m_dfr,
        "se_dfr":          se_dfr,
        "dfr_strict_rate": strict_rate,
        "dfr_strict_count":strict_count,
        "mean_asr":        m_asr,
        "asr_count":       sum(int(v) for v in asr_vals),
        "mean_map_drop":   m_map,
        "se_map_drop":     se_map,
        "mean_lpips":      m_lpips,
        "mean_steps":      float(sum(steps_vals) / max(len(steps_vals), 1)),
    }


def _print_row(tag: str, s: dict) -> None:
    def fmt(key, w, plus=False):
        v = s.get(key)
        if v is None or (isinstance(v, float) and not (v == v)):
            return " " * w + "N/A"
        prefix = "+" if plus and v > 0 else ""
        return f"{prefix}{v:{'+' if plus else ''}.4f}".rjust(w)

    print(
        f"{tag:<28} "
        f"{fmt('mean_dfr', 7, plus=True)} "
        f"{fmt('dfr_strict_rate', 9)} "
        f"{fmt('mean_asr', 7)} "
        f"{fmt('mean_map_drop', 10)} "
        f"{fmt('mean_lpips', 8)} "
        f"{s.get('mean_steps', 0):>7.1f}  "
        f"{s.get('n_frames', 0):>4}"
    )


# ── main sweep ────────────────────────────────────────────────────────────────


def sweep_config(
    config_path: Path,
    frames: list[Path],
    output_dir: Path,
    detector: YOLOv8Wrapper,
    vae: SDVAE,
    lpips_fn,
    resume: bool = True,
    verbose: bool = True,
) -> dict:
    """Run the attack for one config on all frames, return summary dict."""
    tag = config_path.stem

    with open(config_path) as f:
        cfg_yaml = yaml.safe_load(f)

    atk_cfg = _build_attack_config(cfg_yaml)
    imgsz   = int(cfg_yaml.get("detector", {}).get("imgsz", 640))
    device  = vae.device

    config_out = output_dir / tag
    config_out.mkdir(parents=True, exist_ok=True)
    records_path = config_out / "per_frame.jsonl"

    # Load previously completed frames if --resume
    done_frames: set[str] = set()
    records: list[dict] = []
    if resume and records_path.exists():
        with open(records_path) as f:
            for line in f:
                r = json.loads(line)
                records.append(r)
                done_frames.add(r["frame_id"])

    attack = LatentObjectAttack(detector=detector, vae=vae, config=atk_cfg)

    with open(records_path, "a") as fout:
        for img_path in frames:
            stem = img_path.stem
            if stem in done_frames:
                continue

            t0 = time.time()
            x = _load_image(img_path, imgsz, device)

            # run attack
            result = attack.attack(x)

            # re-detect on adversarial image
            D_adv = detector.detect_nms(
                result.x_adv,
                conf_thr=atk_cfg.conf_thr,
                iou_thr=atk_cfg.iou_nms,
            )

            # convert to FrameDetections
            fd_clean = _detections_to_fd(result.detections_clean, device)
            fd_adv   = _detections_to_fd(D_adv, device)

            n_clean = len(result.detections_clean)
            n_adv   = len(D_adv)

            dfr_val  = per_frame_dfr(fd_clean, fd_adv, conf_thr=0.0)
            asr_val  = per_frame_asr(fd_clean, fd_adv, conf_thr=0.0)
            map_drop = (per_frame_map_drop(fd_clean, fd_adv)
                        if _TORCHMETRICS_OK and n_clean > 0 else None)
            lpips_val = _compute_lpips_value(
                result.x_adv, x, result.M, lpips_fn)

            rec = {
                "frame_id":  stem,
                "n_clean":   n_clean,
                "n_adv":     n_adv,
                "dfr":       dfr_val,
                "dfr_strict": int(n_adv == 0 and n_clean > 0),
                "asr":       bool(asr_val),
                "map_drop":  map_drop,
                "lpips":     lpips_val,
                "steps":     result.steps_taken,
                "elapsed_s": round(time.time() - t0, 2),
            }
            records.append(rec)
            fout.write(json.dumps(rec) + "\n")
            fout.flush()

            if verbose:
                print(f"  [{tag}] {stem}  n_clean={n_clean} n_adv={n_adv} "
                      f"dfr={dfr_val:+.3f} strict={rec['dfr_strict']} "
                      f"asr={int(asr_val)} steps={result.steps_taken} "
                      f"({rec['elapsed_s']:.1f}s)")

    summary = _summarise(records)
    with open(config_out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",      required=True, help="Directory of input frames (.jpg)")
    parser.add_argument("--n_frames",  type=int, default=30, help="Max frames to process")
    parser.add_argument("--output",    default="results/phase3", help="Output root dir")
    parser.add_argument("--configs",   nargs="+", help="Explicit list of config YAML paths")
    parser.add_argument("--all_phase3", action="store_true",
                        help="Auto-discover configs/phase3_*.yaml")
    parser.add_argument("--resume",    action="store_true", default=True,
                        help="Skip already-processed frames (default: True)")
    parser.add_argument("--no_resume", dest="resume", action="store_false")
    parser.add_argument("--seed",      type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # ---- config list ----
    if args.all_phase3:
        config_paths = sorted(Path("configs").glob("phase3_*.yaml"))
    elif args.configs:
        config_paths = [Path(p) for p in args.configs]
    else:
        # default: run all phase3_*.yaml
        config_paths = sorted(Path("configs").glob("phase3_*.yaml"))

    if not config_paths:
        print("ERROR: no phase3 config files found. "
              "Use --configs or --all_phase3.", file=sys.stderr)
        sys.exit(1)

    # ---- frame list ----
    data_dir = Path(args.data)
    all_frames = sorted(data_dir.glob("*.jpg"))
    if not all_frames:
        all_frames = sorted(data_dir.glob("*.png"))
    frames = all_frames[:args.n_frames]
    if not frames:
        print(f"ERROR: no frames found in {data_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"Frames: {len(frames)} of {len(all_frames)} available")

    # ---- shared models (loaded once) ----
    # We load from the first config; all phase3 configs use the same weights
    with open(config_paths[0]) as f:
        first_cfg = yaml.safe_load(f)

    device_str = first_cfg.get("runtime", {}).get("device", "cuda")
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    det_weights = first_cfg["detector"]["weights"]
    vae_id      = first_cfg["vae"]["model_id"]

    print(f"Loading detector: {det_weights}")
    detector = YOLOv8Wrapper(weights=det_weights, device=str(device))

    print(f"Loading VAE: {vae_id}")
    vae = SDVAE(model_id=vae_id, device=str(device))

    # Shared LPIPS instance (AlexNet, used for evaluation even when attack uses masked_l2)
    lpips_fn = None
    if _LPIPS_OK2:
        import lpips as _lpips_lib3
        lpips_fn = _lpips_lib3.LPIPS(net="alex", verbose=False).to(device)
        for p in lpips_fn.parameters():
            p.requires_grad_(False)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── sweep ────────────────────────────────────────────────────────────────
    header = (f"{'Config':<28} {'DFR':>7} {'DFR_str':>9} {'ASR':>7} "
              f"{'mAP_drop':>10} {'LPIPS':>8} {'Steps':>7}  {'N':>4}")
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    all_summaries: dict[str, dict] = {}

    for cfg_path in config_paths:
        tag = cfg_path.stem
        print(f"\n▶ Running: {tag} ({cfg_path})")
        try:
            summary = sweep_config(
                config_path=cfg_path,
                frames=frames,
                output_dir=output_dir,
                detector=detector,
                vae=vae,
                lpips_fn=lpips_fn,
                resume=args.resume,
            )
            all_summaries[tag] = summary
            _print_row(tag, summary)
        except Exception as exc:
            print(f"  ERROR in {tag}: {exc}")
            import traceback; traceback.print_exc()

    # ── combined summary ──────────────────────────────────────────────────────
    combined_path = output_dir / "phase3_summary.json"
    with open(combined_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\n✓ Combined summary → {combined_path}")

    # ── markdown table ────────────────────────────────────────────────────────
    md_lines = [
        "| Config | DFR | DFR_strict | ASR | mAP_drop | LPIPS | Steps | N |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for tag, s in all_summaries.items():
        def _f(k, plus=False):
            v = s.get(k)
            if v is None or (isinstance(v, float) and v != v):
                return "N/A"
            prefix = "+" if plus and v > 0 else ""
            return f"{prefix}{v:.4f}"
        md_lines.append(
            f"| {tag} "
            f"| {_f('mean_dfr', plus=True)} "
            f"| {_f('dfr_strict_rate')} "
            f"| {_f('mean_asr')} "
            f"| {_f('mean_map_drop')} "
            f"| {_f('mean_lpips')} "
            f"| {s.get('mean_steps', 0):.1f} "
            f"| {s.get('n_frames', 0)} |"
        )

    md_path = output_dir / "phase3_table.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"✓ Markdown table → {md_path}")
    print("\n" + "\n".join(md_lines))


if __name__ == "__main__":
    main()
