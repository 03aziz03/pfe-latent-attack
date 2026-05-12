"""Generate publication-ready qualitative figures for the PFE report.

Runs the latent attack (Option 1 config) on a small set of representative
frames, applies bilateral post-processing (Option 2), and saves:

    results/qualitative/
        frames/
            {stem}_clean.png
            {stem}_opt1_adv.png
            {stem}_opt2_bilateral.png
            {stem}_diff_opt1.png          amplified |opt1 - clean| heatmap
            {stem}_diff_opt2.png          amplified |opt2 - clean| heatmap
            {stem}_mask.png               bounding-box mask (white on black)

        panels/
            f15_comparison_{stem}.png     clean | opt1 | opt2 | heatmap (4-col)
            f16_filter_effect_{stem}.png  opt1 | bilateral | |diff| (3-col)
            f17_ablation_grid.png         N-frame grid: 4 rows × 4 cols
            f18_perturbation_heatmap.png  shared-vmax heatmap: opt1 vs opt2

All panels are 300 dpi, publication-ready (no plt.show(), tight layout).

Usage
-----
    python scripts/generate_qualitative.py \\
        --config  configs/phase2_option1.yaml \\
        --data    data/images_50 \\
        --frames  img00001 img00008 img00020 img00026 \\
        --output  results/qualitative

    # or select by index
    python scripts/generate_qualitative.py --frame_indices 0 7 19 25
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.attack import AttackConfig, LatentObjectAttack
from src.detector import YOLOv8Wrapper
from src.losses import MaskedLPIPS
from src.masks import boxes_to_pixel_mask
from src.utils import load_config, load_image, set_seed
from src.vae import SDVAE


# ---------------------------------------------------------------------------
# Publication style
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "figure.dpi":        150,
    "font.family":       "DejaVu Sans",
    "font.size":         9,
    "axes.titlesize":    10,
    "axes.titleweight":  "bold",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.pad_inches": 0.05,
})

# Colour palette
COL_CLEAN  = "#2196F3"   # blue
COL_OPT1   = "#FF5722"   # deep orange
COL_OPT2   = "#4CAF50"   # green
COL_HEATMAP = "inferno"


# ---------------------------------------------------------------------------
# Image I/O helpers
# ---------------------------------------------------------------------------

def tensor_to_np(t: torch.Tensor) -> np.ndarray:
    """(1, 3, H, W) float32 [0,1] → (H, W, 3) uint8."""
    return (t[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)


def tensor_to_float(t: torch.Tensor) -> np.ndarray:
    """(1, 3, H, W) float32 [0,1] → (H, W, 3) float32."""
    return t[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()


def uint8_to_tensor(arr: np.ndarray, device: str) -> torch.Tensor:
    """(H, W, 3) uint8 → (1, 3, H, W) float32 [0,1]."""
    return torch.from_numpy(arr).float().div(255).permute(2, 0, 1).unsqueeze(0).to(device)


def save_png(arr_uint8: np.ndarray, path: Path) -> None:
    """Save (H, W, 3) uint8 RGB as PNG."""
    cv2.imwrite(str(path), cv2.cvtColor(arr_uint8, cv2.COLOR_RGB2BGR))


# ---------------------------------------------------------------------------
# Bilateral post-processing
# ---------------------------------------------------------------------------

def bilateral_masked(
    x_adv:   torch.Tensor,
    x_clean: torch.Tensor,
    M:       torch.Tensor,
    d:       int   = 9,
    sc:      float = 50,
    ss:      float = 50,
) -> torch.Tensor:
    device = x_adv.device
    x_np   = tensor_to_np(x_adv)
    filt   = cv2.bilateralFilter(x_np, d=d, sigmaColor=sc, sigmaSpace=ss)
    filt_t = uint8_to_tensor(filt, str(device))
    return (M * filt_t + (1 - M) * x_clean).clamp(0, 1)


# ---------------------------------------------------------------------------
# Draw bounding boxes on a numpy image
# ---------------------------------------------------------------------------

COLORS_BBOX = [
    (255,  80,  80),   # red — clean
    ( 80, 200,  80),   # green — adversarial
]

def draw_boxes(
    img_uint8:  np.ndarray,         # (H, W, 3) uint8 RGB
    detections: list,               # list[Detection]  (from detector)
    color:      tuple[int,int,int],
    thickness:  int = 2,
) -> np.ndarray:
    out = img_uint8.copy()
    for d in detections:
        x1, y1, x2, y2 = (int(v) for v in d.box)
        cv2.rectangle(out, (x1, y1), (x2, y2), color[::-1], thickness)  # BGR
    return out


def draw_boxes_both(
    img_uint8:  np.ndarray,
    clean_dets: list,
    adv_dets:   list,
) -> np.ndarray:
    """Draw clean boxes (red) and adversarial boxes (green) on same image."""
    out = draw_boxes(img_uint8, clean_dets, (220, 50,  50))
    out = draw_boxes(out,        adv_dets,   ( 50, 200, 50))
    return out


# ---------------------------------------------------------------------------
# Heatmap helpers
# ---------------------------------------------------------------------------

def diff_heatmap_np(x_adv: np.ndarray, x_clean: np.ndarray) -> np.ndarray:
    """(H,W,3) float → (H,W) per-channel mean absolute diff."""
    return np.abs(x_adv - x_clean).mean(axis=2)


def apply_colormap(heat: np.ndarray, vmax: float | None = None) -> np.ndarray:
    """(H,W) float → (H,W,3) uint8 using inferno colormap."""
    if vmax is None or vmax < 1e-8:
        vmax = heat.max() + 1e-8
    normed = np.clip(heat / vmax, 0, 1)
    cm     = plt.get_cmap("inferno")
    rgba   = (cm(normed) * 255).astype(np.uint8)
    return rgba[:, :, :3]   # drop alpha


# ---------------------------------------------------------------------------
# Panel generators
# ---------------------------------------------------------------------------

def make_comparison_panel(
    stem:        str,
    clean_np:    np.ndarray,
    opt1_np:     np.ndarray,
    opt2_np:     np.ndarray,
    clean_dets:  list,
    opt1_dets:   list,
    opt2_dets:   list,
    lpips_opt1:  float,
    lpips_opt2:  float,
    dfr_opt1:    float,
    dfr_opt2:    float,
) -> plt.Figure:
    """4-column panel: Clean | Option 1 | Option 2 | Perturbation heatmap."""
    heat1 = diff_heatmap_np(opt1_np.astype(np.float32)/255,
                             clean_np.astype(np.float32)/255)
    heat2 = diff_heatmap_np(opt2_np.astype(np.float32)/255,
                             clean_np.astype(np.float32)/255)
    shared_vmax = max(heat1.max(), heat2.max())

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # Col 0 — Clean
    ax = axes[0]
    ax.imshow(draw_boxes(clean_np, clean_dets, (220, 50, 50)))
    ax.set_title(f"Clean\n{len(clean_dets)} detections", color=COL_CLEAN)
    ax.axis("off")

    # Col 1 — Option 1 (LPIPS loss, original VAE)
    ax = axes[1]
    ax.imshow(draw_boxes_both(opt1_np, clean_dets, opt1_dets))
    ax.set_title(
        f"Option 1 — LPIPS loss\nDFR={dfr_opt1:+.3f}  LPIPS={lpips_opt1:.3f}",
        color=COL_OPT1,
    )
    ax.axis("off")

    # Col 2 — Option 2 (+ bilateral)
    ax = axes[2]
    ax.imshow(draw_boxes_both(opt2_np, clean_dets, opt2_dets))
    ax.set_title(
        f"Option 2 — + Bilateral filter\nDFR={dfr_opt2:+.3f}  LPIPS={lpips_opt2:.3f}",
        color=COL_OPT2,
    )
    ax.axis("off")

    # Col 3 — Heatmap overlay (Opt1 vs Opt2 side-by-side within the panel)
    ax = axes[3]
    # Stack heatmaps vertically (top=opt1, bottom=opt2)
    heat1_c = apply_colormap(heat1, vmax=shared_vmax)
    heat2_c = apply_colormap(heat2, vmax=shared_vmax)
    H = heat1_c.shape[0]
    divider = np.ones((4, heat1_c.shape[1], 3), dtype=np.uint8) * 200
    combined = np.concatenate([heat1_c, divider, heat2_c], axis=0)
    ax.imshow(combined)
    ax.set_title(
        f"Perturbation heatmap\n"
        f"Top: Opt1 (max={heat1.max():.4f})\n"
        f"Bot: Opt2 (max={heat2.max():.4f})",
        fontsize=8,
    )
    ax.axis("off")

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=COL_HEATMAP,
                                norm=plt.Normalize(vmin=0, vmax=shared_vmax))
    sm.set_array([])
    fig.colorbar(sm, ax=axes[3], fraction=0.046, pad=0.04,
                 label="|Δ| mean (shared vmax)")

    # Legend
    legend_elements = [
        mpatches.Patch(color=np.array([220,50,50])/255,  label="Clean boxes"),
        mpatches.Patch(color=np.array([50,200,50])/255,  label="Adv boxes"),
    ]
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=2, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(f"Frame {stem} — Qualitative Ablation Comparison",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    return fig


def make_filter_panel(
    stem:       str,
    opt1_np:    np.ndarray,
    opt2_np:    np.ndarray,
    clean_np:   np.ndarray,
    lpips_opt1: float,
    lpips_opt2: float,
) -> plt.Figure:
    """3-column panel: Option 1 | Bilateral output | Diff (opt1 vs opt2)."""
    diff_before  = diff_heatmap_np(opt1_np.astype(np.float32)/255,
                                   clean_np.astype(np.float32)/255)
    diff_after   = diff_heatmap_np(opt2_np.astype(np.float32)/255,
                                   clean_np.astype(np.float32)/255)
    diff_filter  = diff_heatmap_np(opt2_np.astype(np.float32)/255,
                                   opt1_np.astype(np.float32)/255)
    shared_vmax  = max(diff_before.max(), diff_after.max())

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    titles_top = [
        (opt1_np,  f"Option 1 (no filter)\nLPIPS={lpips_opt1:.4f}", COL_OPT1),
        (opt2_np,  f"Option 2 (bilateral)\nLPIPS={lpips_opt2:.4f}", COL_OPT2),
        (None,     "Filter effect |Opt2 − Opt1|", "black"),
    ]

    for col, (img, title, col_t) in enumerate(titles_top):
        ax = axes[0, col]
        if img is not None:
            ax.imshow(img)
        else:
            ax.imshow(apply_colormap(diff_filter))
        ax.set_title(title, color=col_t)
        ax.axis("off")

    titles_bot = [
        (diff_before, f"|Opt1 − Clean|\nmax={diff_before.max():.4f}", shared_vmax),
        (diff_after,  f"|Opt2 − Clean|\nmax={diff_after.max():.4f}", shared_vmax),
        (diff_filter, f"|Opt2 − Opt1|\nmax={diff_filter.max():.4f}", None),
    ]
    im_last = None
    for col, (heat, title, vmax) in enumerate(titles_bot):
        ax = axes[1, col]
        im = ax.imshow(heat, cmap=COL_HEATMAP, vmin=0,
                       vmax=vmax if vmax else heat.max() + 1e-8)
        ax.set_title(title, fontsize=8)
        ax.axis("off")
        if col < 2:
            im_last = im

    if im_last is not None:
        fig.colorbar(im_last, ax=axes[1, :2].ravel().tolist(),
                     shrink=0.7, pad=0.02,
                     label="|Δ| mean pixel (shared vmax = Opt1 scale)")

    fig.suptitle(
        f"Frame {stem} — Bilateral Filter Effect\n"
        "Top row: RGB output  |  Bottom row: perturbation heatmaps",
        fontsize=10, fontweight="bold",
    )
    fig.tight_layout()
    return fig


def make_ablation_grid(
    stems:      list[str],
    clean_imgs: dict[str, np.ndarray],
    opt1_imgs:  dict[str, np.ndarray],
    opt2_imgs:  dict[str, np.ndarray],
    dfr_opt1:   dict[str, float],
    lpips_opt1: dict[str, float],
    lpips_opt2: dict[str, float],
) -> plt.Figure:
    """N-frame × 4-col grid: Clean | Opt1 | Opt2 | Heatmap."""
    stems = list(stems)
    n = len(stems)
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Clean", "Option 1 (LPIPS loss)", "Option 2 (+ Bilateral)", "Perturbation |Δ|"]
    col_colors = [COL_CLEAN, COL_OPT1, COL_OPT2, "black"]
    for col, (title, color) in enumerate(zip(col_titles, col_colors)):
        axes[0, col].set_title(title, color=color, fontsize=10, fontweight="bold", pad=8)

    for row, stem in enumerate(stems):
        c_np  = clean_imgs[stem]
        o1_np = opt1_imgs[stem]
        o2_np = opt2_imgs[stem]
        heat  = diff_heatmap_np(o1_np.astype(np.float32)/255,
                                c_np.astype(np.float32)/255)

        axes[row, 0].imshow(c_np)
        axes[row, 0].set_ylabel(stem, fontsize=8, rotation=0,
                                 labelpad=55, va="center")

        axes[row, 1].imshow(o1_np)
        axes[row, 1].text(0.02, 0.96,
                          f"DFR={dfr_opt1.get(stem, float('nan')):+.3f}\n"
                          f"LPIPS={lpips_opt1.get(stem, float('nan')):.3f}",
                          transform=axes[row, 1].transAxes,
                          fontsize=7, va="top", color="white",
                          bbox=dict(boxstyle="round,pad=0.2",
                                    fc=COL_OPT1, alpha=0.75))

        axes[row, 2].imshow(o2_np)
        axes[row, 2].text(0.02, 0.96,
                          f"LPIPS={lpips_opt2.get(stem, float('nan')):.3f}",
                          transform=axes[row, 2].transAxes,
                          fontsize=7, va="top", color="white",
                          bbox=dict(boxstyle="round,pad=0.2",
                                    fc=COL_OPT2, alpha=0.75))

        im = axes[row, 3].imshow(heat, cmap=COL_HEATMAP)
        axes[row, 3].text(0.02, 0.96,
                          f"max={heat.max():.4f}",
                          transform=axes[row, 3].transAxes,
                          fontsize=7, va="top", color="white",
                          bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.6))

        for col in range(4):
            axes[row, col].axis("off")

    fig.suptitle(
        "Qualitative Ablation Grid — Latent Attack on UA-DETRAC\n"
        "Red boxes = clean detections  |  Green boxes = adversarial detections",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def make_shared_heatmap(
    stems:      list[str],
    clean_imgs: dict[str, np.ndarray],
    opt1_imgs:  dict[str, np.ndarray],
    opt2_imgs:  dict[str, np.ndarray],
) -> plt.Figure:
    """Shared-vmax heatmap grid: rows=frames, cols=Opt1 heat / Opt2 heat / delta."""
    n = len(stems)
    all_vals = []
    for stem in stems:
        c = clean_imgs[stem].astype(np.float32) / 255
        o1 = opt1_imgs[stem].astype(np.float32) / 255
        all_vals.append(np.abs(o1 - c).mean())
    shared_vmax = max(all_vals) * 1.2 + 1e-8

    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    axes[0, 0].set_title("Option 1 |adv − clean|",  fontsize=10, fontweight="bold")
    axes[0, 1].set_title("Option 2 |bilateral − clean|", fontsize=10, fontweight="bold")
    axes[0, 2].set_title("Filter gain |Opt1| − |Opt2|", fontsize=10, fontweight="bold")

    im_ref = None
    for row, stem in enumerate(stems):
        c  = clean_imgs[stem].astype(np.float32) / 255
        o1 = opt1_imgs[stem].astype(np.float32) / 255
        o2 = opt2_imgs[stem].astype(np.float32) / 255

        h1 = diff_heatmap_np(o1, c)
        h2 = diff_heatmap_np(o2, c)
        hd = h1 - h2   # positive = filter reduced distortion

        im_ref = axes[row, 0].imshow(h1, cmap="inferno",
                                      vmin=0, vmax=shared_vmax)
        axes[row, 1].imshow(h2, cmap="inferno", vmin=0, vmax=shared_vmax)
        axes[row, 2].imshow(hd, cmap="RdBu_r",
                             vmin=-shared_vmax, vmax=shared_vmax)
        axes[row, 0].set_ylabel(stem, fontsize=8, rotation=0,
                                  labelpad=55, va="center")
        for col in range(3):
            axes[row, col].axis("off")

    if im_ref is not None:
        fig.colorbar(im_ref, ax=axes[:, :2].ravel().tolist(),
                     shrink=0.5, pad=0.02,
                     label="|Δ| mean channel (shared vmax)")

    fig.suptitle("Perturbation Magnitude Heatmaps — Option 1 vs Option 2",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate qualitative report figures for the ablation study."
    )
    ap.add_argument("--config",        default="configs/phase2_option1.yaml")
    ap.add_argument("--data",          default="data/images_50")
    ap.add_argument("--output",        default="results/qualitative")
    ap.add_argument("--frames",        nargs="+",
                    default=["img00001", "img00008", "img00020", "img00026"],
                    help="Specific frame stems to use (without extension)")
    ap.add_argument("--frame_indices", nargs="+", type=int, default=None,
                    help="Frame indices in sorted order (overrides --frames)")
    ap.add_argument("--eps_z",         type=float, default=0.50)
    args = ap.parse_args()

    cfg    = load_config(args.config)
    device = cfg["runtime"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    set_seed(cfg["runtime"]["seed"])

    out_dir    = Path(args.output)
    frames_dir = out_dir / "frames"
    panels_dir = out_dir / "panels"
    frames_dir.mkdir(parents=True, exist_ok=True)
    panels_dir.mkdir(parents=True, exist_ok=True)

    # Select frames
    all_paths = sorted(Path(args.data).glob("*.jpg"))
    if args.frame_indices is not None:
        selected = [all_paths[i] for i in args.frame_indices if i < len(all_paths)]
    else:
        stem_set = set(args.frames)
        selected = [p for p in all_paths if p.stem in stem_set]
    if not selected:
        raise FileNotFoundError(f"No matching frames in {args.data}")
    print(f"Processing {len(selected)} frames: {[p.stem for p in selected]}")

    # Load models
    print("Loading models …")
    detector = YOLOv8Wrapper(cfg["detector"]["weights"], device=device)
    vae_cfg  = cfg["vae"]
    vae = SDVAE(
        model_id=vae_cfg["model_id"],
        scale=vae_cfg["scale"],
        device=device,
        finetuned_weights=vae_cfg.get("finetuned_weights"),
    )
    lpips_fn = MaskedLPIPS(net="alex", device=device)

    acfg = AttackConfig(
        eps_z=args.eps_z,
        gamma=cfg["attack"]["gamma"],
        lambda_p=cfg["attack"]["lambda_p"],
        lambda_r=cfg["attack"]["lambda_r"],
        lr=cfg["attack"]["lr"],
        num_steps=cfg["attack"]["num_steps"],
        early_stop=cfg["attack"]["early_stop"],
        early_stop_margin=cfg["attack"]["early_stop_margin"],
        conf_thr=cfg["detector"]["conf_thr"],
        iou_nms=cfg["detector"]["iou_nms"],
        use_lpips=cfg["attack"].get("use_lpips", True),
        lpips_net=cfg["attack"].get("lpips_net", "alex"),
    )
    attack = LatentObjectAttack(detector, vae, acfg)

    conf_thr = cfg["detector"]["conf_thr"]
    iou_nms  = cfg["detector"]["iou_nms"]

    # Storage for ablation grid
    clean_imgs: dict[str, np.ndarray] = {}
    opt1_imgs:  dict[str, np.ndarray] = {}
    opt2_imgs:  dict[str, np.ndarray] = {}
    dfr_opt1:   dict[str, float]      = {}
    lpips_opt1: dict[str, float]      = {}
    lpips_opt2: dict[str, float]      = {}

    for img_path in selected:
        stem = img_path.stem
        print(f"\n  [{stem}]")

        x          = load_image(img_path).to(device)
        clean_dets = detector.detect_nms(x, conf_thr=conf_thr, iou_thr=iou_nms)
        H, W       = x.shape[2], x.shape[3]

        # --- attack ---
        result  = attack.attack(x)
        x_opt1  = result.x_adv
        M_pix   = boxes_to_pixel_mask(clean_dets, H=H, W=W, device=device) \
                  if clean_dets else torch.zeros(1,1,H,W,device=device)

        opt1_dets = detector.detect_nms(x_opt1, conf_thr=conf_thr, iou_thr=iou_nms)
        n_clean   = len(clean_dets)
        n_opt1    = len(opt1_dets)
        dfr1      = (n_clean - n_opt1) / n_clean if n_clean > 0 else 0.0

        with torch.no_grad():
            lp1 = float(lpips_fn(x_opt1, x, M_pix).item()) if n_clean > 0 else 0.0

        # --- bilateral filter ---
        x_opt2    = bilateral_masked(x_opt1, x, M_pix)
        opt2_dets = detector.detect_nms(x_opt2, conf_thr=conf_thr, iou_thr=iou_nms)
        with torch.no_grad():
            lp2 = float(lpips_fn(x_opt2, x, M_pix).item()) if n_clean > 0 else 0.0

        dfr2 = (n_clean - len(opt2_dets)) / n_clean if n_clean > 0 else 0.0

        print(f"    clean={n_clean}  opt1={n_opt1} (DFR={dfr1:+.3f}, LPIPS={lp1:.4f})"
              f"  opt2={len(opt2_dets)} (DFR={dfr2:+.3f}, LPIPS={lp2:.4f})")

        # Convert to numpy
        c_np  = tensor_to_np(x)
        o1_np = tensor_to_np(x_opt1)
        o2_np = tensor_to_np(x_opt2)
        mask_np = (M_pix[0, 0].cpu().numpy() * 255).astype(np.uint8)

        # Save individual frames
        save_png(c_np,   frames_dir / f"{stem}_clean.png")
        save_png(o1_np,  frames_dir / f"{stem}_opt1_adv.png")
        save_png(o2_np,  frames_dir / f"{stem}_opt2_bilateral.png")
        cv2.imwrite(str(frames_dir / f"{stem}_mask.png"), mask_np)

        # Diff heatmaps (amplified)
        heat1 = diff_heatmap_np(o1_np.astype(np.float32)/255,
                                 c_np.astype(np.float32)/255)
        heat2 = diff_heatmap_np(o2_np.astype(np.float32)/255,
                                 c_np.astype(np.float32)/255)
        amp   = 1.0 / (heat1.max() + 1e-8) * 0.8   # auto-scale to 80% of max
        save_png(apply_colormap(heat1 * amp),
                 frames_dir / f"{stem}_diff_opt1.png")
        save_png(apply_colormap(heat2 * amp),
                 frames_dir / f"{stem}_diff_opt2.png")

        # Store for composite panels
        clean_imgs[stem] = c_np
        opt1_imgs[stem]  = o1_np
        opt2_imgs[stem]  = o2_np
        dfr_opt1[stem]   = dfr1
        lpips_opt1[stem] = lp1
        lpips_opt2[stem] = lp2

        # --- Panel f15: comparison panel ---
        fig15 = make_comparison_panel(
            stem       = stem,
            clean_np   = c_np,
            opt1_np    = o1_np,
            opt2_np    = o2_np,
            clean_dets = clean_dets,
            opt1_dets  = opt1_dets,
            opt2_dets  = opt2_dets,
            lpips_opt1 = lp1,
            lpips_opt2 = lp2,
            dfr_opt1   = dfr1,
            dfr_opt2   = dfr2,
        )
        fig15.savefig(panels_dir / f"f15_comparison_{stem}.png")
        plt.close(fig15)
        print(f"    → f15_comparison_{stem}.png")

        # --- Panel f16: filter effect ---
        fig16 = make_filter_panel(
            stem       = stem,
            opt1_np    = o1_np,
            opt2_np    = o2_np,
            clean_np   = c_np,
            lpips_opt1 = lp1,
            lpips_opt2 = lp2,
        )
        fig16.savefig(panels_dir / f"f16_filter_effect_{stem}.png")
        plt.close(fig16)
        print(f"    → f16_filter_effect_{stem}.png")

    # --- Panel f17: ablation grid ---
    stems_done = list(clean_imgs.keys())
    if stems_done:
        print("\nGenerating ablation grid …")
        fig17 = make_ablation_grid(
            stems      = stems_done,
            clean_imgs = clean_imgs,
            opt1_imgs  = opt1_imgs,
            opt2_imgs  = opt2_imgs,
            dfr_opt1   = dfr_opt1,
            lpips_opt1 = lpips_opt1,
            lpips_opt2 = lpips_opt2,
        )
        fig17.savefig(panels_dir / "f17_ablation_grid.png")
        plt.close(fig17)
        print(f"  → f17_ablation_grid.png")

        # --- Panel f18: shared heatmap ---
        print("Generating perturbation heatmap …")
        fig18 = make_shared_heatmap(
            stems      = stems_done,
            clean_imgs = clean_imgs,
            opt1_imgs  = opt1_imgs,
            opt2_imgs  = opt2_imgs,
        )
        fig18.savefig(panels_dir / "f18_perturbation_heatmap.png")
        plt.close(fig18)
        print(f"  → f18_perturbation_heatmap.png")

    # Summary
    print(f"\n{'='*55}")
    print(f"{'Stem':<12} {'DFR_opt1':>10} {'LPIPS_opt1':>11} {'LPIPS_opt2':>11} {'Δ_LPIPS':>9}")
    print(f"{'-'*55}")
    for stem in stems_done:
        delta_lp = lpips_opt1[stem] - lpips_opt2[stem]
        print(f"{stem:<12} {dfr_opt1[stem]:>+10.4f} "
              f"{lpips_opt1[stem]:>11.4f} "
              f"{lpips_opt2[stem]:>11.4f} "
              f"{delta_lp:>+9.4f}")
    print(f"{'='*55}")
    print(f"\nAll outputs saved to: {out_dir}/")
    print(f"  frames/  — {len(stems_done)*6} individual PNGs")
    print(f"  panels/  — {len(stems_done)*2 + 2} composite figures")


if __name__ == "__main__":
    main()
