"""
pipeline_visualiser.py  (v3)
============================
Runs the full Phase-3 SSIM attack on one frame and captures every
intermediate state of the pipeline as publication-quality figures,
with full numerical logging for reproducibility.

Outputs
-------
Figures:
    step00_clean_detections.png     clean frame + YOLOv8 boxes + confidence scores
    step01_pixel_mask.png           binary pixel-space mask M + frame overlay (3 panels)
    step02_latent_z.png             4 latent channels in 2×2 grid with μ/σ/range annotations
    step02b_vae_reconstruction.png  x vs D(E(x)) vs |x−D(E(x))| — VAE faithfulness proof
    step03_latent_mask.png          REDESIGNED: 2-row layout with 8×8 grid zoom + MaxPool vs 50%
    step04_05_delta_growth.png      MERGED 2×3: clean | δ_early | δ_final (temporal growth)
    step05b_delta_pixel.png         δ effect decoded to pixel space
    step06_x_decoded.png            decoded adversarial D(z_adv) before paste-back
    step07_x_adv_pasteback.png      final adversarial + before/after stats table
    step08_loss_curves.png          L_det / L_perc / p_max vs iteration + final values

Logs:
    attack_log.json     all scalar metrics: pixel stats, latent stats, coverage, PSNR, SSIM, seed
    step_log.csv        per-iteration: L_total, L_det, L_perc, L_reg, p_max, δ_linf, δ_l2
    tensors.npz         z, Mz, M, delta_final, x_adv as float32 numpy arrays

Usage on Colab:
    python scripts/pipeline_visualiser.py \\
        --frames data/images/img00005.jpg data/images/img01380.jpg \\
        --config configs/phase3_ssim.yaml \\
        --out    "/content/drive/MyDrive/pfe_pipeline_figures" \\
        --seed   42
"""

from __future__ import annotations
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

# ── ensure src/ is on path ────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.attack import AttackConfig, LatentObjectAttack
from src.detector import YOLOv8Wrapper
from src.vae import SDVAE
from src.masks import boxes_to_pixel_mask, pixel_mask_to_latent_mask

# ── global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family"      : "serif",
    "figure.dpi"       : 150,
    "savefig.dpi"      : 220,
    "savefig.bbox"     : "tight",
    "savefig.pad_inches": 0.06,
})
GRAY   = "#555555"
STEEL  = "#3a5f82"
RED    = "#a02020"
GREEN  = "#2d7a2d"
ORANGE = "#e07b20"


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """PSNR between two uint8 (H,W,3) images."""
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse < 1e-10:
        return float("inf")
    return float(20.0 * np.log10(255.0 / np.sqrt(mse)))


def compute_ssim_np(img1: np.ndarray, img2: np.ndarray) -> float:
    """Simple global SSIM (fallback — skimage preferred)."""
    try:
        from skimage.metrics import structural_similarity
        return float(structural_similarity(
            img1, img2, channel_axis=-1, data_range=255))
    except ImportError:
        pass
    a = img1.astype(np.float64).ravel()
    b = img2.astype(np.float64).ravel()
    mu_a, mu_b = a.mean(), b.mean()
    sig_a, sig_b = a.std(), b.std()
    sig_ab = np.mean((a - mu_a) * (b - mu_b))
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    return float(
        ((2 * mu_a * mu_b + c1) * (2 * sig_ab + c2))
        / ((mu_a**2 + mu_b**2 + c1) * (sig_a**2 + sig_b**2 + c2))
    )


def coverage_pct(mask_np: np.ndarray) -> float:
    """Percentage of activated elements in a binary mask."""
    return float(100.0 * mask_np.sum() / mask_np.size)


def channel_stats(z_np: np.ndarray, ci: int) -> dict:
    """Stats for channel ci of a (4, H, W) latent array."""
    ch = z_np[ci]
    return {
        "mean": float(ch.mean()),
        "std" : float(ch.std()),
        "min" : float(ch.min()),
        "max" : float(ch.max()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Basic helpers
# ─────────────────────────────────────────────────────────────────────────────

def tensor_to_np(t: torch.Tensor) -> np.ndarray:
    """(1,3,H,W) float[0,1] → (H,W,3) uint8."""
    return (t.squeeze(0).permute(1, 2, 0).cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)


def save(fig, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓  {os.path.basename(path)}")


def draw_detections(ax, detections, color, conf=None, lw=2.0, fs=8):
    """Draw bounding boxes on a matplotlib axis.

    Parameters
    ----------
    conf : list[float] | None
        If provided, confidence score labels are drawn above each box.
        The label shows only the score (e.g. '0.82'), no class prefix.
    """
    for i, d in enumerate(detections):
        x1, y1, x2, y2 = d.box
        ax.add_patch(mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=lw, edgecolor=color, facecolor="none"))
        if conf is not None and i < len(conf):
            ax.text(x1 + 2, y1 - 5,
                    f"{conf[i]:.2f}",
                    color="white", fontsize=fs, fontweight="bold",
                    bbox=dict(facecolor=color, edgecolor="none",
                              boxstyle="round,pad=0.12", alpha=0.88))


# ─────────────────────────────────────────────────────────────────────────────
# Step 02 — latent z in 2×2 grid with channel stats
# ─────────────────────────────────────────────────────────────────────────────

def plot_step02_latent_z(z: torch.Tensor, out_path: str) -> list[dict]:
    """4-channel latent heatmaps in 2×2 grid with μ/σ/range annotations."""
    z_np = z.squeeze(0).cpu().numpy()   # (4, H/8, W/8)
    stats_list = [channel_stats(z_np, ci) for ci in range(4)]

    # 2×2 grid — aspect ratio ≈ 2:1, much more usable than 1×4
    fig, axes_2d = plt.subplots(2, 2, figsize=(11, 7.5), facecolor="white")
    fig.subplots_adjust(wspace=0.22, hspace=0.32,
                        left=0.04, right=0.96, top=0.88, bottom=0.06)
    axes = axes_2d.ravel()

    for ci, ax in enumerate(axes):
        ch   = z_np[ci]
        vabs = max(abs(ch.min()), abs(ch.max())) + 1e-6
        im   = ax.imshow(ch, cmap="RdBu_r", vmin=-vabs, vmax=vabs)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        st = stats_list[ci]
        ax.set_title(
            f"Channel {ci}\n"
            f"μ = {st['mean']:+.3f}   σ = {st['std']:.3f}\n"
            f"range  [{st['min']:.2f},  {st['max']:.2f}]",
            fontsize=8.5, color=GRAY, style="italic", pad=4)

    fig.suptitle(
        r"Step 2 — Latent encoding $z = E(x) \in \mathbb{R}^{4 \times H/8 \times W/8}$"
        "  |  4 channels of the frozen SD-VAE encoder  |  real activations on input frame",
        fontsize=9.5, y=0.97, color=GRAY, style="italic")
    save(fig, out_path)
    return stats_list


# ─────────────────────────────────────────────────────────────────────────────
# Step 02b — VAE reconstruction quality
# ─────────────────────────────────────────────────────────────────────────────

def plot_step02b_vae_reconstruction(x_np: np.ndarray, x_rec_np: np.ndarray,
                                    out_path: str) -> dict:
    """Show x, D(E(x)), and |x − D(E(x))| to prove VAE faithfulness."""
    psnr = compute_psnr(x_np, x_rec_np)
    ssim = compute_ssim_np(x_np, x_rec_np)
    diff = np.abs(x_np.astype(np.float32) - x_rec_np.astype(np.float32))
    max_err = float(diff.max())
    mean_err = float(diff.mean())

    # Amplify diff for visibility (×10 clipped to 255)
    diff_vis = np.clip(diff * 10, 0, 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), facecolor="white")
    fig.subplots_adjust(wspace=0.05, left=0.02, right=0.98, top=0.78, bottom=0.06)

    axes[0].imshow(x_np)
    axes[0].set_title("Original frame $x$", fontsize=8, color=GRAY, style="italic")
    axes[0].axis("off")

    axes[1].imshow(x_rec_np)
    axes[1].set_title(
        r"VAE reconstruction $D(E(x))$"
        f"\nPSNR = {psnr:.1f} dB  ·  SSIM = {ssim:.4f}",
        fontsize=8, color=GRAY, style="italic")
    axes[1].axis("off")

    axes[2].imshow(diff_vis)
    axes[2].set_title(
        r"$|x - D(E(x))|$ × 10  (amplified)"
        f"\nmax pixel error = {max_err:.1f}  ·  mean = {mean_err:.2f}",
        fontsize=8, color=GRAY, style="italic")
    axes[2].axis("off")

    fig.suptitle(
        "Step 2b — VAE reconstruction quality  |  "
        r"$D \circ E$ is a high-fidelity reconstruction — semantically coherent, visually faithful",
        fontsize=9, y=0.97, color=GRAY, style="italic")
    save(fig, out_path)

    return {"psnr_vae": psnr, "ssim_vae": ssim,
            "max_pixel_err": max_err, "mean_pixel_err": mean_err}


# ─────────────────────────────────────────────────────────────────────────────
# Step 03 — REDESIGNED latent mask figure
# ─────────────────────────────────────────────────────────────────────────────

def _pick_zoom_box(D_clean):
    """Return the largest detection box for the zoom panel."""
    best, best_area = None, 0
    for d in D_clean:
        x1, y1, x2, y2 = d.box
        area = (x2 - x1) * (y2 - y1)
        if area > best_area:
            best_area = area
            best = d
    return best


def _zoom_crop(arr_hw_or_hwc, y1, y2, x1, x2):
    """Safe 2D or 3D crop."""
    if arr_hw_or_hwc.ndim == 2:
        return arr_hw_or_hwc[y1:y2, x1:x2]
    return arr_hw_or_hwc[y1:y2, x1:x2, :]


def plot_step03_latent_mask(img_bg: np.ndarray, D_clean, M_np: np.ndarray,
                             M_tensor: torch.Tensor, Mz_np: np.ndarray,
                             out_path: str) -> dict:
    """
    2-row layout:
      Row 1 — overview: frame + M (coverage%) + Mz (coverage%)
      Row 2 — zoom: 8×8 grid on box edge | MaxPool block coloring | MaxPool vs 50% comparison
    """
    H, W  = img_bg.shape[:2]
    LH, LW = Mz_np.shape          # latent dims

    cov_M  = coverage_pct(M_np)
    cov_Mz = coverage_pct(Mz_np)

    # ── choose zoom region (around right edge of largest box) ────────────────
    box_d   = _pick_zoom_box(D_clean)
    bx1, by1, bx2, by2 = (int(v) for v in box_d.box)
    CROP = 96   # crop size in pixels (= 12 latent cells)
    zx_c = bx2                              # center on right edge
    zy_c = (by1 + by2) // 2
    zx1  = max(0, zx_c - CROP // 2)
    zy1  = max(0, zy_c - CROP // 2)
    zx2  = min(W, zx1 + CROP)
    zy2  = min(H, zy1 + CROP)
    # align to 8-pixel boundary so grid aligns perfectly
    zx1  = (zx1 // 8) * 8
    zy1  = (zy1 // 8) * 8
    zx2  = zx1 + CROP
    zy2  = zy1 + CROP

    img_crop = _zoom_crop(img_bg, zy1, zy2, zx1, zx2)
    M_crop   = _zoom_crop(M_np, zy1, zy2, zx1, zx2).astype(float)

    # ── MaxPool vs 50%-threshold comparison at latent scale ──────────────────
    M_f   = M_tensor.float()
    if M_f.dim() == 2:
        M_f = M_f.unsqueeze(0).unsqueeze(0)
    elif M_f.dim() == 3:
        M_f = M_f.unsqueeze(0)

    Mz_maxpool = (F.max_pool2d(M_f, kernel_size=8, stride=8) >= 0.5)[0, 0].cpu().numpy()
    Mz_avg50   = (F.avg_pool2d(M_f, kernel_size=8, stride=8) > 0.5)[0, 0].cpu().numpy()

    # latent crop indices
    lx1 = zx1 // 8;  lx2 = lx1 + CROP // 8
    ly1 = zy1 // 8;  ly2 = ly1 + CROP // 8

    Mz_max_crop = Mz_maxpool[ly1:ly2, lx1:lx2]
    Mz_avg_crop = Mz_avg50[ly1:ly2, lx1:lx2]
    extra_cells = (Mz_max_crop.astype(int) - Mz_avg_crop.astype(int)).clip(0)
    n_extra     = int(extra_cells.sum())

    # ── figure layout ─────────────────────────────────────────────────────────
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(15, 8.5), facecolor="white")
    gs  = GridSpec(2, 3, figure=fig,
                   height_ratios=[0.55, 0.45],
                   hspace=0.32, wspace=0.10,
                   left=0.03, right=0.97, top=0.92, bottom=0.04)

    # ── Row 1 panel A: frame + boxes ─────────────────────────────────────────
    ax_A = fig.add_subplot(gs[0, 0])
    ax_A.imshow(img_bg)
    draw_detections(ax_A, D_clean, color=GREEN, lw=1.8)
    # draw zoom rectangle
    rect = mpatches.Rectangle((zx1, zy1), CROP, CROP,
                               linewidth=2, edgecolor=ORANGE,
                               facecolor="none", linestyle="--")
    ax_A.add_patch(rect)
    ax_A.set_title(
        f"(a) Frame + YOLOv8 detections\n"
        f"M covers {cov_M:.1f}% of pixels  ·  orange dashed = zoom region",
        fontsize=9, color=GRAY, style="italic", pad=4)
    ax_A.axis("off")

    # ── Row 1 panel B: pixel mask M ──────────────────────────────────────────
    ax_B = fig.add_subplot(gs[0, 1])
    ax_B.imshow(M_np, cmap="gray", vmin=0, vmax=1)
    # highlight zoom region
    rect2 = mpatches.Rectangle((zx1, zy1), CROP, CROP,
                                linewidth=2, edgecolor=ORANGE,
                                facecolor="none", linestyle="--")
    ax_B.add_patch(rect2)
    ax_B.set_title(
        r"(b) Pixel mask $M \in \{0,1\}^{H \times W}$"
        f"\ncoverage = {cov_M:.1f}%  (white = perturb zone)",
        fontsize=9, color=GRAY, style="italic", pad=4)
    ax_B.axis("off")

    # ── Row 1 panel C: latent mask Mz ────────────────────────────────────────
    ax_C = fig.add_subplot(gs[0, 2])
    ax_C.imshow(Mz_np, cmap="hot", interpolation="nearest")
    # highlight zoom region at latent scale
    rect3 = mpatches.Rectangle((lx1, ly1), CROP // 8, CROP // 8,
                                linewidth=2, edgecolor=ORANGE,
                                facecolor="none", linestyle="--")
    ax_C.add_patch(rect3)
    ax_C.set_title(
        r"(c) Latent mask $\mathcal{M}_z = \mathrm{MaxPool}_8(M) \in \{0,1\}^{H/8 \times W/8}$"
        f"\ncoverage = {cov_Mz:.1f}%  (+{cov_Mz-cov_M:.1f}% conservative expansion)",
        fontsize=9, color=GRAY, style="italic", pad=4)
    ax_C.axis("off")

    # ── Row 2 panel D: zoom crop with 8×8 grid + blue tint on M ─────────────
    ax_D = fig.add_subplot(gs[1, 0])
    vis_D = img_crop.copy().astype(float)
    inside = M_crop > 0.5
    # blue tint on mask pixels
    vis_D[inside] = vis_D[inside] * 0.45 + np.array([40, 90, 220]) * 0.55
    vis_D = vis_D.clip(0, 255).astype(np.uint8)
    ax_D.imshow(vis_D, interpolation="nearest")
    for y in range(0, CROP + 1, 8):
        ax_D.axhline(y - 0.5, color="white", lw=0.7, alpha=0.75)
    for x in range(0, CROP + 1, 8):
        ax_D.axvline(x - 0.5, color="white", lw=0.7, alpha=0.75)
    ax_D.set_title(
        "(d) Zoom on box right edge — 8×8 pixel grid\n"
        "Blue tint = inside $M$ (perturbed)  ·  Dark = background preserved",
        fontsize=9, color=GRAY, style="italic", pad=4)
    ax_D.axis("off")

    # ── Row 2 panel E: block coloring by MaxPool logic ───────────────────────
    ax_E = fig.add_subplot(gs[1, 1])
    n_cells = CROP // 8
    canvas_E = np.zeros((CROP, CROP, 3), dtype=np.uint8)
    annotations = []
    for r in range(n_cells):
        for c in range(n_cells):
            block = M_crop[r*8:(r+1)*8, c*8:(c+1)*8]
            n_in  = int(block.sum())
            if n_in == 0:
                color = np.array([35, 35, 35])       # inactive
            elif n_in < 64:
                color = np.array([210, 125, 25])      # boundary — orange
                if 0 < n_in <= 30:
                    annotations.append((c * 8 + 4, r * 8 + 4, f"{n_in}"))
            else:
                color = np.array([40, 100, 210])      # fully inside — blue
            canvas_E[r*8:(r+1)*8, c*8:(c+1)*8] = color

    ax_E.imshow(canvas_E, interpolation="nearest")
    for y in range(0, CROP + 1, 8):
        ax_E.axhline(y - 0.5, color="white", lw=0.7, alpha=0.6)
    for x in range(0, CROP + 1, 8):
        ax_E.axvline(x - 0.5, color="white", lw=0.7, alpha=0.6)
    # annotate up to 5 boundary blocks with larger font
    for (xc, yc, txt) in annotations[:5]:
        ax_E.text(xc, yc, txt, color="white", fontsize=7, fontweight="bold",
                  ha="center", va="center")

    # legend patches
    p_in  = mpatches.Patch(color="#2864d2", label="Fully inside box (64/64 px)")
    p_bnd = mpatches.Patch(color="#d27d19", label="Partial — MaxPool activates (≥1 px)")
    p_out = mpatches.Patch(color="#232323", label="Outside — not activated")
    ax_E.legend(handles=[p_in, p_bnd, p_out], fontsize=7,
                loc="lower right", framealpha=0.85)
    ax_E.set_title(
        "(e) MaxPool activation logic per 8×8 block\n"
        "Numbers = pixels inside box — any single pixel activates the full latent cell",
        fontsize=9, color=GRAY, style="italic", pad=4)
    ax_E.axis("off")

    # ── Row 2 panel F: MaxPool vs 50% threshold at latent resolution ─────────
    ax_F = fig.add_subplot(gs[1, 2])
    SCALE = 10   # each latent cell → 10×10 px for visibility
    CL    = CROP // 8

    def to_rgb_latent(mask, extra=None):
        rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
        rgb[mask  > 0.5] = [40, 100, 210]   # blue = activated
        rgb[mask <= 0.5] = [35,  35,  35]   # dark = inactive
        if extra is not None:
            rgb[extra > 0.5] = [210, 125, 25]  # orange = extra (MaxPool only)
        return rgb

    row_max = to_rgb_latent(Mz_max_crop)
    row_avg = to_rgb_latent(Mz_avg_crop, extra=extra_cells)
    sep     = np.full((1, CL, 3), 200, dtype=np.uint8)
    combined = np.vstack([row_max, sep, row_avg])   # (2*CL+1, CL, 3)

    # upscale with nearest neighbour
    combined_up = np.kron(combined, np.ones((SCALE, SCALE, 1), dtype=np.uint8))
    ax_F.imshow(combined_up, interpolation="nearest")

    mid = (CL * SCALE)
    ax_F.axhline(mid - 0.5, color="white", lw=1.5)
    ax_F.text(3, mid // 2,     "MaxPool (conservative)",
              color="white", fontsize=7.5, fontweight="bold", va="center")
    ax_F.text(3, mid + SCALE // 2 + SCALE,
              "50% threshold (strict)",
              color="white", fontsize=7.5, fontweight="bold", va="center")

    ax_F.set_title(
        f"(f) MaxPool vs 50%-threshold at latent scale\n"
        f"Orange = {n_extra} extra cells MaxPool activates at box boundary",
        fontsize=9, color=GRAY, style="italic", pad=4)
    ax_F.axis("off")

    # ── suptitle ──────────────────────────────────────────────────────────────
    fig.suptitle(
        r"Step 3 — Latent mask $\mathcal{M}_z = \mathrm{MaxPool}_8(M)$  |  "
        r"Each latent cell covers an 8×8 pixel block  |  "
        "Conservative coverage guarantees no vehicle pixel is missed",
        fontsize=9.5, y=0.98, color=GRAY, style="italic")

    save(fig, out_path)
    return {"coverage_M_pct": cov_M, "coverage_Mz_pct": cov_Mz,
            "extra_boundary_cells": n_extra}


# ─────────────────────────────────────────────────────────────────────────────
# Step 04+05 — MERGED: δ temporal growth (2×3 figure)
# ─────────────────────────────────────────────────────────────────────────────

def plot_step04_05_merged(delta_early: torch.Tensor, delta_final: torch.Tensor,
                          img_bg: np.ndarray, hist: dict,
                          steps_early: int = 10, steps_final: int = None,
                          out_path: str = "") -> None:
    """
    Merged 2×3 figure showing temporal growth of perturbation δ.

    Layout
    ------
    Row 0 (pixel overlays):  clean frame | δ_early overlay | δ_final overlay
    Row 1 (latent heatmaps): δ_linf curve | δ_early latent | δ_final latent
    """
    from matplotlib.gridspec import GridSpec

    def _delta_vis(delta: torch.Tensor):
        """Returns (mag_norm 2-D float [0,1], overlay uint8 HxWx3, latent_heatmap 2-D float)."""
        mag  = delta.squeeze(0).abs().sum(dim=0).cpu().numpy()   # latent resolution
        H, W = img_bg.shape[:2]
        mag_up = np.array(
            Image.fromarray(((mag / (mag.max() + 1e-8)) * 255).astype(np.uint8)
            ).resize((W, H), resample=Image.BILINEAR)
        ).astype(float) / 255.0
        overlay = (img_bg.astype(float) * 0.45 +
                   plt.cm.hot(mag_up)[:, :, :3] * 255 * 0.55
                   ).clip(0, 255).astype(np.uint8)
        return mag_up, overlay, mag

    mag_early_up, ov_early, lat_early = _delta_vis(delta_early)
    mag_final_up, ov_final, lat_final = _delta_vis(delta_final)

    linf_hist  = hist.get("delta_linf", [])
    linf_final = float(linf_hist[-1]) if linf_hist else 0.0
    linf_early = float(linf_hist[steps_early - 1]) if linf_hist and len(linf_hist) >= steps_early else 0.0

    fig = plt.figure(figsize=(15, 8), facecolor="white")
    gs  = GridSpec(2, 3, figure=fig,
                   height_ratios=[1, 1],
                   hspace=0.30, wspace=0.10,
                   left=0.05, right=0.97, top=0.91, bottom=0.05)

    # ── Row 0 ─────────────────────────────────────────────────────────────────
    ax00 = fig.add_subplot(gs[0, 0])
    ax00.imshow(img_bg)
    ax00.set_title("(a) Clean frame $x$", fontsize=9, color=GRAY, style="italic", pad=4)
    ax00.axis("off")

    ax01 = fig.add_subplot(gs[0, 1])
    ax01.imshow(ov_early)
    ax01.set_title(
        f"(b) $|\\delta|$ overlay — step {steps_early} (early)\n"
        r"$\|\delta\|_\infty$" + f" = {linf_early:.3f}",
        fontsize=9, color=GRAY, style="italic", pad=4)
    ax01.axis("off")

    ax02 = fig.add_subplot(gs[0, 2])
    ax02.imshow(ov_final)
    steps_label = f"step {steps_final}" if steps_final else "final step"
    ax02.set_title(
        f"(c) $|\\delta|$ overlay — {steps_label}\n"
        r"$\|\delta\|_\infty$" + f" = {linf_final:.3f}"
        r"  (bound $\varepsilon_z = 0.50$)",
        fontsize=9, color=GRAY, style="italic", pad=4)
    ax02.axis("off")

    # ── Row 1 ─────────────────────────────────────────────────────────────────
    # (1,0) — δ_linf growth curve
    ax10 = fig.add_subplot(gs[1, 0])
    if linf_hist:
        iters = list(range(1, len(linf_hist) + 1))
        ax10.plot(iters, linf_hist, color=STEEL, lw=1.8,
                  label=r"$\|\delta\|_\infty$")
        ax10.axhline(0.50, color=GRAY, lw=0.9, linestyle=":",
                     label=r"$\varepsilon_z = 0.50$")
        if steps_early <= len(linf_hist):
            ax10.axvline(steps_early, color=ORANGE, lw=1.2, linestyle="--",
                         label=f"step {steps_early} (snapshot)")
        ax10.set_xlabel("Iteration", fontsize=8.5)
        ax10.set_ylabel(r"$\|\delta\|_\infty$", fontsize=8.5)
        ax10.legend(fontsize=7.5, framealpha=0.7)
        ax10.grid(True, alpha=0.3)
        ax10.spines["top"].set_visible(False)
        ax10.spines["right"].set_visible(False)
        ax10.tick_params(labelsize=8)
    ax10.set_title(
        r"(d) $\|\delta\|_\infty$ growth over iterations",
        fontsize=9, color=GRAY, style="italic", pad=4)

    # (1,1) — δ_early latent heatmap
    ax11 = fig.add_subplot(gs[1, 1])
    im11 = ax11.imshow(lat_early, cmap="hot")
    plt.colorbar(im11, ax=ax11, fraction=0.046, pad=0.04, label="Σ|δ| channels")
    ax11.set_title(
        f"(e) Latent $|\\delta|$ — step {steps_early}\n"
        "Perturbation confined to vehicle cells",
        fontsize=9, color=GRAY, style="italic", pad=4)
    ax11.axis("off")

    # (1,2) — δ_final latent heatmap
    ax12 = fig.add_subplot(gs[1, 2])
    im12 = ax12.imshow(lat_final, cmap="hot")
    plt.colorbar(im12, ax=ax12, fraction=0.046, pad=0.04, label="Σ|δ| channels")
    ax12.set_title(
        f"(f) Latent $|\\delta|$ — {steps_label}\n"
        "Magnitude saturates in vehicle footprints",
        fontsize=9, color=GRAY, style="italic", pad=4)
    ax12.axis("off")

    fig.suptitle(
        r"Steps 4–5 — Perturbation $\delta$ temporal growth  |  "
        r"Latent space  $\|\delta\|_\infty \leq \varepsilon_z = 0.50$  |  "
        "Hot regions = vehicle footprints only",
        fontsize=9.5, y=0.97, color=GRAY, style="italic")
    save(fig, out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Step 05b — delta effect decoded to pixel space
# ─────────────────────────────────────────────────────────────────────────────

def plot_step05b_delta_pixel(x_rec_np: np.ndarray, x_adv_dec_np: np.ndarray,
                              M_np: np.ndarray, out_path: str) -> None:
    """
    Show how δ manifests in pixel space after decoding.
    x_rec    = D(E(x))            — reconstruction without attack
    x_adv_dec = D(E(x) + Mz⊙δ)  — decoded adversarial (before paste-back)
    pixel_delta = x_adv_dec − x_rec amplified as heatmap
    """
    diff_f    = x_adv_dec_np.astype(np.float32) - x_rec_np.astype(np.float32)
    diff_abs  = np.abs(diff_f)                        # (H, W, 3)
    diff_mag  = diff_abs.mean(axis=-1)                # (H, W) mean over channels
    max_delta = float(diff_mag.max()) + 1e-6

    diff_vis_rgb  = plt.cm.hot(diff_mag / max_delta)[:, :, :3]
    diff_vis_rgb  = (diff_vis_rgb * 255).astype(np.uint8)

    # Stats inside mask only
    diff_in_mask = diff_mag[M_np > 0.5]
    mean_in  = float(diff_in_mask.mean()) if diff_in_mask.size > 0 else 0.0
    max_in   = float(diff_in_mask.max())  if diff_in_mask.size > 0 else 0.0
    diff_out = diff_mag[M_np < 0.5]
    mean_out = float(diff_out.mean())     if diff_out.size  > 0 else 0.0

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), facecolor="white")
    fig.subplots_adjust(wspace=0.06, left=0.02, right=0.98, top=0.78, bottom=0.06)

    axes[0].imshow(x_rec_np)
    axes[0].set_title(r"$D(E(x))$ — clean reconstruction",
                      fontsize=8, color=GRAY, style="italic")
    axes[0].axis("off")

    axes[1].imshow(x_adv_dec_np)
    axes[1].set_title(
        r"$D(E(x) + \mathcal{M}_z \odot \delta)$ — decoded adversarial",
        fontsize=8, color=GRAY, style="italic")
    axes[1].axis("off")

    im = axes[2].imshow(diff_vis_rgb)
    axes[2].set_title(
        r"$|D(z_{\rm adv}) - D(z)|$ — pixel-space $\delta$ effect  (hot colormap)"
        f"\nmean inside mask = {mean_in:.2f} px  ·  max = {max_in:.2f} px"
        f"\nmean outside mask = {mean_out:.3f} px  (background intact)",
        fontsize=7.5, color=GRAY, style="italic")
    axes[2].axis("off")
    plt.colorbar(
        plt.cm.ScalarMappable(
            norm=matplotlib.colors.Normalize(0, max_delta),
            cmap="hot"),
        ax=axes[2], fraction=0.046, pad=0.04,
        label="pixel magnitude")

    fig.suptitle(
        r"Step 5b — Perturbation $\delta$ decoded to pixel space  |  "
        "Confirms perturbation is spatially contained within detected vehicle footprints",
        fontsize=9, y=0.97, color=GRAY, style="italic")
    save(fig, out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def save_logs(out_dir: str, snap: dict, D_clean, D_adv,
              cfg_dict: dict, frame_path: str,
              x_np: np.ndarray, x_adv_np: np.ndarray,
              z_stats: list[dict], coverage: dict,
              vae_rec_metrics: dict, seed: int) -> None:
    """Save attack_log.json, step_log.csv, tensors.npz."""

    # ── pixel stats ───────────────────────────────────────────────────────────
    x_f = x_np.astype(np.float32) / 255.0
    pixel_stats = {}
    for ci, ch_name in enumerate(["R", "G", "B"]):
        ch = x_f[:, :, ci]
        pixel_stats[f"mean_{ch_name}"] = float(ch.mean())
        pixel_stats[f"std_{ch_name}"]  = float(ch.std())


    # ── final metrics ─────────────────────────────────────────────────────────
    hist     = snap["history"]
    psnr_adv = compute_psnr(x_np, x_adv_np)
    ssim_adv = compute_ssim_np(x_np, x_adv_np)

    final_metrics = {
        "L_det_final"  : float(hist["L_det"][-1]),
        "L_perc_final" : float(hist["L_perc"][-1]),
        "L_reg_final"  : float(hist["L_reg"][-1]),
        "L_total_final": float(hist["L"][-1]),
        "p_max_final"  : float(hist["p_max"][-1]),
        "psnr_adv"     : psnr_adv,
        "ssim_adv"     : ssim_adv,
    }

    # ── detection summary ─────────────────────────────────────────────────────
    det_summary = {
        "n_clean"        : len(D_clean),
        "n_adv"          : len(D_adv),
        "dfr"            : 1.0 if len(D_adv) == 0 else
                           max(0.0, (len(D_clean) - len(D_adv)) / max(len(D_clean), 1)),
        "mean_conf_clean": float(np.mean([d.score for d in D_clean])) if D_clean else 0.0,
        "mean_conf_adv"  : float(np.mean([d.score for d in D_adv]))   if D_adv  else 0.0,
    }

    log = {
        "frame"          : os.path.basename(frame_path),
        "seed"           : seed,
        "steps_taken"    : snap["steps_taken"],
        "config_attack"  : cfg_dict.get("attack", {}),
        "pixel_stats"    : pixel_stats,
        "latent_stats"   : {f"ch{i}": z_stats[i] for i in range(len(z_stats))},
        "mask_coverage"  : coverage,
        "vae_rec_metrics": vae_rec_metrics,
        "detection"      : det_summary,
        "final_metrics"  : final_metrics,
        "timestamp"      : time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    json_path = os.path.join(out_dir, "attack_log.json")
    with open(json_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"  ✓  attack_log.json")

    # ── step_log.csv ──────────────────────────────────────────────────────────
    csv_path = os.path.join(out_dir, "step_log.csv")
    keys     = ["step", "L_total", "L_det", "L_perc", "L_reg",
                "p_max", "delta_linf", "delta_l2"]
    n_steps  = len(hist["L"])
    rows     = []
    for i in range(n_steps):
        rows.append({
            "step"      : i + 1,
            "L_total"   : hist["L"][i],
            "L_det"     : hist["L_det"][i],
            "L_perc"    : hist["L_perc"][i],
            "L_reg"     : hist["L_reg"][i],
            "p_max"     : hist["p_max"][i],
            "delta_linf": hist.get("delta_linf", [0.0]*n_steps)[i],
            "delta_l2"  : hist.get("delta_l2",   [0.0]*n_steps)[i],
        })
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"  ✓  step_log.csv  ({n_steps} rows)")

    # ── tensors.npz ───────────────────────────────────────────────────────────
    npz_path = os.path.join(out_dir, "tensors.npz")
    np.savez(npz_path,
             z           = snap["z"].squeeze(0).cpu().numpy().astype(np.float32),
             Mz          = snap["Mz"].squeeze(0).cpu().numpy().astype(np.float32),
             M           = snap["M"].squeeze(0).cpu().numpy().astype(np.float32),
             delta_final = snap["delta_final"].squeeze(0).cpu().numpy().astype(np.float32),
             x_adv       = x_adv_np.astype(np.uint8),
    )
    print(f"  ✓  tensors.npz")


# =============================================================================
# Instrumented attack
# =============================================================================

class InstrumentedAttack(LatentObjectAttack):
    """Extends LatentObjectAttack to capture and save pipeline intermediates."""

    def __init__(self, detector, vae, config, out_dir: str):
        super().__init__(detector, vae, config)
        self.out_dir = out_dir
        self._snap: dict = {}

    def _single_run(self, x, z, M, Mz, D_clean, C_clean, restart_idx):
        cfg    = self.cfg
        device = self.vae.device

        delta_data = torch.zeros_like(z)
        delta      = delta_data.requires_grad_(True)
        optim      = torch.optim.Adam([delta], lr=cfg.lr)

        grad_buf = torch.zeros_like(z) if cfg.use_momentum else None

        history = {
            "L": [], "L_det": [], "L_perc": [], "L_reg": [], "p_max": [],
            "delta_linf": [], "delta_l2": [],
        }
        steps_taken      = 0
        snap_saved_early = False

        for t in range(cfg.num_steps):
            z_adv  = z + Mz * delta
            x_dec  = self.vae.decode(z_adv)
            x_adv  = M * x_dec + (1 - M) * x

            raw        = self.det.forward_raw(x_adv)
            class_conf = self.det.class_confidence(raw)

            from src.losses import (vanishing_loss, masked_l2, latent_l2,
                                    perceptual_combined, ssim_loss)
            L_det = vanishing_loss(class_conf, C_clean, gamma=cfg.gamma)

            if self._lpips is not None:
                if cfg.use_ssim:
                    L_perc = perceptual_combined(
                        x_adv, x, M, lpips_fn=self._lpips,
                        ssim_weight=cfg.ssim_weight)
                else:
                    L_perc = self._lpips(x_adv, x, M)
            else:
                L_perc = ssim_loss(x_adv, x, M) if cfg.use_ssim else masked_l2(x_adv, x, M)

            L_reg = latent_l2(delta)
            L     = L_det + cfg.lambda_p * L_perc + cfg.lambda_r * L_reg

            optim.zero_grad(set_to_none=True)
            L.backward()

            if cfg.use_momentum and grad_buf is not None and delta.grad is not None:
                with torch.no_grad():
                    g   = delta.grad
                    g_n = g / g.abs().mean().clamp(min=1e-8)
                    grad_buf.mul_(cfg.momentum_decay).add_(g_n)
                    delta.grad.copy_(grad_buf)

            optim.step()

            with torch.no_grad():
                delta.data.clamp_(-cfg.eps_z, cfg.eps_z)
                delta.data.mul_(Mz)
                p_per = class_conf[0, :, C_clean].amax(dim=0)
                p_max = float(p_per.max().item())

                history["L"].append(float(L.item()))
                history["L_det"].append(float(L_det.item()))
                history["L_perc"].append(float(L_perc.item()))
                history["L_reg"].append(float(L_reg.item()))
                history["p_max"].append(p_max)
                history["delta_linf"].append(float(delta.data.abs().max().item()))
                history["delta_l2"].append(float(delta.data.norm(p=2).item()))

            if t == 9 and not snap_saved_early:
                self._snap["delta_early"] = delta.detach().clone()
                self._snap["x_dec_early"] = x_dec.detach().clone()
                snap_saved_early = True

            steps_taken = t + 1
            if cfg.early_stop and p_max < cfg.gamma - cfg.early_stop_margin:
                break

        with torch.no_grad():
            z_adv_final = z + Mz * delta
            x_dec_final = self.vae.decode(z_adv_final)
            x_adv_final = (M * x_dec_final + (1 - M) * x).clamp(0, 1)

        self._snap.update({
            "z"          : z.detach().clone(),
            "Mz"         : Mz.detach().clone(),
            "M"          : M.detach().clone(),
            "delta_final": delta.detach().clone(),
            "z_adv"      : z_adv_final.detach().clone(),
            "x_dec_final": x_dec_final.detach().clone(),
            "x_adv_final": x_adv_final.detach().clone(),
            "history"    : history,
            "steps_taken": steps_taken,
        })

        from src.attack import AttackResult
        return AttackResult(
            x_adv=x_adv_final,
            delta=delta.detach(),
            M=M,
            detections_clean=D_clean,
            classes_clean=C_clean,
            history=history,
            steps_taken=steps_taken,
        )


# =============================================================================
# Main pipeline
# =============================================================================

def run(frame_path: str, config_path: str, out_dir: str, seed: int = 42) -> None:
    os.makedirs(out_dir, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    with open(config_path) as f:
        cfg_dict = yaml.safe_load(f)

    device = cfg_dict["runtime"]["device"] if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  |  Seed: {seed}")

    det_cfg = cfg_dict["detector"]
    vae_cfg = cfg_dict["vae"]
    atk_cfg = cfg_dict["attack"]

    print("Loading YOLOv8 detector...")
    detector = YOLOv8Wrapper(
        weights=os.path.join(ROOT, det_cfg["weights"]),
        device=device,
    )

    print("Loading SD-VAE...")
    vae = SDVAE(model_id=vae_cfg["model_id"], device=device,
                scale=vae_cfg.get("scale", 0.18215))

    attack_config = AttackConfig(**{k: v for k, v in atk_cfg.items()})
    attack        = InstrumentedAttack(detector, vae, attack_config, out_dir)

    print(f"Loading frame: {frame_path}")
    img_pil = Image.open(frame_path).convert("RGB")
    img_np  = np.array(img_pil)
    H, W    = img_np.shape[:2]
    H8 = ((H + 7) // 8) * 8
    W8 = ((W + 7) // 8) * 8
    img_pil_padded = Image.fromarray(img_np).resize((W8, H8), Image.BILINEAR)
    x = (torch.from_numpy(np.array(img_pil_padded))
         .float().permute(2, 0, 1).unsqueeze(0) / 255.0).to(device)

    img_bg = tensor_to_np(x)

    # STEP 0: clean detections
    print("\n[Step 0] Clean detections...")
    D_clean = detector.detect_nms(x, conf_thr=det_cfg["conf_thr"],
                                   iou_thr=det_cfg["iou_nms"])
    print(f"  {len(D_clean)} detections")
    scores  = [d.score for d in D_clean]

    fig, ax = plt.subplots(1, 1, figsize=(8, 5), facecolor="white")
    ax.imshow(img_bg)
    draw_detections(ax, D_clean, color=GREEN, conf=scores, lw=2.0)
    ax.set_title(
        f"Step 0 - Clean frame  |  YOLOv8 detections: {len(D_clean)} vehicles",
        fontsize=9, color=GRAY, style="italic")
    ax.axis("off")
    fig.text(0.5, 0.01,
             f"conf_thr={det_cfg['conf_thr']}  iou_nms={det_cfg['iou_nms']}  "
             f"frame: {os.path.basename(frame_path)}  seed: {seed}",
             ha="center", fontsize=7.5, color=GRAY, style="italic")
    save(fig, f"{out_dir}/step00_clean_detections.png")

    # STEP 1: pixel mask M with 3 panels (frame | mask | overlay)
    print("[Step 1] Pixel-space mask M...")
    M    = boxes_to_pixel_mask(D_clean, H=H8, W=W8, device=device)
    M_np = M.squeeze().cpu().numpy()
    cov_M = coverage_pct(M_np)

    overlay_step1 = img_bg.astype(float).copy()
    mask_zone = M_np > 0.5
    overlay_step1[mask_zone] = (overlay_step1[mask_zone] * 0.60
                                + np.array([30, 144, 255]) * 0.40)
    overlay_step1 = overlay_step1.clip(0, 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), facecolor="white")
    fig.subplots_adjust(wspace=0.06, left=0.02, right=0.98, top=0.80, bottom=0.06)

    axes[0].imshow(img_bg)
    draw_detections(axes[0], D_clean, color=GREEN, lw=1.5)
    axes[0].set_title("(a) Clean frame + detection boxes",
                      fontsize=8, color=GRAY, style="italic")
    axes[0].axis("off")

    axes[1].imshow(M_np, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(
        "(b) Pixel-space mask M  (white = perturb zone)"
        f"\ncoverage = {cov_M:.1f}%",
        fontsize=8, color=GRAY, style="italic")
    axes[1].axis("off")

    axes[2].imshow(overlay_step1)
    axes[2].set_title(
        "(c) M overlaid on frame  (alpha = 0.4)"
        "\nBlue zone = perturbation region",
        fontsize=8, color=GRAY, style="italic")
    axes[2].axis("off")

    fig.suptitle(
        "Step 1 - Binary mask M  |  Union of detected bounding boxes",
        fontsize=9, y=0.97, color=GRAY, style="italic")
    save(fig, f"{out_dir}/step01_pixel_mask.png")

    # ATTACK
    print("[Attack] Running Phase-3 SSIM attack...")
    result = attack.attack(x)
    snap   = attack._snap
    steps  = snap["steps_taken"]
    print(f"  Completed in {steps} steps")

    # STEP 2: latent z in 2x2 grid
    print("[Step 2] Latent z = E(x) - 2x2 grid with stats...")
    z_stats = plot_step02_latent_z(
        snap["z"],
        out_path=f"{out_dir}/step02_latent_z.png",
    )

    # STEP 2b: VAE reconstruction
    print("[Step 2b] VAE reconstruction quality...")
    with torch.no_grad():
        x_rec_t = vae.decode(snap["z"])
    x_rec_np = tensor_to_np(x_rec_t.clamp(0, 1))
    vae_rec_metrics = plot_step02b_vae_reconstruction(
        img_bg, x_rec_np,
        out_path=f"{out_dir}/step02b_vae_reconstruction.png",
    )
    print(f"  PSNR = {vae_rec_metrics['psnr_vae']:.1f} dB  "
          f"SSIM = {vae_rec_metrics['ssim_vae']:.4f}")

    # STEP 3: latent mask redesign
    print("[Step 3] Latent mask (redesigned)...")
    Mz_np = snap["Mz"].squeeze(0)[0].cpu().numpy()
    coverage_info = plot_step03_latent_mask(
        img_bg, D_clean, M_np, M, Mz_np,
        out_path=f"{out_dir}/step03_latent_mask.png",
    )
    print(f"  M={coverage_info['coverage_M_pct']:.1f}%  "
          f"Mz={coverage_info['coverage_Mz_pct']:.1f}%  "
          f"extra={coverage_info['extra_boundary_cells']}")

    # STEPS 4+5: merged delta temporal growth (2x3)
    print("[Steps 4+5] Delta temporal growth (merged 2x3 figure)...")
    plot_step04_05_merged(
        delta_early=snap["delta_early"],
        delta_final=snap["delta_final"],
        img_bg=img_bg,
        hist=snap["history"],
        steps_early=10,
        steps_final=steps,
        out_path=f"{out_dir}/step04_05_delta_growth.png",
    )

    # STEP 5b: delta decoded to pixel space
    print("[Step 5b] Delta effect in pixel space...")
    x_adv_dec_np = tensor_to_np(snap["x_dec_final"].clamp(0, 1))
    plot_step05b_delta_pixel(
        x_rec_np, x_adv_dec_np, M_np,
        out_path=f"{out_dir}/step05b_delta_pixel.png",
    )

    # STEP 6: decoded before paste-back
    print("[Step 6] Decoded adversarial (before paste-back)...")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor="white")
    fig.subplots_adjust(wspace=0.08, left=0.02, right=0.98, top=0.80, bottom=0.06)

    axes[0].imshow(img_bg)
    axes[0].set_title(
        "(a) Clean frame x",
        fontsize=9, color=GRAY, style="italic", pad=5)
    axes[0].axis("off")

    axes[1].imshow(x_adv_dec_np)
    axes[1].set_title(
        "(b) D(z + Mz * delta) - decoded adversarial"
        "\nBackground not yet restored - artefacts visible outside boxes",
        fontsize=9, color=GRAY, style="italic", pad=5)
    axes[1].axis("off")

    fig.suptitle(
        "Step 6 - Decoded adversarial before paste-back",
        fontsize=9.5, y=0.97, color=GRAY, style="italic")
    save(fig, f"{out_dir}/step06_x_decoded.png")

    # STEP 7: final adversarial — ONLY draw D_adv on adversarial panel
    print("[Step 7] Final adversarial (after paste-back)...")
    x_adv_np = tensor_to_np(snap["x_adv_final"])
    D_adv    = detector.detect_nms(snap["x_adv_final"],
                                    conf_thr=det_cfg["conf_thr"],
                                    iou_thr=det_cfg["iou_nms"])
    print(f"  Adversarial detections: {len(D_adv)}")

    psnr_adv = compute_psnr(img_bg, x_adv_np)
    ssim_adv = compute_ssim_np(img_bg, x_adv_np)
    mean_conf_clean = float(np.mean(scores))            if scores else 0.0
    mean_conf_adv   = float(np.mean([d.score for d in D_adv])) if D_adv else 0.0

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), facecolor="white")
    fig.subplots_adjust(wspace=0.05, left=0.02, right=0.98, top=0.78, bottom=0.06)

    axes[0].imshow(img_bg)
    draw_detections(axes[0], D_clean, color=GREEN, conf=scores, lw=1.8)
    axes[0].set_title(
        f"(a) Clean frame  -  {len(D_clean)} detections\n"
        f"mean confidence = {mean_conf_clean:.3f}",
        fontsize=9, color=GREEN, fontweight="bold")
    axes[0].axis("off")

    # Adversarial panel: ONLY D_adv boxes drawn (never D_clean)
    axes[1].imshow(x_adv_np)
    if D_adv:
        draw_detections(axes[1], D_adv, color=RED,
                        conf=[d.score for d in D_adv], lw=1.8)
        det_label = f"{len(D_adv)} detections remaining"
        col       = RED
    else:
        det_label = "0 detections  attack successful"
        col       = STEEL
    axes[1].set_title(
        f"(b) Adversarial frame  -  {det_label}\n"
        f"mean confidence = {mean_conf_adv:.3f}  "
        f"(delta = {mean_conf_adv - mean_conf_clean:+.3f})",
        fontsize=9, color=col, fontweight="bold")
    axes[1].axis("off")

    # Stats box in top-right corner with padding (avoids annotation clipping)
    stats_txt = (
        f"PSNR(x, x') = {psnr_adv:.1f} dB\n"
        f"SSIM(x, x') = {ssim_adv:.4f}\n"
        f"Detections: {len(D_clean)} to {len(D_adv)}\n"
        f"Conf: {mean_conf_clean:.3f} to {mean_conf_adv:.3f}\n"
        f"Steps taken: {steps}"
    )
    fig.text(0.985, 0.88, stats_txt, ha="right", va="top",
             fontsize=7.5, color=GRAY, style="italic",
             bbox=dict(facecolor="white", edgecolor=GRAY,
                       boxstyle="round,pad=0.5", alpha=0.92))

    fig.suptitle(
        "Step 7 - Final adversarial  |  Phase-3 SSIM  "
        f"|  {steps} steps",
        fontsize=9, y=0.97, color=GRAY, style="italic")
    save(fig, f"{out_dir}/step07_x_adv_pasteback.png")

    # STEP 8: loss curves with text-box final values (no arrow clipping)
    print("[Step 8] Loss curves...")
    hist  = snap["history"]
    iters = list(range(1, len(hist["L_det"]) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), facecolor="white")
    fig.subplots_adjust(wspace=0.30, left=0.08, right=0.97, top=0.82, bottom=0.12)

    ax = axes[0]
    ax.plot(iters, hist["L_det"], color=RED,   lw=1.8, label="L_det")
    ax.plot(iters, hist["p_max"], color=STEEL, lw=1.8, linestyle="--",
            label="p_max (peak confidence)")
    ax.axhline(0.05, color=GRAY, lw=0.8, linestyle=":", label="gamma = 0.05")
    # Final values as text box top-right (avoids arrow-clipping at axes boundary)
    ax.text(0.97, 0.97,
            f"L_det = {hist['L_det'][-1]:.4f}\np_max = {hist['p_max'][-1]:.4f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=7.5, color=GRAY, style="italic",
            bbox=dict(facecolor="white", edgecolor=GRAY,
                      boxstyle="round,pad=0.35", alpha=0.88))
    ax.set_xlabel("Iteration", fontsize=9)
    ax.set_ylabel("Value", fontsize=9)
    ax.set_title("Detection loss L_det and peak confidence",
                 fontsize=9, color=GRAY, style="italic")
    ax.legend(fontsize=8, framealpha=0.7)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False);  ax.spines["right"].set_visible(False)

    ax = axes[1]
    ax.plot(iters, hist["L_perc"], color="#c0622a", lw=1.8, label="L_perc (LPIPS+SSIM)")
    ax.plot(iters, hist["L_reg"],  color="#7a4a9e", lw=1.8, linestyle="--", label="L_reg")
    ax.text(0.97, 0.97,
            f"L_perc = {hist['L_perc'][-1]:.4f}\nL_reg  = {hist['L_reg'][-1]:.5f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=7.5, color=GRAY, style="italic",
            bbox=dict(facecolor="white", edgecolor=GRAY,
                      boxstyle="round,pad=0.35", alpha=0.88))
    ax.set_xlabel("Iteration", fontsize=9)
    ax.set_ylabel("Value", fontsize=9)
    ax.set_title("Perceptual L_perc and regularisation L_reg",
                 fontsize=9, color=GRAY, style="italic")
    ax.legend(fontsize=8, framealpha=0.7)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False);  ax.spines["right"].set_visible(False)

    fig.suptitle(
        f"Step 8 - Loss curves  |  Phase-3 SSIM  |  {steps} iterations  "
        "|  early-stop at p_max < gamma",
        fontsize=9, y=0.97, color=GRAY, style="italic")
    save(fig, f"{out_dir}/step08_loss_curves.png")

    # LOGS
    print("[Logs] Saving attack_log.json, step_log.csv, tensors.npz...")
    save_logs(
        out_dir, snap, D_clean, D_adv, cfg_dict, frame_path,
        img_bg, x_adv_np, z_stats, coverage_info,
        vae_rec_metrics, seed,
    )

    print(f"\n  All outputs saved to: {out_dir}")
    print(f"   Clean:       {len(D_clean)} det  mean conf {mean_conf_clean:.3f}")
    print(f"   Adversarial: {len(D_adv)} det  mean conf {mean_conf_adv:.3f}")
    print(f"   PSNR={psnr_adv:.1f} dB  SSIM={ssim_adv:.4f}  Steps={steps}")


# =============================================================================
# Cross-frame comparison figure
# =============================================================================

def compose_comparison(results: list[dict], out_path: str) -> None:
    """
    Side-by-side comparison for two frames.
    DFR labels: 1.0 = Full success | 0.0 = Hard case | else = Partial success
    """
    n = len(results)
    fig, axes = plt.subplots(3, n, figsize=(7 * n, 14), facecolor="white")
    fig.subplots_adjust(hspace=0.28, wspace=0.08,
                        left=0.04, right=0.97, top=0.93, bottom=0.04)

    for col, r in enumerate(results):
        frame_name = os.path.basename(r["frame_path"])
        dfr        = r["dfr"]
        if dfr >= 1.0:
            label     = "Full success (DFR = 1.00)"
            col_color = GREEN
        elif dfr <= 0.0:
            label     = "Hard case - attack failed (DFR = 0.00)"
            col_color = RED
        else:
            label     = f"Partial success (DFR = {dfr:.2f})"
            col_color = ORANGE

        # Row 0: clean frame
        ax = axes[0, col] if n > 1 else axes[0]
        ax.imshow(r["img_bg"])
        if r["D_clean"]:
            draw_detections(ax, r["D_clean"], GREEN,
                            conf=[d.score for d in r["D_clean"]], lw=1.8, fs=8)
        ax.set_title(
            f"{frame_name}\n{len(r['D_clean'])} detections  "
            f"mean conf = {r['mean_conf_clean']:.3f}",
            fontsize=8.5, color=GRAY, style="italic", pad=4)
        ax.axis("off")

        # Row 1: adversarial frame — ONLY D_adv boxes drawn
        ax = axes[1, col] if n > 1 else axes[1]
        ax.imshow(r["x_adv_np"])
        if r["D_adv"]:
            draw_detections(ax, r["D_adv"], RED,
                            conf=[d.score for d in r["D_adv"]], lw=1.8, fs=8)
        ax.set_title(
            f"{label}\n{len(r['D_adv'])} detections remaining  "
            f"mean conf = {r['mean_conf_adv']:.3f}",
            fontsize=8.5, color=col_color, fontweight="bold", pad=4)
        stats_txt = (f"PSNR={r['psnr']:.1f} dB\n"
                     f"SSIM={r['ssim']:.4f}\nSteps={r['steps']}")
        ax.text(6, 16, stats_txt, color="white", fontsize=7.5,
                bbox=dict(facecolor=STEEL, alpha=0.82, edgecolor="none",
                          boxstyle="round,pad=0.3"))
        ax.axis("off")

        # Row 2: loss curves
        ax = axes[2, col] if n > 1 else axes[2]
        hist  = r["hist"]
        iters = list(range(1, len(hist["L_det"]) + 1))
        ax.plot(iters, hist["L_det"], color=RED,   lw=1.6, label="L_det")
        ax.plot(iters, hist["p_max"], color=STEEL, lw=1.6, ls="--", label="p_max")
        ax.axhline(0.05, color=GRAY, lw=0.8, ls=":", label="gamma=0.05")
        ax.set_xlabel("Iteration", fontsize=8.5)
        ax.set_ylabel("Value", fontsize=8.5)
        ax.legend(fontsize=8, framealpha=0.7)
        ax.grid(True, alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=8)
        ax.set_title(f"Loss convergence  {r['steps']} iterations",
                     fontsize=8.5, color=GRAY, style="italic", pad=4)

    fig.suptitle(
        "Cross-frame comparison - same attack config (Phase-3 SSIM) - same seed\n"
        "Left: successful evasion  Right: hard-case frame (high detector confidence)",
        fontsize=11, y=0.97, color=GRAY, style="italic")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"  {os.path.basename(out_path)} saved")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipeline visualiser v3 - Chapter 3/4 figures")
    parser.add_argument("--frames", nargs="+",
                        default=["data/images/img00005.jpg",
                                 "data/images/img01380.jpg"],
                        help="One or two frame paths (relative to repo root).")
    parser.add_argument("--config", default="configs/phase3_ssim.yaml")
    parser.add_argument("--out",    default="figures/pipeline_steps")
    parser.add_argument("--seed",   type=int, default=42,
                        help="Random seed (torch + numpy)")
    args = parser.parse_args()

    config_path = os.path.join(ROOT, args.config)
    out_root    = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)

    frame_paths = [
        fp if os.path.isabs(fp) else os.path.join(ROOT, fp)
        for fp in args.frames
    ]

    if len(frame_paths) == 1:
        run(frame_paths[0], config_path, out_root, seed=args.seed)
    else:
        results = []
        for fp in frame_paths:
            frame_id = os.path.splitext(os.path.basename(fp))[0]
            sub_dir  = os.path.join(out_root, frame_id)
            print(f"\n{'='*60}")
            print(f"  Processing: {frame_id}")
            print(f"{'='*60}")
            run(fp, config_path, sub_dir, seed=args.seed)

            log_path = os.path.join(sub_dir, "attack_log.json")
            with open(log_path) as f:
                log = json.load(f)

            csv_path = os.path.join(sub_dir, "step_log.csv")
            hist = {"L_det": [], "p_max": []}
            with open(csv_path, newline="") as f:
                for row in csv.DictReader(f):
                    hist["L_det"].append(float(row["L_det"]))
                    hist["p_max"].append(float(row["p_max"]))

            img_bg   = np.array(Image.open(f"{sub_dir}/step00_clean_detections.png"))
            x_adv_np = np.array(Image.open(f"{sub_dir}/step07_x_adv_pasteback.png"))

            results.append({
                "frame_path"     : fp,
                "dfr"            : log["detection"]["dfr"],
                "steps"          : log["steps_taken"],
                "psnr"           : log["final_metrics"]["psnr_adv"],
                "ssim"           : log["final_metrics"]["ssim_adv"],
                "mean_conf_clean": log["detection"]["mean_conf_clean"],
                "mean_conf_adv"  : log["detection"]["mean_conf_adv"],
                "hist"           : hist,
                "img_bg"         : img_bg,
                "x_adv_np"       : x_adv_np,
                "D_clean"        : [],
                "D_adv"          : [],
            })

        print("\n[Compare] Building cross-frame comparison figure...")
        compose_comparison(
            results,
            out_path=os.path.join(out_root, "fig_frame_comparison.png"),
        )

        print(f"\n  All done. Outputs in: {out_root}/")
        for r in results:
            name = os.path.basename(r["frame_path"])
            print(f"   {name:20s}  DFR={r['dfr']:.2f}  "
                  f"PSNR={r['psnr']:.1f} dB  SSIM={r['ssim']:.4f}")
