"""
pipeline_visualiser.py  (v2)
============================
Runs the full Phase-3 SSIM attack on one frame and captures every
intermediate state of the pipeline as publication-quality figures,
with full numerical logging for reproducibility.

Outputs
-------
Figures:
    step00_clean_detections.png     clean frame + YOLOv8 boxes + confidence scores
    step01_pixel_mask.png           binary pixel-space mask M with coverage %
    step02_latent_z.png             4 latent channels with μ/σ/range annotations per channel
    step02b_vae_reconstruction.png  x vs D(E(x)) vs |x−D(E(x))| — VAE faithfulness proof
    step03_latent_mask.png          REDESIGNED: 2-row layout with 8×8 grid zoom + MaxPool vs 50% comparison
    step04_delta_early.png          perturbation δ at step 10
    step05_delta_final.png          perturbation δ at final step (latent space)
    step05b_delta_pixel.png         δ effect decoded to pixel space
    step06_x_decoded.png            decoded adversarial D(z_adv) before paste-back
    step07_x_adv_pasteback.png      final adversarial + before/after stats table
    step08_loss_curves.png          L_det / L_perc / p_max vs iteration + final values annotated
    step09_delta_overlay.png        |δ| magnitude overlaid on clean frame
    fig_ch3_pipeline_grid.png       composed 3×3 grid for Chapter 3

Logs:
    attack_log.json     all scalar metrics: pixel stats, latent stats, coverage, PSNR, SSIM, seed
    step_log.csv        per-iteration: L_total, L_det, L_perc, L_reg, p_max, δ_linf, δ_l2
    tensors.npz         z, Mz, M, delta_final, x_adv as float32 numpy arrays

Usage on Colab:
    python scripts/pipeline_visualiser.py \\
        --frame  data/images/img00005.jpg \\
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


def draw_detections(ax, detections, color, label_prefix="", conf=None, lw=2.0, fs=8):
    """Draw bounding boxes on a matplotlib axis."""
    for i, d in enumerate(detections):
        x1, y1, x2, y2 = d.box
        ax.add_patch(mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=lw, edgecolor=color, facecolor="none"))
        if conf is not None and i < len(conf):
            ax.text(x1 + 2, y1 - 5,
                    f"{label_prefix}{conf[i]:.2f}",
                    color="white", fontsize=fs, fontweight="bold",
                    bbox=dict(facecolor=color, edgecolor="none",
                              boxstyle="round,pad=0.12", alpha=0.88))


# ─────────────────────────────────────────────────────────────────────────────
# Step 02 — latent z with channel stats
# ─────────────────────────────────────────────────────────────────────────────

def plot_step02_latent_z(z: torch.Tensor, out_path: str) -> list[dict]:
    """4-channel latent heatmaps with μ/σ/range annotations. Returns stats list."""
    z_np = z.squeeze(0).cpu().numpy()   # (4, H/8, W/8)
    stats_list = [channel_stats(z_np, ci) for ci in range(4)]

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4), facecolor="white")
    fig.subplots_adjust(wspace=0.10, left=0.02, right=0.98, top=0.78, bottom=0.14)

    for ci, ax in enumerate(axes):
        ch   = z_np[ci]
        vabs = max(abs(ch.min()), abs(ch.max())) + 1e-6
        im   = ax.imshow(ch, cmap="RdBu_r", vmin=-vabs, vmax=vabs)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        st = stats_list[ci]
        ax.set_title(
            f"Channel {ci}\n"
            f"μ={st['mean']:+.2f}  σ={st['std']:.2f}\n"
            f"[{st['min']:.2f}, {st['max']:.2f}]",
            fontsize=7.5, color=GRAY, style="italic", pad=3)

    fig.suptitle(
        r"Step 2 — Latent encoding $z = E(x) \in \mathbb{R}^{4 \times H/8 \times W/8}$"
        "  |  4 channels of the frozen SD-VAE encoder  |  real activations on input frame",
        fontsize=9, y=0.97, color=GRAY, style="italic")
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
        r"$D \circ E \approx \mathrm{Id}$ confirms the encoder is perceptually lossless",
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
                   hspace=0.28, wspace=0.10,
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
        fontsize=8, color=GRAY, style="italic", pad=3)
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
        fontsize=8, color=GRAY, style="italic", pad=3)
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
        fontsize=8, color=GRAY, style="italic", pad=3)
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
        "Blue tint = inside $M$ (will be perturbed)  ·  Dark = background preserved",
        fontsize=7.5, color=GRAY, style="italic", pad=3)
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
    # annotate up to 5 boundary blocks
    for (xc, yc, txt) in annotations[:5]:
        ax_E.text(xc, yc, txt, color="white", fontsize=5.5, fontweight="bold",
                  ha="center", va="center")

    # legend patches
    p_in  = mpatches.Patch(color="#2864d2", label="Fully inside box (64/64 px)")
    p_bnd = mpatches.Patch(color="#d27d19", label="Partial — MaxPool activates (≥1 px)")
    p_out = mpatches.Patch(color="#232323", label="Outside — not activated")
    ax_E.legend(handles=[p_in, p_bnd, p_out], fontsize=5.5,
                loc="lower right", framealpha=0.85)
    ax_E.set_title(
        "(e) MaxPool activation logic per 8×8 block\n"
        "Numbers = pixels inside box — any single pixel activates the full latent cell",
        fontsize=7.5, color=GRAY, style="italic", pad=3)
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
              color="white", fontsize=6, fontweight="bold", va="center")
    ax_F.text(3, mid + SCALE // 2 + SCALE,
              "50% threshold (strict)",
              color="white", fontsize=6, fontweight="bold", va="center")

    ax_F.set_title(
        f"(f) MaxPool vs 50%-threshold at latent scale\n"
        f"Orange = {n_extra} extra cells MaxPool activates at box boundary",
        fontsize=7.5, color=GRAY, style="italic", pad=3)
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

    # Amplify ×8 for visibility
    diff_vis_gray = (diff_mag / max_delta * 255).clip(0, 255).astype(np.uint8)
    diff_vis_rgb  = plt.cm.hot(diff_mag / max_delta)[:, :, :3]
    diff_vis_rgb  = (diff_vis_rgb * 255).astype(np.uint8)

    # Masked region outline overlay
    mask_outline = np.zeros_like(diff_mag)
    mask_outline[M_np > 0.5] = 1.0

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
# visualise_delta helper (unchanged logic, shared by step04/05/09)
# ─────────────────────────────────────────────────────────────────────────────

def visualise_delta(delta: torch.Tensor, img_bg: np.ndarray,
                    title: str, out_path: str, alpha: float = 0.65) -> None:
    delta_mag = delta.squeeze(0).abs().sum(dim=0).cpu().numpy()
    H, W      = img_bg.shape[:2]
    mag_pil   = Image.fromarray(
        ((delta_mag / (delta_mag.max() + 1e-8)) * 255).astype(np.uint8)
    ).resize((W, H), resample=Image.BILINEAR)
    mag_up    = np.array(mag_pil).astype(float) / 255.0

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), facecolor="white")
    fig.subplots_adjust(wspace=0.05, left=0.02, right=0.98, top=0.82, bottom=0.06)

    axes[0].imshow(img_bg)
    axes[0].set_title("Clean frame", fontsize=8, color=GRAY, style="italic")
    axes[0].axis("off")

    axes[1].imshow(img_bg, alpha=0.45)
    axes[1].imshow(mag_up, cmap="hot", alpha=alpha, vmin=0, vmax=1)
    axes[1].set_title(r"$|\delta|$ magnitude overlay", fontsize=8, color=GRAY, style="italic")
    axes[1].axis("off")

    im = axes[2].imshow(delta_mag, cmap="hot")
    axes[2].set_title(r"$|\delta|$ (latent resolution)", fontsize=8, color=GRAY, style="italic")
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=9, y=0.97, color=GRAY, style="italic")
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


# ─────────────────────────────────────────────────────────────────────────────
# Instrumented attack — subclass that saves intermediate states
# ─────────────────────────────────────────────────────────────────────────────

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
        steps_taken    = 0
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


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run(frame_path: str, config_path: str, out_dir: str, seed: int = 42) -> None:
    os.makedirs(out_dir, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    with open(config_path) as f:
        cfg_dict = yaml.safe_load(f)

    device = cfg_dict["runtime"]["device"] if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  ·  Seed: {seed}")

    det_cfg = cfg_dict["detector"]
    vae_cfg = cfg_dict["vae"]
    atk_cfg = cfg_dict["attack"]

    print("Loading YOLOv8 detector…")
    detector = YOLOv8Wrapper(
        weights=os.path.join(ROOT, det_cfg["weights"]),
        device=device,
    )

    print("Loading SD-VAE…")
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

    img_bg = tensor_to_np(x)   # uint8 (H8, W8, 3) — used for all vis

    # ── STEP 0: clean detections ──────────────────────────────────────────────
    print("\n[Step 0] Clean detections…")
    D_clean = detector.detect_nms(x, conf_thr=det_cfg["conf_thr"],
                                   iou_thr=det_cfg["iou_nms"])
    print(f"  {len(D_clean)} detections")
    scores  = [d.score for d in D_clean]

    fig, ax = plt.subplots(1, 1, figsize=(8, 5), facecolor="white")
    ax.imshow(img_bg)
    draw_detections(ax, D_clean, color=GREEN, label_prefix="car ", conf=scores, lw=2.0)
    ax.set_title(
        f"Step 0 — Clean frame  |  YOLOv8 detections: {len(D_clean)} vehicles",
        fontsize=9, color=GRAY, style="italic")
    ax.axis("off")
    fig.text(0.5, 0.01,
             f"conf_thr={det_cfg['conf_thr']}  ·  iou_nms={det_cfg['iou_nms']}  ·  "
             f"frame: {os.path.basename(frame_path)}  ·  seed: {seed}",
             ha="center", fontsize=7.5, color=GRAY, style="italic")
    save(fig, f"{out_dir}/step00_clean_detections.png")

    # ── STEP 1: pixel mask M ─────────────────────────────────────────────────
    print("[Step 1] Pixel-space mask M…")
    M    = boxes_to_pixel_mask(D_clean, H=H8, W=W8, device=device)
    M_np = M.squeeze().cpu().numpy()
    cov_M = coverage_pct(M_np)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor="white")
    axes[0].imshow(img_bg)
    draw_detections(axes[0], D_clean, color=GREEN, lw=1.5)
    axes[0].set_title("Clean frame with boxes", fontsize=8, color=GRAY, style="italic")
    axes[0].axis("off")
    axes[1].imshow(M_np, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(
        rf"Pixel-space mask $M$  (white = perturb zone)  ·  coverage = {cov_M:.1f}%",
        fontsize=8, color=GRAY, style="italic")
    axes[1].axis("off")
    fig.suptitle(
        r"Step 1 — Binary mask $M \in \{0,1\}^{H \times W}$"
        "  |  Union of detected bounding boxes",
        fontsize=9, y=0.97, color=GRAY, style="italic")
    save(fig, f"{out_dir}/step01_pixel_mask.png")

    # ── ATTACK ───────────────────────────────────────────────────────────────
    print("[Attack] Running Phase-3 SSIM attack…")
    result = attack.attack(x)
    snap   = attack._snap
    steps  = snap["steps_taken"]
    print(f"  Completed in {steps} steps")

    # ── STEP 2: latent z with channel stats ───────────────────────────────────
    print("[Step 2] Latent z = E(x) with stats…")
    z_stats = plot_step02_latent_z(
        snap["z"],
        out_path=f"{out_dir}/step02_latent_z.png",
    )

    # ── STEP 2b: VAE reconstruction ───────────────────────────────────────────
    print("[Step 2b] VAE reconstruction quality…")
    with torch.no_grad():
        x_rec_t   = vae.decode(snap["z"])
    x_rec_np      = tensor_to_np(x_rec_t.clamp(0, 1))
    vae_rec_metrics = plot_step02b_vae_reconstruction(
        img_bg, x_rec_np,
        out_path=f"{out_dir}/step02b_vae_reconstruction.png",
    )
    print(f"  PSNR = {vae_rec_metrics['psnr_vae']:.1f} dB  ·  "
          f"SSIM = {vae_rec_metrics['ssim_vae']:.4f}")

    # ── STEP 3: latent mask redesign ──────────────────────────────────────────
    print("[Step 3] Latent mask (redesigned)…")
    Mz_np = snap["Mz"].squeeze(0)[0].cpu().numpy()
    coverage_info = plot_step03_latent_mask(
        img_bg, D_clean, M_np, M, Mz_np,
        out_path=f"{out_dir}/step03_latent_mask.png",
    )
    print(f"  M = {coverage_info['coverage_M_pct']:.1f}%  "
          f"Mz = {coverage_info['coverage_Mz_pct']:.1f}%  "
          f"extra boundary cells = {coverage_info['extra_boundary_cells']}")

    # ── STEP 4: delta early ───────────────────────────────────────────────────
    print("[Step 4] Delta at step 10 (early)…")
    visualise_delta(
        snap["delta_early"], img_bg,
        title=r"Step 4 — Perturbation $\delta$ at iteration 10  (early accumulation)",
        out_path=f"{out_dir}/step04_delta_early.png",
    )

    # ── STEP 5: delta final in latent ─────────────────────────────────────────
    print("[Step 5] Delta final (latent space)…")
    hist_linf = snap["history"].get("delta_linf", [])
    linf_final = f"{hist_linf[-1]:.3f}" if hist_linf else "—"
    visualise_delta(
        snap["delta_final"], img_bg,
        title=(r"Step 5 — Perturbation $\delta$ at final iteration "
               f"(step {steps})  |  "
               r"$\|\delta\|_\infty$" + f" = {linf_final}"
               r"  (bound $\varepsilon_z = 0.50$)"),
        out_path=f"{out_dir}/step05_delta_final.png",
    )

    # ── STEP 5b: delta decoded to pixel space ─────────────────────────────────
    print("[Step 5b] Delta effect in pixel space…")
    x_adv_dec_np = tensor_to_np(snap["x_dec_final"].clamp(0, 1))
    plot_step05b_delta_pixel(
        x_rec_np, x_adv_dec_np, M_np,
        out_path=f"{out_dir}/step05b_delta_pixel.png",
    )

    # ── STEP 6: decoded before paste-back ────────────────────────────────────
    print("[Step 6] Decoded adversarial (before paste-back)…")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor="white")
    axes[0].imshow(img_bg)
    axes[0].set_title(r"Clean frame $x$", fontsize=8, color=GRAY, style="italic")
    axes[0].axis("off")
    axes[1].imshow(x_adv_dec_np)
    axes[1].set_title(
        r"$D(z + \mathcal{M}_z \odot \delta)$ — decoded adversarial"
        "\n(background not yet restored — artefacts visible outside boxes)",
        fontsize=8, color=GRAY, style="italic")
    axes[1].axis("off")
    fig.suptitle(
        r"Step 6 — Decoded adversarial $\hat{x}_{\mathrm{adv}} = D(z + \mathcal{M}_z \odot \delta)$"
        "  |  Before paste-back",
        fontsize=9, y=0.97, color=GRAY, style="italic")
    save(fig, f"{out_dir}/step06_x_decoded.png")

    # ── STEP 7: final adversarial + stats table ───────────────────────────────
    print("[Step 7] Final adversarial (after paste-back)…")
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
        f"Clean frame  —  {len(D_clean)} detections\n"
        f"mean confidence = {mean_conf_clean:.3f}",
        fontsize=9, color=GREEN, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(x_adv_np)
    if D_adv:
        draw_detections(axes[1], D_adv, color=RED,
                        conf=[d.score for d in D_adv], lw=1.8)
        det_label = f"{len(D_adv)} detections remaining"
        col       = RED
    else:
        det_label = "0 detections  ✓  attack successful"
        col       = STEEL
    axes[1].set_title(
        f"Adversarial frame  —  {det_label}\n"
        f"mean confidence = {mean_conf_adv:.3f}  "
        f"(Δ = {mean_conf_adv - mean_conf_clean:+.3f})",
        fontsize=9, color=col, fontweight="bold")
    axes[1].axis("off")

    # Numerical stats box
    stats_txt = (
        f"PSNR(x, x') = {psnr_adv:.1f} dB\n"
        f"SSIM(x, x') = {ssim_adv:.4f}\n"
        f"Detections: {len(D_clean)} → {len(D_adv)}\n"
        f"Conf: {mean_conf_clean:.3f} → {mean_conf_adv:.3f}\n"
        f"Steps taken: {steps}"
    )
    fig.text(0.985, 0.08, stats_txt, ha="right", va="bottom",
             fontsize=7.5, color=GRAY, style="italic",
             bbox=dict(facecolor="white", edgecolor=GRAY,
                       boxstyle="round,pad=0.4", alpha=0.90))

    fig.suptitle(
        r"Step 7 — Final adversarial $x' = M \odot \hat{x}_{\mathrm{adv}} + (1-M) \odot x$"
        f"  |  Phase-3 SSIM  ·  {steps} steps",
        fontsize=9, y=0.97, color=GRAY, style="italic")
    save(fig, f"{out_dir}/step07_x_adv_pasteback.png")

    # ── STEP 8: loss curves with final-value annotations ─────────────────────
    print("[Step 8] Loss curves…")
    hist  = snap["history"]
    iters = list(range(1, len(hist["L_det"]) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), facecolor="white")
    fig.subplots_adjust(wspace=0.30, left=0.08, right=0.97, top=0.82, bottom=0.12)

    ax = axes[0]
    ax.plot(iters, hist["L_det"], color=RED,   lw=1.8,
            label=r"$\mathcal{L}_{\mathrm{det}}$")
    ax.plot(iters, hist["p_max"], color=STEEL, lw=1.8, linestyle="--",
            label=r"$p_{\max}$ (peak confidence)")
    ax.axhline(0.05, color=GRAY, lw=0.8, linestyle=":", label=r"$\gamma = 0.05$")
    # Annotate final values
    ax.annotate(f"L_det={hist['L_det'][-1]:.4f}",
                xy=(iters[-1], hist["L_det"][-1]),
                xytext=(-35, 8), textcoords="offset points",
                fontsize=7, color=RED,
                arrowprops=dict(arrowstyle="-", color=RED, lw=0.8))
    ax.annotate(f"p_max={hist['p_max'][-1]:.4f}",
                xy=(iters[-1], hist["p_max"][-1]),
                xytext=(-35, -14), textcoords="offset points",
                fontsize=7, color=STEEL,
                arrowprops=dict(arrowstyle="-", color=STEEL, lw=0.8))
    ax.set_xlabel("Iteration", fontsize=9)
    ax.set_ylabel("Value", fontsize=9)
    ax.set_title(r"Detection loss $\mathcal{L}_{\mathrm{det}}$ & peak confidence",
                 fontsize=9, color=GRAY, style="italic")
    ax.legend(fontsize=8, framealpha=0.7)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False);  ax.spines["right"].set_visible(False)

    ax = axes[1]
    ax.plot(iters, hist["L_perc"], color="#c0622a", lw=1.8,
            label=r"$\mathcal{L}_{\mathrm{perc}}$ (LPIPS+SSIM)")
    ax.plot(iters, hist["L_reg"],  color="#7a4a9e", lw=1.8, linestyle="--",
            label=r"$\mathcal{L}_{\mathrm{reg}}$")
    ax.annotate(f"L_perc={hist['L_perc'][-1]:.4f}",
                xy=(iters[-1], hist["L_perc"][-1]),
                xytext=(-50, 8), textcoords="offset points",
                fontsize=7, color="#c0622a",
                arrowprops=dict(arrowstyle="-", color="#c0622a", lw=0.8))
    ax.set_xlabel("Iteration", fontsize=9)
    ax.set_ylabel("Value", fontsize=9)
    ax.set_title(r"Perceptual $\mathcal{L}_{\mathrm{perc}}$ & regularisation $\mathcal{L}_{\mathrm{reg}}$",
                 fontsize=9, color=GRAY, style="italic")
    ax.legend(fontsize=8, framealpha=0.7)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False);  ax.spines["right"].set_visible(False)

    fig.suptitle(
        f"Step 8 — Loss curves  |  Phase-3 SSIM  ·  {steps} iterations  ·  "
        r"early-stop at $p_{\max} < \gamma$",
        fontsize=9, y=0.97, color=GRAY, style="italic")
    save(fig, f"{out_dir}/step08_loss_curves.png")

    # ── STEP 9: delta overlay ─────────────────────────────────────────────────
    print("[Step 9] Delta magnitude overlay…")
    visualise_delta(
        snap["delta_final"], img_bg,
        title=(r"Step 9 — Perturbation magnitude $|\delta|$ overlaid on clean frame"
               "\nHot regions = high latent perturbation = vehicle footprints only"),
        out_path=f"{out_dir}/step09_delta_overlay.png",
        alpha=0.72,
    )

    # ── LOGS ─────────────────────────────────────────────────────────────────
    print("[Logs] Saving attack_log.json, step_log.csv, tensors.npz…")
    save_logs(
        out_dir, snap, D_clean, D_adv, cfg_dict, frame_path,
        img_bg, x_adv_np, z_stats, coverage_info,
        vae_rec_metrics, seed,
    )

    # ── COMPOSED GRID ─────────────────────────────────────────────────────────
    print("[Compose] Building Chapter 3 pipeline grid…")
    _compose_grid(
        img_bg=img_bg, x_adv_np=x_adv_np, x_rec_np=x_rec_np,
        M_np=M_np, Mz_np=Mz_np,
        delta_final=snap["delta_final"],
        D_clean=D_clean, D_adv=D_adv,
        hist=hist, steps=steps,
        psnr=psnr_adv, ssim=ssim_adv,
        out_path=f"{out_dir}/fig_ch3_pipeline_grid.png",
    )

    print(f"\n✓  All outputs saved to: {out_dir}")
    print(f"   Clean:       {len(D_clean)} det  ·  mean conf {mean_conf_clean:.3f}")
    print(f"   Adversarial: {len(D_adv)}  det  ·  mean conf {mean_conf_adv:.3f}")
    print(f"   PSNR={psnr_adv:.1f} dB  ·  SSIM={ssim_adv:.4f}  ·  Steps={steps}")


# ─────────────────────────────────────────────────────────────────────────────
# Composed Chapter 3 grid
# ─────────────────────────────────────────────────────────────────────────────

def _compose_grid(img_bg, x_adv_np, x_rec_np, M_np, Mz_np, delta_final,
                  D_clean, D_adv, hist, steps, psnr, ssim, out_path):
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(16, 10.5), facecolor="white")
    gs  = GridSpec(3, 3, figure=fig,
                   hspace=0.38, wspace=0.12,
                   left=0.04, right=0.97, top=0.93, bottom=0.05)

    H_img, W_img = img_bg.shape[:2]

    def ax_img(pos, img, title, color=GRAY):
        ax = fig.add_subplot(gs[pos])
        ax.imshow(img)
        ax.set_title(title, fontsize=8, color=color, style="italic", pad=3)
        ax.axis("off")
        return ax

    # (0,0) clean + detections
    ax = ax_img((0, 0), img_bg,
                f"(a) Clean frame  —  {len(D_clean)} detections")
    draw_detections(ax, D_clean, GREEN, conf=[d.score for d in D_clean], lw=1.5, fs=7)

    # (0,1) VAE reconstruction
    ax_img((0, 1), x_rec_np,
           f"(b) VAE reconstruction $D(E(x))$\nPSNR = {compute_psnr(img_bg, x_rec_np):.1f} dB")

    # (0,2) Latent mask Mz (upscaled)
    Mz_up = np.array(Image.fromarray(
        (Mz_np * 255).astype(np.uint8)).resize((W_img, H_img), Image.NEAREST))
    Mz_rgb = (plt.cm.hot(Mz_up / 255.0)[:, :, :3] * 255).astype(np.uint8)
    ax_img((0, 2), Mz_rgb,
           r"(c) Latent mask $\mathcal{M}_z$ (MaxPool stride-8)")

    # (1,0) δ magnitude overlay
    delta_mag = delta_final.squeeze(0).abs().sum(dim=0).cpu().numpy()
    mag_up    = np.array(Image.fromarray(
        ((delta_mag / (delta_mag.max() + 1e-8)) * 255).astype(np.uint8)
    ).resize((W_img, H_img), Image.BILINEAR)).astype(float) / 255.0
    overlay   = (img_bg.astype(float) * 0.45 +
                 plt.cm.hot(mag_up)[:, :, :3] * 255 * 0.55).clip(0, 255).astype(np.uint8)
    ax_img((1, 0), overlay,
           r"(d) Perturbation $|\delta|$ — vehicle footprints only")

    # (1,1) loss curves
    ax   = fig.add_subplot(gs[1, 1])
    iters = list(range(1, len(hist["L_det"]) + 1))
    ax.plot(iters, hist["L_det"], color=RED,   lw=1.6,
            label=r"$\mathcal{L}_{\rm det}$")
    ax.plot(iters, hist["p_max"], color=STEEL, lw=1.6, ls="--",
            label=r"$p_{\max}$")
    ax.axhline(0.05, color=GRAY, lw=0.7, ls=":", label=r"$\gamma$")
    ax.set_title(f"(e) Loss convergence  ({steps} steps)",
                 fontsize=8, color=GRAY, style="italic", pad=3)
    ax.set_xlabel("Iteration", fontsize=7.5)
    ax.legend(fontsize=7, framealpha=0.6)
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False);  ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=7)

    # (1,2) decoded before paste-back
    step6_path = os.path.join(os.path.dirname(out_path), "step06_x_decoded.png")
    if os.path.exists(step6_path):
        ax_img((1, 2), np.array(Image.open(step6_path)),
               r"(f) $D(z_{\rm adv})$ before paste-back")
    else:
        ax = fig.add_subplot(gs[1, 2])
        ax.text(0.5, 0.5, "(f) see step06_x_decoded.png",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=8, color=GRAY)
        ax.axis("off")

    # (2,0)-(2,1) adversarial frame
    ax = fig.add_subplot(gs[2, :2])
    ax.imshow(x_adv_np)
    if D_adv:
        draw_detections(ax, D_adv, RED, conf=[d.score for d in D_adv], lw=1.5, fs=7)
    n_adv     = len(D_adv)
    title_adv = (f"(g) Adversarial frame  —  {n_adv} detections"
                 if n_adv else
                 "(g) Adversarial frame  —  0 detections  ✓  attack successful")
    ax.set_title(title_adv, fontsize=9,
                 color=RED if n_adv else STEEL,
                 fontweight="bold", pad=3)
    # Stats overlay
    stats_txt = f"PSNR={psnr:.1f} dB  ·  SSIM={ssim:.4f}"
    ax.text(5, x_adv_np.shape[0] - 8, stats_txt,
            color="white", fontsize=7.5, fontweight="bold",
            bbox=dict(facecolor=STEEL, alpha=0.80, edgecolor="none",
                      boxstyle="round,pad=0.25"))
    ax.axis("off")

    # (2,2) clean vs adversarial split
    ax = fig.add_subplot(gs[2, 2])
    combined = np.concatenate([img_bg[:, W_img//2:], x_adv_np[:, W_img//2:]], axis=1)
    ax.imshow(combined)
    ax.axvline(0, color="white", lw=1.5)
    ax.text(5, 15, "Clean", color="white", fontsize=7, fontweight="bold",
            bbox=dict(facecolor=GREEN, alpha=0.75, edgecolor="none",
                      boxstyle="round,pad=0.1"))
    ax.text(combined.shape[1] // 2 + 5, 15, "Adv.", color="white", fontsize=7,
            fontweight="bold",
            bbox=dict(facecolor=RED, alpha=0.75, edgecolor="none",
                      boxstyle="round,pad=0.1"))
    ax.set_title("(h) Right-half comparison", fontsize=8,
                 color=GRAY, style="italic", pad=3)
    ax.axis("off")

    fig.suptitle(
        "Figure — Full Latent Attack Pipeline  |  Phase-3 SSIM configuration  |  UA-DETRAC\n"
        r"$x' = M \odot D(E(x) + \mathcal{M}_z \odot \delta) + (1-M) \odot x$"
        f"   ·   {len(D_clean)} clean detections → {len(D_adv)} adversarial",
        fontsize=10, y=0.98, color=GRAY, style="italic")

    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"  ✓  fig_ch3_pipeline_grid.png")


# ─────────────────────────────────────────────────────────────────────────────
# Cross-frame comparison figure
# ─────────────────────────────────────────────────────────────────────────────

def compose_comparison(results: list[dict], out_path: str) -> None:
    """
    Side-by-side comparison figure for two frames:
    one full success (DFR=1.0) and one partial success.
    Each column = one frame. Rows: clean | adversarial | loss curve | stats.
    """
    n = len(results)
    fig, axes = plt.subplots(3, n, figsize=(7 * n, 14), facecolor="white")
    fig.subplots_adjust(hspace=0.28, wspace=0.08,
                        left=0.04, right=0.97, top=0.93, bottom=0.04)

    for col, r in enumerate(results):
        frame_name  = os.path.basename(r["frame_path"])
        dfr         = r["dfr"]
        label       = "Full success" if dfr >= 1.0 else f"Partial success (DFR={dfr:.2f})"
        col_color   = GREEN if dfr >= 1.0 else ORANGE

        # Row 0 — clean frame
        ax = axes[0, col] if n > 1 else axes[0]
        ax.imshow(r["img_bg"])
        draw_detections(ax, r["D_clean"], GREEN,
                        conf=[d.score for d in r["D_clean"]], lw=1.8, fs=8)
        ax.set_title(
            f"{frame_name}\n{len(r['D_clean'])} detections  ·  "
            f"mean conf = {r['mean_conf_clean']:.3f}",
            fontsize=8.5, color=GRAY, style="italic", pad=4)
        ax.axis("off")

        # Row 1 — adversarial frame
        ax = axes[1, col] if n > 1 else axes[1]
        ax.imshow(r["x_adv_np"])
        if r["D_adv"]:
            draw_detections(ax, r["D_adv"], RED,
                            conf=[d.score for d in r["D_adv"]], lw=1.8, fs=8)
        ax.set_title(
            f"{label}\n{len(r['D_adv'])} detections remaining  ·  "
            f"mean conf = {r['mean_conf_adv']:.3f}",
            fontsize=8.5, color=col_color, fontweight="bold", pad=4)
        stats_txt = (f"PSNR={r['psnr']:.1f} dB\n"
                     f"SSIM={r['ssim']:.4f}\n"
                     f"Steps={r['steps']}")
        ax.text(6, r["x_adv_np"].shape[0] - 8, stats_txt,
                color="white", fontsize=7.5,
                bbox=dict(facecolor=STEEL, alpha=0.82, edgecolor="none",
                          boxstyle="round,pad=0.3"))
        ax.axis("off")

        # Row 2 — loss curves
        ax = axes[2, col] if n > 1 else axes[2]
        hist  = r["hist"]
        iters = list(range(1, len(hist["L_det"]) + 1))
        ax.plot(iters, hist["L_det"], color=RED,   lw=1.6,
                label=r"$\mathcal{L}_{\rm det}$")
        ax.plot(iters, hist["p_max"], color=STEEL, lw=1.6, ls="--",
                label=r"$p_{\max}$")
        ax.axhline(0.05, color=GRAY, lw=0.8, ls=":", label=r"$\gamma=0.05$")
        ax.set_xlabel("Iteration", fontsize=8.5)
        ax.set_ylabel("Value", fontsize=8.5)
        ax.legend(fontsize=8, framealpha=0.7)
        ax.grid(True, alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=8)
        ax.set_title(f"Loss convergence  ·  {r['steps']} iterations",
                     fontsize=8.5, color=GRAY, style="italic", pad=4)

    fig.suptitle(
        "Comparison: full-success frame vs. partial-success frame\n"
        "Same attack configuration (Phase-3 SSIM)  —  same hyperparameters  —  same seed",
        fontsize=11, y=0.97, color=GRAY, style="italic")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"  ✓  {os.path.basename(out_path)}")


def run_and_collect(frame_path: str, config_path: str,
                    out_dir: str, seed: int = 42) -> dict:
    """
    Wrapper around run() that also returns a summary dict
    for the cross-frame comparison figure.
    """
    run(frame_path, config_path, out_dir, seed=seed)

    # Re-load the attack_log.json written by run() to get the metrics
    log_path = os.path.join(out_dir, "attack_log.json")
    with open(log_path) as f:
        log = json.load(f)

    # Re-load saved images for the comparison figure
    img_bg   = np.array(Image.open(f"{out_dir}/step00_clean_detections.png"))
    x_adv_np = np.array(Image.open(f"{out_dir}/step07_x_adv_pasteback.png"))

    return {
        "frame_path"     : frame_path,
        "dfr"            : log["detection"]["dfr"],
        "steps"          : log["steps_taken"],
        "psnr"           : log["final_metrics"]["psnr_adv"],
        "ssim"           : log["final_metrics"]["ssim_adv"],
        "mean_conf_clean": log["detection"]["mean_conf_clean"],
        "mean_conf_adv"  : log["detection"]["mean_conf_adv"],
        "hist"           : None,   # curves already saved per-frame
        "img_bg"         : img_bg,
        "x_adv_np"       : x_adv_np,
        "D_clean"        : [],     # boxes not needed here (already drawn)
        "D_adv"          : [],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline visualiser v2 — Chapter 3/4 figures")
    parser.add_argument("--frames", nargs="+",
                        default=["data/images/img00005.jpg",
                                 "data/images/img01380.jpg"],
                        help="One or two frame paths (relative to repo root). "
                             "First = full success, second = partial success.")
    parser.add_argument("--config", default="configs/phase3_ssim.yaml",
                        help="Attack config YAML")
    parser.add_argument("--out",    default="figures/pipeline_steps",
                        help="Root output directory")
    parser.add_argument("--seed",   type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    config_path = os.path.join(ROOT, args.config)
    out_root    = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)

    frame_paths = [
        fp if os.path.isabs(fp) else os.path.join(ROOT, fp)
        for fp in args.frames
    ]

    if len(frame_paths) == 1:
        # Single frame — original behaviour
        run(frame_paths[0], config_path, out_root, seed=args.seed)

    else:
        # Two frames — run each in its own subdirectory, then compare
        from src.attack import AttackConfig
        import importlib

        results = []
        sub_dirs = []
        for fp in frame_paths:
            frame_id = os.path.splitext(os.path.basename(fp))[0]
            sub_dir  = os.path.join(out_root, frame_id)
            print(f"\n{'='*60}")
            print(f"  Processing frame: {frame_id}")
            print(f"{'='*60}")
            run(fp, config_path, sub_dir, seed=args.seed)
            sub_dirs.append(sub_dir)

            # Load metrics from saved log
            log_path = os.path.join(sub_dir, "attack_log.json")
            with open(log_path) as f:
                log = json.load(f)

            # Load step_log.csv to rebuild history for loss curves
            csv_path = os.path.join(sub_dir, "step_log.csv")
            hist = {"L_det": [], "p_max": []}
            with open(csv_path, newline="") as f:
                for row in csv.DictReader(f):
                    hist["L_det"].append(float(row["L_det"]))
                    hist["p_max"].append(float(row["p_max"]))

            results.append({
                "frame_path"     : fp,
                "dfr"            : log["detection"]["dfr"],
                "steps"          : log["steps_taken"],
                "psnr"           : log["final_metrics"]["psnr_adv"],
                "ssim"           : log["final_metrics"]["ssim_adv"],
                "mean_conf_clean": log["detection"]["mean_conf_clean"],
                "mean_conf_adv"  : log["detection"]["mean_conf_adv"],
                "hist"           : hist,
                "img_bg"         : np.array(Image.open(
                    f"{sub_dir}/step00_clean_detections.png")),
                "x_adv_np"       : np.array(Image.open(
                    f"{sub_dir}/step07_x_adv_pasteback.png")),
                "D_clean"        : [],
                "D_adv"          : [],
            })

        # Compose cross-frame comparison figure
        print("\n[Compare] Building cross-frame comparison figure…")
        compose_comparison(
            results,
            out_path=os.path.join(out_root, "fig_frame_comparison.png"),
        )

        print(f"\n✓  All done. Outputs in: {out_root}/")
        for r in results:
            name = os.path.basename(r["frame_path"])
            print(f"   {name:20s}  DFR={r['dfr']:.2f}  "
                  f"PSNR={r['psnr']:.1f} dB  SSIM={r['ssim']:.4f}")
