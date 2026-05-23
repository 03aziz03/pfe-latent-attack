"""
pipeline_visualiser.py
======================
Runs the full Phase-3 SSIM attack on one frame and captures every
intermediate state of the pipeline as a publication-quality figure.

Outputs  (saved to figures/pipeline_steps/):
    step00_clean_detections.png   – clean frame + YOLOv8 boxes + confidence
    step01_pixel_mask.png         – binary pixel-space mask M
    step02_latent_z.png           – 4-channel latent z = E(x) visualised
    step03_latent_mask.png        – latent mask Mz (low-res heatmap)
    step04_delta_early.png        – perturbation delta at step 10 (heatmap)
    step05_delta_final.png        – perturbation delta at final step (heatmap)
    step06_x_decoded.png          – decoded adversarial D(z+Mz*delta) before paste-back
    step07_x_adv_pasteback.png    – final adversarial after paste-back (0 detections)
    step08_loss_curves.png        – L_det / L_perc / p_max vs. iteration
    step09_delta_overlay.png      – |delta| magnitude overlaid on clean frame
    fig_ch3_pipeline_grid.png     – composed 3×3 grid for Chapter 3

Usage on Colab:
    # 1. Mount drive / clone repo, then:
    import subprocess
    subprocess.run(["python", "scripts/pipeline_visualiser.py",
                    "--frame", "data/images/img00005.jpg",
                    "--config", "configs/phase3_ssim.yaml",
                    "--out",   "figures/pipeline_steps"])
"""

from __future__ import annotations
import argparse
import os
import sys
import copy

import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

# ── ensure src/ is on path ───────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.attack import AttackConfig, LatentObjectAttack
from src.detector import YOLOv8Wrapper
from src.vae import SDVAE
from src.masks import boxes_to_pixel_mask, pixel_mask_to_latent_mask

# ── global style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family"     : "serif",
    "figure.dpi"      : 150,
    "savefig.dpi"     : 220,
    "savefig.bbox"    : "tight",
    "savefig.pad_inches": 0.06,
})
GRAY  = "#555555"
STEEL = "#3a5f82"
RED   = "#a02020"
GREEN = "#2d7a2d"


# ─────────────────────────────────────────────────────────────────────────────
# Helper utilities
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
        x1, y1, x2, y2 = d.x1, d.y1, d.x2, d.y2
        ax.add_patch(mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=lw, edgecolor=color, facecolor="none"))
        if conf is not None and i < len(conf):
            ax.text(x1 + 2, y1 - 5,
                    f"{label_prefix}{conf[i]:.2f}",
                    color="white", fontsize=fs, fontweight="bold",
                    bbox=dict(facecolor=color, edgecolor="none",
                              boxstyle="round,pad=0.12", alpha=0.88))


def visualise_latent(z: torch.Tensor, title: str, out_path: str) -> None:
    """Visualise the 4 latent channels as a 2×2 grid of heatmaps."""
    z_np = z.squeeze(0).cpu().numpy()   # (4, H/8, W/8)
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.0), facecolor="white")
    fig.subplots_adjust(wspace=0.08, left=0.02, right=0.98, top=0.82, bottom=0.06)
    cmaps = ["RdBu_r", "RdBu_r", "RdBu_r", "RdBu_r"]
    for ci, ax in enumerate(axes):
        ch = z_np[ci]
        vabs = max(abs(ch.min()), abs(ch.max())) + 1e-6
        im = ax.imshow(ch, cmap=cmaps[ci], vmin=-vabs, vmax=vabs)
        ax.set_title(f"Channel {ci}", fontsize=8, color=GRAY, style="italic")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title, fontsize=9, y=0.97, color=GRAY, style="italic")
    save(fig, out_path)


def visualise_delta(delta: torch.Tensor, img_bg: np.ndarray,
                    title: str, out_path: str, alpha: float = 0.65) -> None:
    """Overlay |delta| magnitude (summed across 4 latent channels) on clean frame."""
    delta_mag = delta.squeeze(0).abs().sum(dim=0).cpu().numpy()  # (H/8, W/8)
    # Upsample to image size
    H, W = img_bg.shape[:2]
    from PIL import Image as _Im
    mag_pil = _Im.fromarray(
        ((delta_mag / (delta_mag.max() + 1e-8)) * 255).astype(np.uint8)
    ).resize((W, H), resample=_Im.BILINEAR)
    mag_up = np.array(mag_pil).astype(float) / 255.0

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), facecolor="white")
    fig.subplots_adjust(wspace=0.05, left=0.02, right=0.98, top=0.82, bottom=0.06)

    # Panel 0 — clean frame
    axes[0].imshow(img_bg)
    axes[0].set_title("Clean frame", fontsize=8, color=GRAY, style="italic")
    axes[0].axis("off")

    # Panel 1 — delta magnitude heatmap (jet colormap)
    axes[1].imshow(img_bg, alpha=0.45)
    axes[1].imshow(mag_up, cmap="hot", alpha=alpha, vmin=0, vmax=1)
    axes[1].set_title(r"$|\delta|$ magnitude overlay", fontsize=8,
                      color=GRAY, style="italic")
    axes[1].axis("off")

    # Panel 2 — heatmap alone
    im = axes[2].imshow(delta_mag, cmap="hot")
    axes[2].set_title(r"$|\delta|$ (latent resolution)", fontsize=8,
                      color=GRAY, style="italic")
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=9, y=0.97, color=GRAY, style="italic")
    save(fig, out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Instrumented attack — subclass that saves intermediate states
# ─────────────────────────────────────────────────────────────────────────────

class InstrumentedAttack(LatentObjectAttack):
    """Extends LatentObjectAttack to capture and save pipeline intermediates."""

    def __init__(self, detector, vae, config, out_dir: str):
        super().__init__(detector, vae, config)
        self.out_dir = out_dir
        # Intermediate states filled during _single_run
        self._snap: dict = {}

    def _single_run(self, x, z, M, Mz, D_clean, C_clean, restart_idx):
        """Overrides parent to inject snapshot hooks."""
        cfg  = self.cfg
        device = self.vae.device

        delta_data = torch.zeros_like(z)
        delta = delta_data.requires_grad_(True)
        optim = torch.optim.Adam([delta], lr=cfg.lr)

        grad_buf = torch.zeros_like(z) if cfg.use_momentum else None

        history = {"L": [], "L_det": [], "L_perc": [], "L_reg": [], "p_max": []}
        steps_taken = 0
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
            L = L_det + cfg.lambda_p * L_perc + cfg.lambda_r * L_reg

            optim.zero_grad(set_to_none=True)
            L.backward()

            if cfg.use_momentum and grad_buf is not None and delta.grad is not None:
                with torch.no_grad():
                    g = delta.grad
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

            # ── snapshot at step ~10 (early) ──────────────────────────────
            if t == 9 and not snap_saved_early:
                self._snap["delta_early"] = delta.detach().clone()
                self._snap["x_dec_early"] = x_dec.detach().clone()
                snap_saved_early = True

            steps_taken = t + 1
            if cfg.early_stop and p_max < cfg.gamma - cfg.early_stop_margin:
                break

        # ── final snapshots ───────────────────────────────────────────────
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
# Main visualisation pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run(frame_path: str, config_path: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # ── load config ───────────────────────────────────────────────────────
    with open(config_path) as f:
        cfg_dict = yaml.safe_load(f)

    device = cfg_dict["runtime"]["device"] if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── load models ───────────────────────────────────────────────────────
    det_cfg = cfg_dict["detector"]
    vae_cfg = cfg_dict["vae"]
    atk_cfg = cfg_dict["attack"]

    print("Loading YOLOv8 detector…")
    detector = YOLOv8Wrapper(
        weights = os.path.join(ROOT, det_cfg["weights"]),
        device  = device,
    )

    print("Loading SD-VAE…")
    vae = SDVAE(model_id=vae_cfg["model_id"], device=device,
                scale=vae_cfg.get("scale", 0.18215))

    attack_config = AttackConfig(**{k: v for k, v in atk_cfg.items()})
    attack = InstrumentedAttack(detector, vae, attack_config, out_dir)

    # ── load and preprocess frame ─────────────────────────────────────────
    print(f"Loading frame: {frame_path}")
    img_pil  = Image.open(frame_path).convert("RGB")
    img_np   = np.array(img_pil)                    # (H, W, 3) uint8
    H, W     = img_np.shape[:2]
    # Pad to multiple of 8
    H8 = ((H + 7) // 8) * 8
    W8 = ((W + 7) // 8) * 8
    img_pil_padded = Image.fromarray(img_np).resize((W8, H8), Image.BILINEAR)
    x = torch.from_numpy(np.array(img_pil_padded)).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    x = x.to(device)

    # ── STEP 0: clean detections ──────────────────────────────────────────
    print("\n[Step 0] Clean detections…")
    D_clean = detector.detect_nms(x, conf_thr=det_cfg["conf_thr"],
                                   iou_thr=det_cfg["iou_nms"])
    print(f"  {len(D_clean)} detections")

    fig, ax = plt.subplots(1, 1, figsize=(8, 5), facecolor="white")
    ax.imshow(tensor_to_np(x))
    scores = [d.score for d in D_clean]
    draw_detections(ax, D_clean, color=GREEN, label_prefix="car ", conf=scores, lw=2.0)
    ax.set_title(
        f"Step 0 — Clean frame  |  YOLOv8 detections: {len(D_clean)} vehicles",
        fontsize=9, color=GRAY, style="italic")
    ax.axis("off")
    fig.text(0.5, 0.01,
             f"conf_thr={det_cfg['conf_thr']}  ·  iou_nms={det_cfg['iou_nms']}  ·  "
             f"frame: {os.path.basename(frame_path)}",
             ha="center", fontsize=7.5, color=GRAY, style="italic")
    save(fig, f"{out_dir}/step00_clean_detections.png")

    # ── STEP 1: pixel-space mask M ────────────────────────────────────────
    print("[Step 1] Pixel-space mask M…")
    M  = boxes_to_pixel_mask(D_clean, H=H8, W=W8, device=device)
    M_np = M.squeeze().cpu().numpy()   # (H, W) binary

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor="white")
    axes[0].imshow(tensor_to_np(x))
    draw_detections(axes[0], D_clean, color=GREEN, lw=1.5)
    axes[0].set_title("Clean frame with boxes", fontsize=8, color=GRAY, style="italic")
    axes[0].axis("off")

    mask_vis = np.stack([M_np * 255] * 3, axis=-1).astype(np.uint8)
    axes[1].imshow(mask_vis, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title(r"Pixel-space mask $M$  (white = perturb zone)",
                      fontsize=8, color=GRAY, style="italic")
    axes[1].axis("off")
    fig.suptitle(
        r"Step 1 — Binary mask $M \in \{0,1\}^{H \times W}$"
        "  |  Union of detected bounding boxes",
        fontsize=9, y=0.97, color=GRAY, style="italic")
    save(fig, f"{out_dir}/step01_pixel_mask.png")

    # ── STEP 2: run attack (captures all intermediates) ───────────────────
    print("[Step 2] Running attack (captures all intermediate states)…")
    result = attack.attack(x)
    snap   = attack._snap
    steps  = snap["steps_taken"]
    print(f"  Attack complete in {steps} steps")

    # ── STEP 3: latent z = E(x) ───────────────────────────────────────────
    print("[Step 3] Visualising latent z = E(x)…")
    visualise_latent(
        snap["z"],
        title=r"Step 2 — Latent encoding $z = E(x) \in \mathbb{R}^{4 \times H/8 \times W/8}$"
              "  |  4 channels of the frozen SD-VAE encoder",
        out_path=f"{out_dir}/step02_latent_z.png",
    )

    # ── STEP 4: latent mask Mz ────────────────────────────────────────────
    print("[Step 4] Visualising latent mask Mz…")
    Mz_np = snap["Mz"].squeeze(0)[0].cpu().numpy()   # single channel (H/8, W/8)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), facecolor="white")
    fig.subplots_adjust(wspace=0.06, left=0.02, right=0.98, top=0.82, bottom=0.06)

    axes[0].imshow(tensor_to_np(x))
    draw_detections(axes[0], D_clean, color=GREEN, lw=1.5)
    axes[0].set_title("Clean frame (pixel space)", fontsize=8, color=GRAY, style="italic")
    axes[0].axis("off")

    axes[1].imshow(M_np, cmap="gray")
    axes[1].set_title(r"$M$  (pixel, $H \times W$)", fontsize=8, color=GRAY, style="italic")
    axes[1].axis("off")

    axes[2].imshow(Mz_np, cmap="hot", interpolation="nearest")
    axes[2].set_title(
        r"$\mathcal{M}_z$  (latent, $H/8 \times W/8$)"
        "\nMaxPool stride-8  →  conservative coverage",
        fontsize=8, color=GRAY, style="italic")
    axes[2].axis("off")

    fig.suptitle(
        r"Step 3 — Latent mask $\mathcal{M}_z = \mathrm{MaxPool}_8(M)$"
        "  |  One latent cell covers an 8×8 pixel block",
        fontsize=9, y=0.97, color=GRAY, style="italic")
    save(fig, f"{out_dir}/step03_latent_mask.png")

    # ── STEP 5: delta at step 10 (early) ─────────────────────────────────
    print("[Step 5] Delta at step 10 (early)…")
    img_bg = tensor_to_np(x)
    visualise_delta(
        snap["delta_early"], img_bg,
        title=r"Step 4 — Perturbation $\delta$ at iteration 10  (early accumulation)",
        out_path=f"{out_dir}/step04_delta_early.png",
    )

    # ── STEP 6: delta final ───────────────────────────────────────────────
    print("[Step 6] Delta final…")
    visualise_delta(
        snap["delta_final"], img_bg,
        title=(r"Step 5 — Perturbation $\delta$ at final iteration "
               f"(step {steps})  |  "
               r"$\|\delta\|_\infty \leq \varepsilon_z = 0.50$"),
        out_path=f"{out_dir}/step05_delta_final.png",
    )

    # ── STEP 7: decoded frame before paste-back ───────────────────────────
    print("[Step 7] Decoded adversarial (before paste-back)…")
    x_dec_np = tensor_to_np(snap["x_dec_final"].clamp(0, 1))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor="white")
    axes[0].imshow(img_bg)
    axes[0].set_title("Clean frame $x$", fontsize=8, color=GRAY, style="italic")
    axes[0].axis("off")
    axes[1].imshow(x_dec_np)
    axes[1].set_title(
        r"$D(z + \mathcal{M}_z \odot \delta)$ — decoded adversarial"
        "\n(background not yet restored — artefacts visible outside boxes)",
        fontsize=8, color=GRAY, style="italic")
    axes[1].axis("off")
    fig.suptitle(
        r"Step 6 — Decoded adversarial frame $\hat{x}_{\mathrm{adv}} = D(z + \mathcal{M}_z \odot \delta)$"
        "  |  Before paste-back",
        fontsize=9, y=0.97, color=GRAY, style="italic")
    save(fig, f"{out_dir}/step06_x_decoded.png")

    # ── STEP 8: final adversarial after paste-back + detections ──────────
    print("[Step 8] Final adversarial (after paste-back)…")
    x_adv_np = tensor_to_np(snap["x_adv_final"])
    D_adv = detector.detect_nms(snap["x_adv_final"], conf_thr=det_cfg["conf_thr"],
                                  iou_thr=det_cfg["iou_nms"])
    print(f"  Adversarial detections: {len(D_adv)}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), facecolor="white")
    fig.subplots_adjust(wspace=0.05, left=0.02, right=0.98, top=0.82, bottom=0.06)

    axes[0].imshow(img_bg)
    draw_detections(axes[0], D_clean, color=GREEN, conf=[d.score for d in D_clean], lw=1.8)
    axes[0].set_title(
        f"Clean frame  —  {len(D_clean)} detections",
        fontsize=9, color=GREEN, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(x_adv_np)
    if D_adv:
        draw_detections(axes[1], D_adv, color=RED,
                        conf=[d.score for d in D_adv], lw=1.8)
        det_label = f"{len(D_adv)} detections remaining"
        col = RED
    else:
        det_label = "0 detections — attack successful"
        col = STEEL
    axes[1].set_title(f"Adversarial frame  —  {det_label}",
                      fontsize=9, color=col, fontweight="bold")
    axes[1].axis("off")

    fig.suptitle(
        r"Step 7 — Final adversarial $x' = M \odot \hat{x}_{\mathrm{adv}} + (1-M) \odot x$"
        f"  |  Phase-3 SSIM config  ·  {steps} steps",
        fontsize=9, y=0.97, color=GRAY, style="italic")
    save(fig, f"{out_dir}/step07_x_adv_pasteback.png")

    # ── STEP 9: loss curves ───────────────────────────────────────────────
    print("[Step 9] Loss curves…")
    hist = snap["history"]
    iters = list(range(1, len(hist["L_det"]) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), facecolor="white")
    fig.subplots_adjust(wspace=0.30, left=0.08, right=0.97, top=0.82, bottom=0.12)

    # Left: detection loss + p_max
    ax = axes[0]
    ax.plot(iters, hist["L_det"], color=RED,   lw=1.8, label=r"$\mathcal{L}_{\mathrm{det}}$")
    ax.plot(iters, hist["p_max"], color=STEEL, lw=1.8, linestyle="--",
            label=r"$p_{\max}$ (peak confidence)")
    ax.axhline(0.05, color=GRAY, lw=0.8, linestyle=":", label=r"$\gamma = 0.05$ threshold")
    ax.set_xlabel("Iteration", fontsize=9)
    ax.set_ylabel("Value", fontsize=9)
    ax.set_title(r"Detection loss  $\mathcal{L}_{\mathrm{det}}$  &  peak confidence",
                 fontsize=9, color=GRAY, style="italic")
    ax.legend(fontsize=8, framealpha=0.7)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Right: perceptual + regularisation
    ax = axes[1]
    ax.plot(iters, hist["L_perc"], color="#c0622a", lw=1.8,
            label=r"$\mathcal{L}_{\mathrm{perc}}$ (LPIPS+SSIM)")
    ax.plot(iters, hist["L_reg"],  color="#7a4a9e", lw=1.8, linestyle="--",
            label=r"$\mathcal{L}_{\mathrm{reg}}$")
    ax.set_xlabel("Iteration", fontsize=9)
    ax.set_ylabel("Value", fontsize=9)
    ax.set_title(r"Perceptual  $\mathcal{L}_{\mathrm{perc}}$  &  regularisation  $\mathcal{L}_{\mathrm{reg}}$",
                 fontsize=9, color=GRAY, style="italic")
    ax.legend(fontsize=8, framealpha=0.7)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle(
        f"Step 8 — Loss curves  |  Phase-3 SSIM  ·  {steps} iterations  ·  "
        f"early-stop at $p_{{\\max}} < \\gamma$",
        fontsize=9, y=0.97, color=GRAY, style="italic")
    save(fig, f"{out_dir}/step08_loss_curves.png")

    # ── STEP 10: delta overlay (publication-quality) ──────────────────────
    print("[Step 10] Delta magnitude overlay…")
    visualise_delta(
        snap["delta_final"], img_bg,
        title=(r"Step 9 — Perturbation magnitude $|\delta|$ overlaid on clean frame"
               "\nHot regions = high latent perturbation = vehicle footprints only"),
        out_path=f"{out_dir}/step09_delta_overlay.png",
        alpha=0.72,
    )

    # ── COMPOSED GRID: Chapter 3 figure ───────────────────────────────────
    print("[Compose] Building Chapter 3 pipeline grid…")
    _compose_grid(
        img_bg      = img_bg,
        x_adv_np    = x_adv_np,
        M_np        = M_np,
        Mz_np       = Mz_np,
        delta_final = snap["delta_final"],
        D_clean     = D_clean,
        D_adv       = D_adv,
        hist        = hist,
        steps       = steps,
        out_path    = f"{out_dir}/fig_ch3_pipeline_grid.png",
    )

    print(f"\n✓  All figures saved to: {out_dir}")
    print(f"   Clean detections  : {len(D_clean)}")
    print(f"   Adversarial detections: {len(D_adv)}")
    print(f"   Steps taken       : {steps}")


def _compose_grid(img_bg, x_adv_np, M_np, Mz_np, delta_final,
                  D_clean, D_adv, hist, steps, out_path):
    """3×3 publication grid for Chapter 3."""
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(15, 10), facecolor="white")
    gs  = GridSpec(3, 3, figure=fig,
                   hspace=0.35, wspace=0.12,
                   left=0.04, right=0.97, top=0.93, bottom=0.06)

    GRAY = "#555555"; GREEN = "#2d7a2d"; RED = "#a02020"; STEEL = "#3a5f82"

    def ax_img(pos, img, title, border_color=None):
        ax = fig.add_subplot(gs[pos])
        ax.imshow(img)
        ax.set_title(title, fontsize=8, color=GRAY, style="italic", pad=3)
        ax.axis("off")
        if border_color:
            for sp in ax.spines.values():
                sp.set_visible(True); sp.set_color(border_color); sp.set_linewidth(2)
        return ax

    # (0,0) Clean frame + detections
    ax = ax_img((0, 0), img_bg,
                f"(a) Clean frame  —  {len(D_clean)} detections")
    draw_detections(ax, D_clean, GREEN, conf=[d.score for d in D_clean], lw=1.5, fs=7)

    # (0,1) Pixel mask M
    ax_img((0, 1),
           np.stack([M_np * 200] * 3, axis=-1).astype(np.uint8),
           r"(b) Pixel mask $M$  (object footprints)")

    # (0,2) Latent mask Mz
    H_img, W_img = img_bg.shape[:2]
    Mz_up = np.array(Image.fromarray(
        (Mz_np * 255).astype(np.uint8)).resize((W_img, H_img), Image.NEAREST))
    Mz_rgb = plt.cm.hot(Mz_up / 255.0)[:, :, :3]
    ax_img((0, 2),
           (Mz_rgb * 255).astype(np.uint8),
           r"(c) Latent mask $\mathcal{M}_z$  (MaxPool stride-8)")

    # (1,0) Delta magnitude overlay
    delta_mag = delta_final.squeeze(0).abs().sum(dim=0).cpu().numpy()
    mag_up = np.array(Image.fromarray(
        ((delta_mag / (delta_mag.max() + 1e-8)) * 255).astype(np.uint8)
    ).resize((W_img, H_img), Image.BILINEAR)).astype(float) / 255.0
    overlay = (img_bg.astype(float) * 0.45 +
               plt.cm.hot(mag_up)[:, :, :3] * 255 * 0.55).clip(0, 255).astype(np.uint8)
    ax_img((1, 0), overlay,
           r"(d) Perturbation $|\delta|$ — vehicle footprints only")

    # (1,1) Loss curves (L_det + p_max)
    ax = fig.add_subplot(gs[1, 1])
    iters = list(range(1, len(hist["L_det"]) + 1))
    ax.plot(iters, hist["L_det"], color=RED,   lw=1.6, label=r"$\mathcal{L}_{\mathrm{det}}$")
    ax.plot(iters, hist["p_max"], color=STEEL, lw=1.6, ls="--", label=r"$p_{\max}$")
    ax.axhline(0.05, color=GRAY, lw=0.7, ls=":", label=r"$\gamma$")
    ax.set_title(f"(e) Loss convergence  ({steps} steps)", fontsize=8, color=GRAY,
                 style="italic", pad=3)
    ax.set_xlabel("Iteration", fontsize=7.5)
    ax.legend(fontsize=7, framealpha=0.6)
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=7)

    # (1,2) Decoded before paste-back (placeholder — load if exists)
    step6_path = os.path.join(os.path.dirname(out_path), "step06_x_decoded.png")
    if os.path.exists(step6_path):
        img6 = np.array(Image.open(step6_path))
        ax_img((1, 2), img6, r"(f) $D(z_{\mathrm{adv}})$ before paste-back")
    else:
        ax = fig.add_subplot(gs[1, 2])
        ax.text(0.5, 0.5, "(f) see step06_x_decoded.png",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=8, color=GRAY)
        ax.axis("off")

    # (2,0)-(2,1) Adversarial frame (spans 2 columns)
    ax = fig.add_subplot(gs[2, :2])
    ax.imshow(x_adv_np)
    if D_adv:
        draw_detections(ax, D_adv, RED, conf=[d.score for d in D_adv], lw=1.5, fs=7)
    n_adv = len(D_adv)
    title_adv = (f"(g) Adversarial frame  —  {n_adv} detections"
                 if n_adv else
                 "(g) Adversarial frame  —  0 detections  ✓  attack successful")
    ax.set_title(title_adv, fontsize=9, color=RED if n_adv else STEEL,
                 fontweight="bold", pad=3)
    ax.axis("off")

    # (2,2) Side-by-side mini comparison
    ax = fig.add_subplot(gs[2, 2])
    combined = np.concatenate([img_bg[:, W_img//2:], x_adv_np[:, W_img//2:]], axis=1)
    ax.imshow(combined)
    ax.axvline(0, color="white", lw=1.5)
    ax.text(5, 15, "Clean", color="white", fontsize=7, fontweight="bold",
            bbox=dict(facecolor=GREEN, alpha=0.75, edgecolor="none", boxstyle="round,pad=0.1"))
    ax.text(combined.shape[1]//2 + 5, 15, "Adversarial", color="white", fontsize=7,
            fontweight="bold",
            bbox=dict(facecolor=RED, alpha=0.75, edgecolor="none", boxstyle="round,pad=0.1"))
    ax.set_title("(h) Clean vs. adversarial (right half)", fontsize=8,
                 color=GRAY, style="italic", pad=3)
    ax.axis("off")

    fig.suptitle(
        "Figure — Full Attack Pipeline  |  Phase-3 SSIM configuration  |  UA-DETRAC\n"
        r"$x' = M \odot D(E(x) + \mathcal{M}_z \odot \delta) + (1-M) \odot x$"
        f"   ·   {len(D_clean)} clean detections → {len(D_adv)} adversarial detections",
        fontsize=10, y=0.97, color=GRAY, style="italic")

    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"  ✓  fig_ch3_pipeline_grid.png")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline visualiser for Chapter 3")
    parser.add_argument("--frame",  default="data/images/img00005.jpg",
                        help="Path to input frame (relative to repo root)")
    parser.add_argument("--config", default="configs/phase3_ssim.yaml",
                        help="Attack config YAML")
    parser.add_argument("--out",    default="figures/pipeline_steps",
                        help="Output directory for step images")
    args = parser.parse_args()

    frame_path  = os.path.join(ROOT, args.frame)
    config_path = os.path.join(ROOT, args.config)
    out_dir     = os.path.join(ROOT, args.out)

    run(frame_path, config_path, out_dir)
