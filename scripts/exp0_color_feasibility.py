"""
Expérience 0 — Color Manipulation Feasibility
==============================================

Objectif : valider que l'optimisation latente peut changer la couleur
d'un véhicule sélectionné manuellement en rouge (ou autre couleur cible).

Pipeline (identique à l'attaque de suppression, sauf la loss) :
    Image → YOLO (masque) → VAE encode → optimise δ → VAE decode → paste-back

Usage (Colab) :
    # Place tes images sélectionnées dans un dossier, ex: data/exp0_selected/
    !python scripts/exp0_color_feasibility.py \
        --images_dir   data/exp0_selected \
        --weights      runs/yolov8n_detrac/best.pt \
        --output_dir   results/exp0_color \
        --target_color red \
        --num_steps    100 \
        --eps_z        1.0

    # Ou pointer vers des fichiers individuels :
    !python scripts/exp0_color_feasibility.py \
        --images img00001.jpg img00005.jpg img00012.jpg \
        --weights runs/yolov8n_detrac/best.pt \
        --output_dir results/exp0_color \
        --target_color red
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ── add repo root to path ────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.vae      import SDVAE
from src.detector import YOLOv8Wrapper
from src.masks    import boxes_to_pixel_mask, pixel_mask_to_latent_mask
from src.utils    import load_image as load_image_tensor


# ════════════════════════════════════════════════════════════════════════════
# 1.  LAB COLOR UTILITIES
# ════════════════════════════════════════════════════════════════════════════

def rgb_to_lab(rgb: torch.Tensor) -> torch.Tensor:
    """Convert (B, 3, H, W) RGB in [0,1] → CIE LAB tensor, same shape.

    Uses the standard D65 illuminant / 2° observer.
    Fully differentiable — gradients flow through the decoder into δ.
    No external dependency required.
    """
    # ── 1. sRGB linearisation ────────────────────────────────────────────
    mask_lo = rgb <= 0.04045
    linear  = torch.where(mask_lo,
                          rgb / 12.92,
                          ((rgb + 0.055) / 1.055) ** 2.4)

    # ── 2. linear RGB → XYZ (D65) ───────────────────────────────────────
    R, G, B = linear[:, 0:1], linear[:, 1:2], linear[:, 2:3]
    X = 0.4124564 * R + 0.3575761 * G + 0.1804375 * B
    Y = 0.2126729 * R + 0.7151522 * G + 0.0721750 * B
    Z = 0.0193339 * R + 0.1191920 * G + 0.9503041 * B

    # ── 3. normalise by D65 white point ─────────────────────────────────
    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
    fx = _f_lab(X / Xn)
    fy = _f_lab(Y / Yn)
    fz = _f_lab(Z / Zn)

    # ── 4. LAB channels ─────────────────────────────────────────────────
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return torch.cat([L, a, b], dim=1)          # (B, 3, H, W)


def _f_lab(t: torch.Tensor) -> torch.Tensor:
    """CIE f() function used in XYZ→LAB (differentiable)."""
    delta = 6.0 / 29.0
    return torch.where(t > delta ** 3,
                       t.clamp(min=1e-8).pow(1.0 / 3.0),
                       t / (3.0 * delta ** 2) + 4.0 / 29.0)


# Target colors in LAB (a, b) — L channel is NOT targeted (preserves luminance)
TARGET_COLORS_AB = {
    "red":    torch.tensor([[ 45.0,  30.0]]),
    "blue":   torch.tensor([[-15.0, -50.0]]),
    "yellow": torch.tensor([[ -5.0,  75.0]]),
    "green":  torch.tensor([[-40.0,  30.0]]),
}

TARGET_COLORS_RGB_DISPLAY = {
    "red":    (220,  50,  50),
    "blue":   ( 50,  80, 200),
    "yellow": (220, 200,   0),
    "green":  ( 30, 140,  60),
}


def paint_mask(x: torch.Tensor,
               M_bbox: torch.Tensor,
               L_min: float = 25.0,
               L_max: float = 75.0) -> torch.Tensor:
    """Refine a bounding-box mask to keep only 'car paint' pixels.

    Excludes:
      - Very dark pixels  (L < L_min): tires, deep shadows
      - Very bright pixels (L > L_max): windows, headlights, specular reflections

    Args:
        x:      (1,3,H,W) image tensor in [0,1]
        M_bbox: (1,1,H,W) full bounding-box mask
        L_min:  luminance lower threshold (CIE LAB scale 0–100)
        L_max:  luminance upper threshold

    Returns:
        (1,1,H,W) float mask — intersection of bbox and paint-luminance range
    """
    with torch.no_grad():
        L = rgb_to_lab(x)[:, 0:1]          # luminance channel
    paint = ((L > L_min) & (L < L_max)).float()
    return M_bbox * paint


def color_loss_lab(x_adv: torch.Tensor,
                   M: torch.Tensor,
                   ab_target: torch.Tensor) -> torch.Tensor:
    """LAB color loss — pushes the (a,b) channels of x_adv inside M toward ab_target.

    Only targets chrominance (a, b). The L channel (luminance/structure)
    is constrained separately by L_struct.
    """
    lab_adv = rgb_to_lab(x_adv)
    ab_adv  = lab_adv[:, 1:]                              # (1, 2, H, W)
    tgt     = ab_target.view(1, 2, 1, 1).to(x_adv.device).expand_as(ab_adv)
    diff    = (ab_adv - tgt) * M
    n_pix   = M.sum() * 2 + 1e-8
    return diff.pow(2).sum() / n_pix


def structure_loss_lab(x_adv: torch.Tensor,
                       x_orig: torch.Tensor,
                       M: torch.Tensor) -> torch.Tensor:
    """Preserve L (luminance) channel inside mask.

    Keeps the car's shape, highlights, and shadows intact while only
    the color (a, b channels) changes.
    """
    L_adv  = rgb_to_lab(x_adv)[:, 0:1]
    L_orig = rgb_to_lab(x_orig)[:, 0:1]
    diff   = (L_adv - L_orig) * M
    n_pix  = M.sum() + 1e-8
    return diff.pow(2).sum() / n_pix


# ════════════════════════════════════════════════════════════════════════════
# 2.  COLOR ATTACK (single image)
# ════════════════════════════════════════════════════════════════════════════

def run_color_attack(
        x:         torch.Tensor,
        z:         torch.Tensor,
        M:         torch.Tensor,
        Mz:        torch.Tensor,
        vae:       SDVAE,
        ab_target: torch.Tensor,
        eps_z:     float = 1.0,
        lr:        float = 0.02,
        num_steps: int   = 100,
        lambda_s:  float = 2.0,
        lambda_bg: float = 10.0,
        lambda_r:  float = 1e-3,
        L_min:     float = 25.0,
        L_max:     float = 75.0,
        verbose:   bool  = True,
) -> tuple[torch.Tensor, dict]:
    """Gradient descent in latent space to change vehicle color.

    Loss:
        L = L_color(M_paint) + lambda_s*L_struct(M_paint)
          + lambda_bg*L_bg(M_bbox) + lambda_r*L_reg

    The key improvement over a naive approach:
      - M_paint = paint_mask(x, M, L_min, L_max)
        → only 'car body' pixels (excludes tires, windows, lights)
        → color and structure losses applied to paint pixels only
      - M (full bbox) used only for paste-back and background loss
        → background outside bbox stays unchanged

    Args:
        x:          (1,3,H,W) original image tensor
        z:          encoded latent (cached, no grad)
        M:          (1,1,H,W) full bounding-box pixel mask
        Mz:         (1,4,H/8,W/8) latent mask
        ab_target:  (1,2) target LAB (a,b) values
        eps_z:      L-inf budget on delta (use 1.0)
        lambda_s:   luminance preservation weight
        lambda_bg:  background preservation weight
        lambda_r:   latent regularization weight
        L_min:      luminance lower bound for paint pixels (exclude tires/shadows)
        L_max:      luminance upper bound for paint pixels (exclude windows/lights)

    Returns:
        x_adv:   (1,3,H,W) color-modified image
        history: dict of per-step loss values
    """
    device    = vae.device
    ab_target = ab_target.to(device)

    # ── Paint mask: only car-body pixels (no tires, no windows, no lights) ──
    M_paint = paint_mask(x, M, L_min=L_min, L_max=L_max)
    paint_ratio = float(M_paint.sum() / (M.sum() + 1e-8))
    if verbose:
        print(f"  [paint_mask] paint pixels = {paint_ratio*100:.1f}% of bbox "
              f"(L_min={L_min}, L_max={L_max})")
    if paint_ratio < 0.05:
        print("  [WARNING] paint mask is nearly empty — "
              "try lowering L_min or raising L_max")

    delta = torch.zeros_like(z, requires_grad=True)
    optim = torch.optim.Adam([delta], lr=lr)

    history = {"L": [], "L_color": [], "L_struct": [], "L_bg": [], "L_reg": []}

    if verbose:
        print(f"  [attack] {num_steps} steps | eps_z={eps_z} | lr={lr} | "
              f"λ_s={lambda_s} | λ_bg={lambda_bg}")

    for step in range(num_steps):
        z_adv  = z + Mz * delta
        x_dec  = vae.decode(z_adv)
        x_adv  = M * x_dec + (1 - M) * x                  # paste-back (full bbox)

        # Color + structure losses on PAINT pixels only (not windows/tires)
        L_color  = color_loss_lab(x_adv, M_paint, ab_target)
        L_struct = structure_loss_lab(x_adv, x, M_paint)
        # Background loss on full bbox complement
        L_bg     = F.mse_loss(x_adv * (1 - M), x * (1 - M))
        L_reg    = delta.pow(2).mean()

        L = (L_color
             + lambda_s  * L_struct
             + lambda_bg * L_bg
             + lambda_r  * L_reg)

        optim.zero_grad(set_to_none=True)
        L.backward()
        optim.step()

        with torch.no_grad():
            delta.data.clamp_(-eps_z, eps_z)
            delta.data.mul_(Mz)

        history["L"].append(float(L.item()))
        history["L_color"].append(float(L_color.item()))
        history["L_struct"].append(float(L_struct.item()))
        history["L_bg"].append(float(L_bg.item()))
        history["L_reg"].append(float(L_reg.item()))

        if verbose and (step == 0 or (step + 1) % 25 == 0):
            print(f"    step {step+1:3d}/{num_steps} | "
                  f"L={L.item():.4f}  "
                  f"L_color={L_color.item():.4f}  "
                  f"L_struct={L_struct.item():.4f}")

    with torch.no_grad():
        z_adv = z + Mz * delta
        x_adv = (M * vae.decode(z_adv) + (1 - M) * x).clamp(0, 1)

    return x_adv.detach(), history


# ════════════════════════════════════════════════════════════════════════════
# 3.  EVALUATION
# ════════════════════════════════════════════════════════════════════════════

def mean_ab_in_mask(x: torch.Tensor,
                    M: torch.Tensor) -> tuple[float, float]:
    with torch.no_grad():
        lab = rgb_to_lab(x)
        a   = float((lab[:, 1] * M[:, 0]).sum() / (M.sum() + 1e-8))
        b   = float((lab[:, 2] * M[:, 0]).sum() / (M.sum() + 1e-8))
    return a, b


def ab_distance(x_adv: torch.Tensor,
                M: torch.Tensor,
                ab_target: torch.Tensor) -> float:
    """Mean Euclidean distance in (a,b) space between output color and target.

    Guide :
        < 10  : très proche de la cible (succès)
        10-25 : changement visible mais incomplet
        > 25  : cible non atteinte
    """
    with torch.no_grad():
        lab_adv = rgb_to_lab(x_adv)
        ab_adv  = lab_adv[:, 1:]
        tgt     = ab_target.view(1, 2, 1, 1).to(x_adv.device).expand_as(ab_adv)
        diff    = (ab_adv - tgt) * M
        n_pix   = M.sum() * 2 + 1e-8
        return float((diff.pow(2).sum() / n_pix).sqrt().item())


# ════════════════════════════════════════════════════════════════════════════
# 4.  VISUALIZATION
# ════════════════════════════════════════════════════════════════════════════

def tensor_to_pil(x: torch.Tensor) -> Image.Image:
    arr = (x.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def save_comparison(
        x_orig:       torch.Tensor,
        x_adv:        torch.Tensor,
        M:            torch.Tensor,
        img_name:     str,
        target_color: str,
        ab_orig:      tuple[float, float],
        ab_adv:       tuple[float, float],
        dist:         float,
        output_path:  Path,
) -> None:
    """Save a 3-panel image: original | mask overlay | modified."""
    orig_pil = tensor_to_pil(x_orig)
    adv_pil  = tensor_to_pil(x_adv)

    # colored mask overlay
    mask_np   = (M.squeeze().cpu().numpy() * 255).astype(np.uint8)
    mask_pil  = Image.fromarray(mask_np).convert("L")
    color_rgb = TARGET_COLORS_RGB_DISPLAY.get(target_color, (200, 50, 50))
    overlay   = Image.new("RGBA", orig_pil.size, color_rgb + (90,))
    mask_rgba = Image.new("RGBA", orig_pil.size, (0, 0, 0, 0))
    mask_rgba.paste(overlay, mask=mask_pil)
    masked_pil = Image.alpha_composite(orig_pil.convert("RGBA"), mask_rgba).convert("RGB")

    W, H   = orig_pil.size
    canvas = Image.new("RGB", (W * 3 + 20, H + 55), (240, 240, 240))
    canvas.paste(orig_pil,   (0,          30))
    canvas.paste(masked_pil, (W + 10,     30))
    canvas.paste(adv_pil,    (W * 2 + 20, 30))

    draw = ImageDraw.Draw(canvas)
    fnt  = ImageFont.load_default()
    a0, b0 = ab_orig
    a1, b1 = ab_adv
    draw.text((4,   6), f"Original  a={a0:+.1f} b={b0:+.1f}", fill=(20,20,20), font=fnt)
    draw.text((W+14, 6), f"Mask ({target_color} target)", fill=(20,20,20), font=fnt)
    draw.text((W*2+24, 6),
              f"Modified  a={a1:+.1f} b={b1:+.1f}  dist={dist:.1f}",
              fill=(20,20,20), font=fnt)

    # verdict bar at bottom
    if dist < 10:
        bar_color, verdict = (60, 180, 60),  "SUCCESS — color reached"
    elif dist < 25:
        bar_color, verdict = (200, 140, 0),  "PARTIAL — color shifted"
    else:
        bar_color, verdict = (200, 50, 50),  "FAIL — color unchanged"

    draw.rectangle([(0, H+30), (W*3+20, H+55)], fill=bar_color)
    draw.text((8, H+35), f"{img_name}  →  {verdict}  (ab_dist={dist:.1f})",
              fill=(255,255,255), font=fnt)

    canvas.save(str(output_path))
    print(f"  Saved → {output_path.name}  [{verdict}]")


def save_loss_curve(history: dict, output_path: Path) -> None:
    lines = [f"step,L,L_color,L_struct,L_bg,L_reg"]
    for i, (L, Lc, Ls, Lb, Lr) in enumerate(zip(
            history["L"], history["L_color"],
            history["L_struct"], history["L_bg"], history["L_reg"])):
        lines.append(f"{i+1},{L:.6f},{Lc:.6f},{Ls:.6f},{Lb:.6f},{Lr:.6f}")
    with open(str(output_path), "w") as f:
        f.write("\n".join(lines))


# ════════════════════════════════════════════════════════════════════════════
# 5.  SUMMARY
# ════════════════════════════════════════════════════════════════════════════

def print_summary(results: list[dict], target_color: str, args) -> None:
    print("\n" + "═" * 68)
    print("  EXPÉRIENCE 0 — RÉSUMÉ")
    print("═" * 68)
    print(f"  Target color : {target_color}")
    print(f"  eps_z        : {args.eps_z}  |  steps : {args.num_steps}  |  lr : {args.lr}")
    print(f"  lambda_s     : {args.lambda_s}  |  lambda_bg : {args.lambda_bg}")
    print()
    print(f"  {'Image':15s}  {'dist':>6s}  {'a_orig':>7s}  {'b_orig':>7s}  "
          f"{'a_adv':>7s}  {'b_adv':>7s}  {'Verdict'}")
    print(f"  {'-'*15}  {'-'*6}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*20}")
    for r in results:
        d = r["dist"]
        verdict = "✓ SUCCESS" if d < 10 else ("~ PARTIAL" if d < 25 else "✗ FAIL")
        print(f"  {r['name']:15s}  {d:6.2f}  {r['a_orig']:+7.1f}  {r['b_orig']:+7.1f}  "
              f"{r['a_adv']:+7.1f}  {r['b_adv']:+7.1f}  {verdict}")

    dists = [r["dist"] for r in results]
    mean_d = sum(dists) / len(dists)
    print()
    print(f"  Mean ab distance : {mean_d:.2f}")
    print()

    n_ok = sum(d < 25 for d in dists)
    if n_ok >= len(dists) * 0.6:
        print("  VERDICT GLOBAL : ✓ OUI — le VAE produit un changement de couleur.")
        print("  → Passer à Phase A complète avec L_struct et métriques CIEDE2000.")
    elif mean_d < 35:
        print("  VERDICT GLOBAL : ~ PARTIEL — changement insuffisant.")
        print("  → Essayer : augmenter eps_z à 1.5 ou 2.0 / réduire lambda_s à 1.0.")
    else:
        print("  VERDICT GLOBAL : ✗ NON — le changement de couleur n'est pas atteint.")
        print("  → Problème probable : eps_z trop faible.")
        print("  → Action : eps_z=2.0, lambda_bg=5.0, lambda_s=1.0")
    print("═" * 68 + "\n")


# ════════════════════════════════════════════════════════════════════════════
# 6.  MAIN
# ════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Exp0 — Color feasibility (manual image selection)")
    # ── image input (two modes) ──────────────────────────────────────
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--images_dir", type=str,
                     help="Folder of pre-selected images (all .jpg/.png inside)")
    grp.add_argument("--images", nargs="+", type=str,
                     help="Explicit list of image file paths")
    # ── model ────────────────────────────────────────────────────────
    p.add_argument("--weights",       default="runs/yolov8n_detrac/best.pt", type=str)
    # ── output ───────────────────────────────────────────────────────
    p.add_argument("--output_dir",    default="results/exp0_color", type=str)
    # ── attack ───────────────────────────────────────────────────────
    p.add_argument("--target_color",  default="red",
                   choices=["red", "blue", "yellow", "green"])
    p.add_argument("--eps_z",         default=1.0,  type=float)
    p.add_argument("--num_steps",     default=100,  type=int)
    p.add_argument("--lr",            default=0.02, type=float)
    p.add_argument("--lambda_s",      default=2.0,  type=float,
                   help="Luminance preservation (higher = more structure preserved)")
    p.add_argument("--lambda_bg",     default=10.0, type=float,
                   help="Background preservation (keep high)")
    p.add_argument("--lambda_r",      default=1e-3, type=float)
    # ── paint mask ───────────────────────────────────────────────────
    p.add_argument("--L_min",         default=25.0, type=float,
                   help="Exclude pixels darker than L_min (tires, shadows). 0–100 scale.")
    p.add_argument("--L_max",         default=75.0, type=float,
                   help="Exclude pixels brighter than L_max (windows, lights). 0–100 scale.")
    # ── runtime ──────────────────────────────────────────────────────
    p.add_argument("--device",        default="cuda", type=str)
    p.add_argument("--imgsz",         default=640, type=int)
    p.add_argument("--conf_thr",      default=0.25, type=float)
    return p.parse_args()


def main():
    args   = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*68}")
    print(f"  Expérience 0 — Color Feasibility  (images sélectionnées manuellement)")
    print(f"  Target  : {args.target_color}")
    print(f"  Device  : {device}  |  eps_z={args.eps_z}  |  steps={args.num_steps}")
    print(f"{'='*68}\n")

    # ── collect image paths ───────────────────────────────────────────
    if args.images_dir:
        img_dir   = Path(args.images_dir)
        img_paths = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    else:
        img_paths = [Path(p) for p in args.images]

    if not img_paths:
        print("[ERROR] No images found. Check --images_dir or --images.")
        sys.exit(1)

    print(f"[main] {len(img_paths)} image(s) to process:")
    for p in img_paths:
        print(f"  {p.name}")
    print()

    # ── output dir ───────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── load models ──────────────────────────────────────────────────
    print("[main] Loading YOLOv8 …")
    detector = YOLOv8Wrapper(weights=str(ROOT / args.weights), device=device)

    print("[main] Loading SD-VAE …")
    vae = SDVAE(device=device)

    ab_target = TARGET_COLORS_AB[args.target_color].to(device)
    print(f"[main] Target (a, b) = {ab_target.tolist()}\n")

    # ── process each image ───────────────────────────────────────────
    results = []

    for i, img_path in enumerate(img_paths):
        img_name = img_path.stem
        print(f"[{i+1}/{len(img_paths)}] {img_name}")

        # ── load & letterbox ─────────────────────────────────────────
        x = load_image_tensor(str(img_path), imgsz=args.imgsz).to(device)
        _, _, H, W = x.shape

        # ── detect ───────────────────────────────────────────────────
        dets = detector.detect_nms(x, conf_thr=args.conf_thr, iou_thr=0.45)
        if not dets:
            print(f"  [skip] No detections in {img_name}.\n")
            continue
        print(f"  Detections : {len(dets)}")

        # ── mask ─────────────────────────────────────────────────────
        M  = boxes_to_pixel_mask(dets, H=H, W=W, device=device)
        Mz = pixel_mask_to_latent_mask(M, latent_channels=4, stride=8)

        # ── original color info ──────────────────────────────────────
        a_orig, b_orig = mean_ab_in_mask(x, M)
        print(f"  Source color : a={a_orig:+.1f}  b={b_orig:+.1f}")

        # ── encode ───────────────────────────────────────────────────
        z = vae.encode(x)

        # ── attack ───────────────────────────────────────────────────
        x_adv, history = run_color_attack(
            x=x, z=z, M=M, Mz=Mz,
            vae=vae,
            ab_target=ab_target,
            eps_z=args.eps_z,
            lr=args.lr,
            num_steps=args.num_steps,
            lambda_s=args.lambda_s,
            lambda_bg=args.lambda_bg,
            lambda_r=args.lambda_r,
            L_min=args.L_min,
            L_max=args.L_max,
            verbose=True,
        )

        # ── evaluate ─────────────────────────────────────────────────
        a_adv, b_adv = mean_ab_in_mask(x_adv, M)
        dist = ab_distance(x_adv, M, ab_target)
        print(f"  Modified color : a={a_adv:+.1f}  b={b_adv:+.1f}  "
              f"dist={dist:.2f}")

        # ── save ─────────────────────────────────────────────────────
        save_comparison(
            x_orig=x, x_adv=x_adv, M=M,
            img_name=img_name,
            target_color=args.target_color,
            ab_orig=(a_orig, b_orig),
            ab_adv=(a_adv, b_adv),
            dist=dist,
            output_path=out_dir / f"{img_name}_{args.target_color}.png",
        )
        tensor_to_pil(x_adv).save(str(out_dir / f"{img_name}_adv_only.png"))
        save_loss_curve(history, out_dir / f"{img_name}_loss.csv")

        results.append(dict(name=img_name,
                            dist=dist,
                            a_orig=a_orig, b_orig=b_orig,
                            a_adv=a_adv,  b_adv=b_adv))
        print()

    if not results:
        print("[ERROR] No images were processed.")
        sys.exit(1)

    print_summary(results, args.target_color, args)
    print(f"[main] Outputs → {out_dir}/\n")


if __name__ == "__main__":
    main()
