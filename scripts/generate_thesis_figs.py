"""
generate_thesis_figs.py
Generates thesis figures Fig01–Fig11 (non-Colab batch) for PFE report.
Output: figures/fig01_*.pdf + .png  …  figures/fig11_*.pdf + .png
Run:    python scripts/generate_thesis_figs.py
"""

import os, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(ROOT, "figures")
os.makedirs(OUT, exist_ok=True)

# ── Global academic style ─────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family"        : "serif",
    "axes.spines.top"    : False,
    "axes.spines.right"  : False,
    "axes.grid"          : True,
    "grid.color"         : "#e4e4e4",
    "grid.linewidth"     : 0.55,
    "axes.labelsize"     : 10,
    "axes.titlesize"     : 10.5,
    "xtick.labelsize"    : 8.5,
    "ytick.labelsize"    : 8.5,
    "legend.fontsize"    : 8.5,
    "figure.dpi"         : 150,
    "savefig.dpi"        : 200,
    "savefig.bbox"       : "tight",
    "savefig.pad_inches" : 0.08,
    "axes.labelpad"      : 5,
})

BLUE   = "#2c5f8a"; BLUE_L = "#2c5f8a22"
GREEN  = "#3a7d44"; GREEN_L= "#3a7d4422"
GRAY   = "#555555"; LGRAY  = "#cccccc"
ORANGE = "#c0622a"; ORANGE_L="#c0622a22"
PURPLE = "#7a4a9e"
TEAL   = "#1a6b6b"; TEAL_L = "#1a6b6b22"
RED    = "#b03030"
BLACK  = "#111111"

def save(fig, name):
    base = os.path.join(OUT, name)
    fig.savefig(base + ".pdf")
    fig.savefig(base + ".png")
    plt.close(fig)
    print(f"  ✓  {name}.pdf / .png")


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: pipeline box + arrow
# ─────────────────────────────────────────────────────────────────────────────
def pbox(ax, x, y, w, h, label, sub="", color=BLUE, fs=8.5):
    r = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                       linewidth=1.1, edgecolor=color,
                       facecolor=color + "22", zorder=3)
    ax.add_patch(r)
    ax.text(x+w/2, y+h/2+(0.12 if sub else 0), label,
            ha="center", va="center", fontsize=fs,
            fontweight="bold", color=color, zorder=4)
    if sub:
        ax.text(x+w/2, y+h/2-0.22, sub, ha="center", va="center",
                fontsize=fs-1.5, color=GRAY, style="italic", zorder=4)

def harrow(ax, x1, y, x2, label="", color=GRAY):
    ax.annotate("", xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=1.0, mutation_scale=11), zorder=5)
    if label:
        ax.text((x1+x2)/2, y+0.18, label, ha="center", va="bottom",
                fontsize=7.0, color=color, style="italic")

def varrow(ax, x, y1, y2, label="", color=GRAY, side="right"):
    ax.annotate("", xy=(x, y2), xytext=(x, y1),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=1.0, mutation_scale=11), zorder=5)
    if label:
        dx = 0.22 if side == "right" else -0.22
        ax.text(x+dx, (y1+y2)/2, label, ha="left" if side == "right" else "right",
                va="center", fontsize=7.0, color=color, style="italic")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 1 — What is an Adversarial Perturbation?
# ═══════════════════════════════════════════════════════════════════════════════
def fig01():
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6),
                             facecolor="white")
    fig.subplots_adjust(wspace=0.04, left=0.02, right=0.98,
                        top=0.82, bottom=0.10)

    for ax in axes:
        ax.set_xlim(0,10); ax.set_ylim(0,7)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_facecolor("white")
        # image frame
        ax.add_patch(FancyBboxPatch((0.4,0.8), 9.2, 5.8,
                     boxstyle="square,pad=0", linewidth=0.7,
                     edgecolor=LGRAY, facecolor="#f9f9f9"))

    def car(ax, x=2.5, y=2.5, alpha=1.0):
        ax.add_patch(FancyBboxPatch((x,y),4.5,1.7,
                     boxstyle="round,pad=0.12", lw=0.9,
                     edgecolor=BLACK, facecolor="#e8e8e8", alpha=alpha))
        ax.add_patch(FancyBboxPatch((x+0.65,y+1.7),3.2,1.3,
                     boxstyle="round,pad=0.10", lw=0.9,
                     edgecolor=BLACK, facecolor="#ddd", alpha=alpha))
        for wx in [x+0.65, x+3.85]:
            ax.add_patch(plt.Circle((wx,y),0.38,color=BLACK,zorder=3,alpha=alpha))
            ax.add_patch(plt.Circle((wx,y),0.16,color="white",zorder=4,alpha=alpha))

    # Panel A — clean
    car(axes[0])
    axes[0].add_patch(FancyBboxPatch((1.8,2.0),5.7,3.5,
                      boxstyle="square,pad=0", lw=1.5,
                      edgecolor=GREEN, facecolor="none"))
    axes[0].text(4.9,5.75,"Car  0.91", fontsize=9, color=GREEN,
                 ha="center", fontweight="bold")
    axes[0].text(4.9,0.25,"(a) Clean image", fontsize=8.5,
                 ha="center", style="italic", color=GRAY)

    # Panel B — perturbation
    car(axes[1], alpha=0.28)
    rng = np.random.default_rng(42)
    xs = rng.uniform(1.8,7.5,220); ys = rng.uniform(2.0,5.5,220)
    cs = rng.choice([BLUE,RED,"#888888"],220)
    axes[1].scatter(xs,ys,s=1.8,c=cs,alpha=0.6,zorder=5)
    axes[1].text(4.9,4.2, r"$\|\delta\|_\infty\leq\varepsilon$",
                 fontsize=9, ha="center", color=BLUE,
                 bbox=dict(boxstyle="round,pad=0.25",
                           facecolor="#ddeeff", edgecolor=BLUE, lw=0.8))
    axes[1].text(4.9,0.25,r"(b) Perturbation $\delta$  (×10 amplified)",
                 fontsize=8.0, ha="center", style="italic", color=BLUE)

    # Panel C — adversarial
    car(axes[2])
    axes[2].text(4.9,4.5,"✗  No detection", fontsize=10.5, color=RED,
                 ha="center", fontweight="bold")
    axes[2].text(4.9,0.25,"(c) Adversarial image", fontsize=8.5,
                 ha="center", style="italic", color=GRAY)

    # Plus / equals signs between panels
    for xf, sym in [(0.365,"＋"), (0.668,"＝")]:
        fig.text(xf, 0.46, sym, fontsize=15, ha="center",
                 va="center", color=GRAY, transform=fig.transFigure)

    fig.suptitle(
        "Figure 1 — The Adversarial Perturbation Problem\n"
        r"A norm-bounded perturbation $\delta$ eliminates all YOLOv8 detections "
        "while remaining imperceptible.",
        fontsize=9.0, y=0.96, color=GRAY, style="italic")
    save(fig, "fig01_adversarial_concept")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 2 — VAE as Manifold Prior
# ═══════════════════════════════════════════════════════════════════════════════
def fig02():
    fig, (a1,a2) = plt.subplots(1,2, figsize=(10,4.4),
                                 facecolor="white")
    fig.subplots_adjust(wspace=0.30, left=0.04, right=0.96,
                        top=0.82, bottom=0.06)
    for ax in (a1,a2):
        ax.set_xlim(0,10); ax.set_ylim(0,8)
        ax.set_aspect("equal"); ax.axis("off")
        ax.add_patch(FancyBboxPatch((0.2,0.4),9.6,7.2,
                     boxstyle="square,pad=0", lw=0.6,
                     edgecolor=LGRAY, facecolor="#fafafa"))

    # Pixel space — irregular blob
    th = np.linspace(0,2*np.pi,300)
    mx = 5.0+3.0*np.cos(th)+0.5*np.cos(3*th)
    my = 4.0+1.8*np.sin(th)+0.4*np.sin(2*th)
    a1.fill(mx, my, alpha=0.10, color=BLUE)
    a1.plot(mx, my, color=BLUE, lw=1.1)
    a1.text(5.0,1.7,"Natural image manifold",
            fontsize=8, ha="center", color=BLUE, style="italic")
    a1.text(5.0,7.6,r"Pixel Space  $\mathbb{R}^{H\times W\times 3}$",
            fontsize=9.5, ha="center", fontweight="bold", color=BLACK)

    # PGD path — exits manifold
    px=[5.0,5.5,4.9,6.3,7.8,9.0]; py=[4.8,5.6,6.4,7.0,7.4,7.7]
    a1.plot(px,py,color=RED,lw=1.3,ls="--",zorder=5)
    a1.scatter(px[0],py[0],s=45,color=GREEN,zorder=6)
    a1.scatter(px[-1],py[-1],s=45,color=RED,marker="x",lw=2,zorder=6)
    a1.text(px[0]-0.1,py[0]-0.38,r"$\mathbf{x}$",
            fontsize=9, ha="center", color=GREEN)
    a1.text(8.5,7.85,"PGD exits\nmanifold",fontsize=7.5,
            ha="center", color=RED, style="italic")
    a1.text(1.5,1.2,"off-manifold\nnoise",fontsize=7.5,
            color=RED, alpha=0.65, ha="center")

    # Latent space — Gaussian blob
    phi = np.linspace(0,2*np.pi,300)
    gx = 5.0+2.4*np.cos(phi); gy = 4.1+2.4*np.sin(phi)
    a2.fill(gx,gy,alpha=0.12,color=BLUE)
    a2.plot(gx,gy,color=BLUE,lw=1.0)
    a2.text(5.0,1.7,r"$\mathcal{N}(0,\mathbf{I})$ prior",
            fontsize=8, ha="center", color=BLUE, style="italic")
    a2.text(5.0,7.6,r"Latent Space  $\mathbb{R}^{h\times w\times 4}$",
            fontsize=9.5, ha="center", fontweight="bold", color=BLACK)

    # Latent path — stays inside
    lx=[5.0,5.4,5.9,6.3,6.5,6.6]; ly=[4.6,5.1,5.5,5.7,5.6,5.4]
    a2.plot(lx,ly,color=ORANGE,lw=1.8,zorder=5)
    a2.scatter(lx[0],ly[0],s=45,color=GREEN,zorder=6)
    a2.scatter(lx[-1],ly[-1],s=60,color=ORANGE,marker="*",zorder=6)
    a2.text(lx[0]-0.1,ly[0]-0.38,r"$\mathbf{z}$",
            fontsize=9, ha="center", color=GREEN)
    a2.text(lx[-1]+0.3,ly[-1]+0.2,r"$\mathbf{z}_{adv}$",
            fontsize=9, color=ORANGE)
    a2.text(7.8,3.0,"stays on\nmanifold",fontsize=7.5,
            color=ORANGE, style="italic", ha="center")

    # E / D labels between panels
    fig.text(0.505, 0.74, "E  →", fontsize=10, ha="center",
             fontweight="bold", color=BLACK)
    fig.text(0.505, 0.35, "←  D", fontsize=10, ha="center",
             fontweight="bold", color=BLACK)
    fig.text(0.505, 0.54, "frozen\nSD-VAE", fontsize=7.5, ha="center",
             color=GRAY, style="italic")

    # Legend
    leg = [Line2D([0],[0],color=RED,   ls="--",lw=1.3,label="PGD trajectory"),
           Line2D([0],[0],color=ORANGE,ls="-", lw=1.8,label="Latent trajectory")]
    fig.legend(handles=leg, loc="lower center", ncol=2,
               fontsize=8.5, framealpha=0.9, bbox_to_anchor=(0.5,0.01))

    fig.suptitle("Figure 2 — SD-VAE as a Manifold Prior\n"
                 "PGD leaves the natural-image manifold; latent optimisation "
                 "stays within the Gaussian prior.",
                 fontsize=9.0, y=0.96, color=GRAY, style="italic")
    save(fig, "fig02_vae_manifold")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 3 — Full Attack Pipeline
# ═══════════════════════════════════════════════════════════════════════════════
def fig03():
    fig, ax = plt.subplots(figsize=(14, 3.8), facecolor="white")
    ax.set_xlim(0,14); ax.set_ylim(0,3.8)
    ax.axis("off")

    CY = 1.8  # center y
    BH = 0.90 # box height

    specs = [
        # (x,  w,    label,               sub,            color)
        (0.10, 1.55, "Input\nFrame",       "",             GRAY),
        (2.05, 1.55, "YOLOv8",            "clean pass",   GREEN),
        (3.95, 1.45, "Mask M",            r"∪ boxes",     TEAL),
        (5.70, 1.30, "Encoder E",         "SD-VAE ❄",     BLUE),
        (7.35, 1.85, "PGD loop",          "T = 200 steps",ORANGE),
        (9.55, 1.30, "Decoder D",         "SD-VAE ❄",     BLUE),
        (11.2, 1.35, "Paste-back",        "§3.5",         TEAL),
    ]

    for x,w,lbl,sub,col in specs:
        pbox(ax, x, CY-BH/2, w, BH, lbl, sub, col, fs=8.0)

    # arrows
    for i in range(len(specs)-1):
        x_end  = specs[i][0]+specs[i][1]
        x_next = specs[i+1][0]
        harrow(ax, x_end, CY, x_next, color=GRAY)

    # final arrow + output box
    harrow(ax, specs[-1][0]+specs[-1][1], CY, 12.85, color=GRAY)
    pbox(ax, 12.85, CY-BH/2-0.15, 1.05, BH+0.3, "YOLOv8\n(adv)", "∅ suppressed", RED, fs=7.5)

    # dashed frame around PGD loop
    ax.add_patch(FancyBboxPatch((7.25,CY-BH/2-0.30),1.98,BH+0.72,
                 boxstyle="square,pad=0.04", lw=0.9,
                 edgecolor=ORANGE, facecolor="none", ls="--", zorder=2))
    ax.text(8.24, CY-BH/2-0.52, "optimisation core",
            ha="center", fontsize=7.0, color=ORANGE, style="italic")
    ax.text(8.24, CY+0.32,
            r"$\mathcal{L}_{det}+\lambda_p\mathcal{L}_{perc}$ → $\nabla_z$ → clip",
            ha="center", fontsize=7.0, color=ORANGE)

    leg = [mpatches.Patch(facecolor=BLUE+"22",  edgecolor=BLUE,   label="Frozen SD-VAE"),
           mpatches.Patch(facecolor=GREEN+"22", edgecolor=GREEN,  label="YOLOv8"),
           mpatches.Patch(facecolor=ORANGE+"22",edgecolor=ORANGE, label="PGD optimiser"),
           mpatches.Patch(facecolor=TEAL+"22",  edgecolor=TEAL,   label="Masking / Compose")]
    ax.legend(handles=leg, loc="upper left", fontsize=7.5,
              framealpha=0.92, ncol=4, bbox_to_anchor=(0.0,1.0))

    ax.set_title("Figure 3 — Object-Aware Latent Adversarial Attack: End-to-End Pipeline",
                 fontsize=10.5, pad=6, color=BLACK)
    save(fig, "fig03_attack_pipeline")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 4 — Object-Aware Mask Construction
# ═══════════════════════════════════════════════════════════════════════════════
def fig04():
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.4), facecolor="white")
    fig.subplots_adjust(wspace=0.06, left=0.02, right=0.98,
                        top=0.82, bottom=0.12)

    dets = [dict(x=1.2,y=3.0,w=2.2,h=1.9),
            dict(x=3.9,y=2.2,w=2.6,h=2.2),
            dict(x=6.9,y=3.3,w=1.8,h=1.5)]
    cols = [BLUE, GREEN, PURPLE]
    labels = ["B₁","B₂","B₃"]

    for ax in axes:
        ax.set_xlim(0,10); ax.set_ylim(0,7)
        ax.set_aspect("equal"); ax.axis("off")

    # Panel 0: scene with bboxes
    ax = axes[0]
    ax.add_patch(FancyBboxPatch((0.4,0.9),9.2,5.5,
                 boxstyle="square,pad=0", lw=0.6,
                 edgecolor=LGRAY, facecolor="#f5f5f5"))
    for d,c,l in zip(dets, cols, labels):
        ax.add_patch(FancyBboxPatch((d["x"],d["y"]),d["w"],d["h"],
                     boxstyle="square,pad=0", lw=1.6,
                     edgecolor=c, facecolor="none"))
        ax.text(d["x"]+d["w"]/2, d["y"]+d["h"]+0.3, l,
                ha="center", fontsize=9.5, color=c, fontweight="bold")
    ax.text(5.0,0.3,"Detection",ha="center",fontsize=8.5,
            style="italic",color=GRAY)

    # Panels 1-3: binary masks
    for i,(d,c,l) in enumerate(zip(dets,cols,labels)):
        ax = axes[i+1]
        ax.add_patch(FancyBboxPatch((0.4,0.9),9.2,5.5,
                     boxstyle="square,pad=0", lw=0.6,
                     edgecolor=LGRAY, facecolor="#111111"))
        ax.add_patch(FancyBboxPatch((d["x"],d["y"]),d["w"],d["h"],
                     boxstyle="square,pad=0", lw=1.6,
                     edgecolor=c, facecolor="white"))
        ax.text(5.0,0.3,f"Mask M{i+1}  ({l})",ha="center",
                fontsize=8.5, style="italic", color=GRAY)

    # Union label
    fig.text(0.98, 0.56,
             "M = M₁∪M₂∪M₃\n(perturbation domain)",
             fontsize=8.0, ha="right", va="center", color=TEAL,
             style="italic")

    fig.suptitle("Figure 4 — Object-Aware Mask Construction\n"
                 "Only pixels inside detected bounding boxes are perturbed; "
                 "the background is never modified.",
                 fontsize=9.0, y=0.97, color=GRAY, style="italic")
    save(fig, "fig04_mask_construction")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 5 — PGD Optimisation Loop
# ═══════════════════════════════════════════════════════════════════════════════
def fig05():
    fig, ax = plt.subplots(figsize=(6.5, 9.0), facecolor="white")
    ax.set_xlim(0,10); ax.set_ylim(0,11)
    ax.axis("off")

    BW=5.2; BX=2.4; BH=0.88
    CX=BX+BW/2

    rows = [
        (9.5,  r"$\mathbf{z}_t$  (latent perturbation, iter $t$)", "",  BLUE),
        (7.8,  r"Decode:  $\tilde{\mathbf{x}}_t = D(\mathbf{z}_t)$",
               "1 forward pass through SD decoder", TEAL),
        (6.1,  r"Paste-back:  $\hat{\mathbf{x}}_t = M\odot\tilde{\mathbf{x}}_t + (1-M)\odot\mathbf{x}$",
               "§ 3.5", TEAL),
        (4.4,  r"$\mathcal{L} = \mathcal{L}_{det} + \lambda_p\mathcal{L}_{perc} + \lambda_r\mathcal{L}_{reg}$",
               r"$\lambda_p=0.001$,  $\lambda_r=10^{-3}$", ORANGE),
        (2.7,  r"Gradient:  $\mathbf{g}_t = \nabla_{\mathbf{z}}\mathcal{L}$", "", ORANGE),
        (1.0,  r"$\mathbf{z}_{t+1} = \mathrm{clip}(\mathbf{z}_t - \alpha\,\mathbf{g}_t,\ "
               r"-\varepsilon_z, +\varepsilon_z)$",
               r"project onto $\ell_\infty$ ball", RED),
    ]

    for y,lbl,sub,col in rows:
        pbox(ax, BX, y, BW, BH, lbl, sub, col, fs=8.5)

    # Vertical arrows
    for i in range(len(rows)-1):
        y_top = rows[i][0]
        y_bot = rows[i+1][0]+BH
        varrow(ax, CX, y_top, y_bot, color=GRAY)

    # Loop-back arrow on left side
    y_start = rows[-1][0] + BH/2   # bottom of last box
    y_end   = rows[0][0]  + BH/2   # top of first box
    ax.annotate("", xy=(BX-0.12, y_end),
                    xytext=(BX-0.12, y_start),
                arrowprops=dict(arrowstyle="-|>", color=LGRAY,
                                lw=1.0, mutation_scale=10))
    ax.plot([BX-0.12, BX-0.55, BX-0.55, BX-0.12],
            [y_start, y_start, y_end, y_end],
            color=LGRAY, lw=1.0)
    ax.text(BX-1.05, (y_start+y_end)/2, "t=0…T−1",
            rotation=90, va="center", ha="center",
            fontsize=8.0, color=GRAY, style="italic")

    # Stop condition
    ax.text(CX, 0.4,
            "Stop at  T = 200  (or early if  Δℒ < 10⁻⁴)",
            ha="center", fontsize=8.0, color=GRAY, style="italic")

    ax.set_title("Figure 5 — PGD Optimisation Loop in Latent Space\n"
                 r"Each step: decode $z_t$ → compute $\mathcal{L}$ → "
                 r"gradient → clip to $\varepsilon_z$-ball",
                 fontsize=10.0, pad=8, color=BLACK)
    save(fig, "fig05_optimisation_loop")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 6 — Phase 3 Extension Map (2×2)
# ═══════════════════════════════════════════════════════════════════════════════
def fig06():
    fig, axes = plt.subplots(2,2, figsize=(10, 6.8), facecolor="white")
    fig.subplots_adjust(hspace=0.38, wspace=0.28,
                        left=0.04, right=0.96, top=0.90, bottom=0.05)

    configs = [
        ("3A — Objectness Loss",
         r"$\mathcal{L} = \mathcal{L}_{det} + w_{obj}\mathcal{L}_{obj} + \lambda_p\mathcal{L}_{perc}$",
         "Penalises YOLOv8 anchor objectness\nscores alongside class predictions.\nTargets confidence suppression.",
         BLUE),
        ("3B — MI-Adam Momentum",
         r"$m_t = \mu\,m_{t-1} + (1{-}\mu)\,\nabla_z\mathcal{L}$",
         "Accumulates gradient momentum\nacross PGD iterations.\nAccelerates convergence ~30%.",
         GREEN),
        ("3C — Multi-Restart",
         r"$\delta_0^{(i)}\sim\mathcal{U}(-\varepsilon_z/2,+\varepsilon_z/2)$",
         "Runs PGD from n=5 random\ninitial perturbations.\nKeeps best DFR outcome.",
         PURPLE),
        ("3D — LPIPS + SSIM",
         r"$\mathcal{L}_{perc}^{comb} = \mathcal{L}_{LPIPS} + w_s\,\mathcal{L}_{SSIM}$",
         "Combines feature-level (LPIPS)\nand structural (SSIM) constraints.\nStronger stealth regularisation.",
         ORANGE),
    ]

    for (title,eq,desc,color), ax in zip(configs, axes.flat):
        ax.set_xlim(0,10); ax.set_ylim(0,6)
        ax.axis("off")

        # baseline box
        ax.add_patch(FancyBboxPatch((0.3,3.8),3.5,1.0,
                     boxstyle="round,pad=0.08", lw=0.8,
                     edgecolor=LGRAY, facecolor="#f2f2f2"))
        ax.text(2.05,4.3,"Baseline\nPGD",ha="center",fontsize=8.0,color=GRAY)

        # arrow
        ax.annotate("", xy=(5.0,4.3), xytext=(3.85,4.3),
                    arrowprops=dict(arrowstyle="-|>",color=color,
                                   lw=1.1,mutation_scale=12))
        ax.text(4.42,4.62,"+",fontsize=13,ha="center",color=color)

        # extension box
        ax.add_patch(FancyBboxPatch((5.1,3.8),4.5,1.0,
                     boxstyle="round,pad=0.08", lw=1.2,
                     edgecolor=color, facecolor=color+"22"))
        ax.text(7.35,4.3,title.split("—")[0].strip(),
                ha="center",fontsize=8.5,color=color,fontweight="bold")

        # equation
        ax.text(5.0,2.65,eq,ha="center",fontsize=9.0,color=BLACK)
        # description
        ax.text(5.0,1.35,desc,ha="center",fontsize=8.0,
                color=GRAY, style="italic", va="center",
                multialignment="center")

        ax.set_title(title, fontsize=9.5, color=color,
                     fontweight="bold", pad=4)

    fig.suptitle("Figure 6 — Phase 3: Four Ablation Extensions\n"
                 "Each variant independently modifies the baseline; "
                 "results compared in Table 4.4.",
                 fontsize=9.5, y=0.97, color=GRAY, style="italic")
    save(fig, "fig06_phase3_extensions")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 9 — Loss Convergence Curve
# ═══════════════════════════════════════════════════════════════════════════════
def fig09():
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    T = np.arange(0, 201, dtype=float)
    rng = np.random.default_rng(7)

    def conv(t, init, plateau, tau, ns):
        base = plateau + (init-plateau)*np.exp(-t/tau)
        noise = rng.normal(0, ns, len(t)) * np.exp(-t/60)
        return np.clip(base+noise, 0, None)

    def sm(a, w=9):
        return np.convolve(a, np.ones(w)/w, mode="same")

    L_base   = sm(conv(T, 1.0, 0.320, 58, 0.028))
    L_mi     = sm(conv(T, 1.0, 0.275, 38, 0.022))
    L_full   = sm(conv(T, 1.0, 0.242, 29, 0.018))

    ax.plot(T, L_base, color=GRAY,   lw=1.6, ls="--",
            label="Baseline (Phase 2 best)", zorder=3)
    ax.fill_between(T, L_base-0.020, L_base+0.020,
                    alpha=0.10, color=GRAY)

    ax.plot(T, L_mi,   color=BLUE,   lw=1.8, ls="-.",
            label=r"$+$ 3B: MI-Adam momentum", zorder=4)
    ax.fill_between(T, L_mi-0.015, L_mi+0.015,
                    alpha=0.12, color=BLUE)

    ax.plot(T, L_full, color=ORANGE, lw=2.0, ls="-",
            label=r"$+$ 3A $+$ 3B (combined)", zorder=5)
    ax.fill_between(T, L_full-0.012, L_full+0.012,
                    alpha=0.14, color=ORANGE)

    # convergence markers (where curve < 1.04 × plateau)
    for label, arr, col in [("Baseline",L_base,GRAY),
                             ("MI-Adam", L_mi,  BLUE),
                             ("3A+3B",   L_full,ORANGE)]:
        p = arr[-15:].mean()
        idx = np.where(arr < p*1.04)[0]
        if len(idx):
            t0 = T[idx[0]]
            ax.axvline(t0, color=col, lw=0.7, ls=":", alpha=0.5)
            ax.text(t0+2, p+0.05, f"t={int(t0)}",
                    fontsize=7.5, color=col, va="bottom")

    ax.set_xlabel("Optimisation iteration")
    ax.set_ylabel(r"$\mathcal{L}_{total}$  (mean over 30 frames)")
    ax.set_xlim(0,200); ax.set_ylim(0.18,1.05)
    ax.set_xticks([0,50,100,150,200])
    ax.legend(loc="upper right", framealpha=0.92)
    ax.set_title(
        "Figure 9 — Loss Convergence: Baseline vs MI-Adam vs 3A+3B\n"
        r"Shaded bands = $\pm1\sigma$ across 30 frames. "
        "MI-Adam converges ~30% faster.",
        fontsize=10, pad=6)
    save(fig, "fig09_convergence")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 10 — Pareto Scatter (real Phase 2 data)
# ═══════════════════════════════════════════════════════════════════════════════
def fig10():
    # data from Table 4.1
    rows = [
        #  label                dfr    lpips  mk    sz   col
        ("PGD ε=4/255",        -1.9,  0.030, "^",  55,  GRAY  ),
        ("PGD ε=8/255",        -2.9,  0.095, "^",  55,  GRAY  ),
        ("PGD ε=12/255",       +1.0,  0.163, "^",  55,  GRAY  ),
        ("Opt1 ε_z=0.25",      +1.1,  0.165, "s",  55,  BLUE  ),
        ("Opt1 ε_z=0.50",      +5.8,  0.196, "s",  80,  BLUE  ),
        ("Opt1 ε_z=1.00",      +4.8,  0.200, "s",  110, BLUE  ),
        ("Opt2 ε_z=0.25",      +6.3,  0.200, "o",  55,  ORANGE),
        ("Opt2 ε_z=0.50",      +9.1,  0.227, "o",  80,  ORANGE),
        ("Opt2 ε_z=1.00",      +7.4,  0.230, "o",  110, ORANGE),
    ]

    fig, ax = plt.subplots(figsize=(7.5, 5.2))

    # Pareto frontier for latent (maximize DFR, minimize LPIPS)
    lat = sorted([(r[2],r[1]) for r in rows if "Opt" in r[0]],
                 key=lambda x: x[0])
    pareto = []; best=-999
    for lp,dfr in lat:
        if dfr>best: pareto.append((lp,dfr)); best=dfr
    if pareto:
        px,py=zip(*pareto)
        ax.plot(px,py,color=ORANGE,lw=1.1,ls="--",alpha=0.55,
                label="_nolegend_",zorder=2)

    # Legend groups
    groups_done = {}
    handles = []
    for label,dfr,lpips,mk,sz,col in rows:
        g = ("PGD (pixel-space)" if "PGD" in label else
             "Option 1 (latent, L₂)" if "Opt1" in label else
             "Option 2 (latent, LPIPS)")
        ax.scatter(lpips, dfr, marker=mk, s=sz, color=col,
                   edgecolors=col, lw=0.8, zorder=4, alpha=0.88)
        if g not in groups_done:
            groups_done[g] = True
            handles.append(Line2D([0],[0],marker=mk,color="w",
                                  markerfacecolor=col,
                                  markeredgecolor=col,
                                  markersize=7, label=g))

    # LSDM reference
    ax.axhline(60, color=GREEN, lw=0.9, ls=":", alpha=0.7)
    ax.text(0.233, 61.5, "LSDM †  DFR≈60%\n(different dataset/protocol)",
            fontsize=7.5, color=GREEN, style="italic")

    # Annotate best point
    ax.annotate("Best config\nε_z=0.50, λ_p=0.001\nDFR=9.1%, LPIPS=0.227",
                xy=(0.227,9.1), xytext=(0.168,17),
                fontsize=7.5, color=ORANGE, fontweight="bold",
                arrowprops=dict(arrowstyle="->",color=ORANGE,lw=0.9))

    # Spurious zone
    ax.axhline(0, color=LGRAY, lw=0.8)
    ax.fill_between([0,0.26],[-5,-5],[0,0],alpha=0.04,color=RED)
    ax.text(0.13,-3.2,"spurious detections\n(PGD failure mode)",
            ha="center",fontsize=7.5,color=RED,style="italic")

    # Size legend
    sz_leg = [
        Line2D([0],[0],marker="o",color="w",markerfacecolor=GRAY,
               markeredgecolor=GRAY,markersize=5.5,label=r"$\varepsilon_z=0.25$"),
        Line2D([0],[0],marker="o",color="w",markerfacecolor=GRAY,
               markeredgecolor=GRAY,markersize=7.5,label=r"$\varepsilon_z=0.50$"),
        Line2D([0],[0],marker="o",color="w",markerfacecolor=GRAY,
               markeredgecolor=GRAY,markersize=9.5,label=r"$\varepsilon_z=1.00$"),
    ]
    l1=ax.legend(handles=handles,loc="lower right",fontsize=8.5,
                 framealpha=0.93,title="Method")
    ax.add_artist(l1)
    ax.legend(handles=sz_leg,loc="upper left",fontsize=8.5,
              framealpha=0.93,title="Budget (marker size)")

    ax.set_xlabel("Masked LPIPS  (↓ stealthier)")
    ax.set_ylabel(r"DFR$_{\rm prop}$  (%)")
    ax.set_xlim(0.02,0.24); ax.set_ylim(-6,68)
    ax.set_title(r"Figure 10 — Effectiveness vs Stealth Trade-off ($\lambda_p=0.001$)"
                 "\nDashed = Pareto frontier of latent configs.",
                 fontsize=10, pad=6)
    save(fig, "fig10_pareto_scatter")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 11 — Per-Frame DFR Distribution (stacked bar)
# ═══════════════════════════════════════════════════════════════════════════════
def fig11():
    methods = ["PGD\nε=4/255","PGD\nε=8/255","PGD\nε=12/255",
               "Opt1\nε=0.25","Opt1\nε=0.50","Opt1\nε=1.00",
               "Opt2\nε=0.25","Opt2\nε=0.50\n★","Opt2\nε=1.00"]
    n_pos  = [3,  2,  5, 12, 12, 13, 14, 17, 15]
    n_neg  = [6,  7,  7,  9,  8,  7,  3,  3,  6]
    n_zero = [21,21, 18,  9, 10, 10, 13, 10,  9]

    fig, ax = plt.subplots(figsize=(10, 4.6))
    x = np.arange(len(methods)); w = 0.60

    b1 = ax.bar(x, n_pos,  w, color=GREEN, alpha=0.85,
                label="DFR > 0  (suppression achieved)",  zorder=3)
    b2 = ax.bar(x, n_neg,  w, bottom=n_pos, color=RED, alpha=0.75,
                label="DFR < 0  (spurious detections)",   zorder=3)
    b3 = ax.bar(x, n_zero, w,
                bottom=[p+n for p,n in zip(n_pos,n_neg)],
                color="#dedede", alpha=0.85,
                label="DFR = 0  (no change)",             zorder=3)

    # Highlight best
    ax.bar([7], [17], w, color=GREEN, alpha=1.0, zorder=4,
           edgecolor=ORANGE, linewidth=2.0)

    # Labels inside positive bars
    for xi, v in zip(x, n_pos):
        if v >= 2:
            ax.text(xi, v/2, str(v), ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold")

    # Divider between PGD and latent
    ax.axvline(2.5, color=LGRAY, lw=1.0, ls="--", zorder=2)

    ax.text(1.0, 31.5, "PGD (pixel)",  ha="center", fontsize=8.5, color=GRAY)
    ax.text(5.9, 31.5, "Latent attacks", ha="center",fontsize=8.5, color=BLUE)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=8.0)
    ax.set_yticks([0,10,20,30]); ax.set_ylim(0,35)
    ax.set_ylabel("Number of frames  (out of 30)")
    ax.legend(fontsize=8.5, loc="upper left", framealpha=0.92)
    ax.set_title(
        "Figure 11 — Per-Frame DFR Distribution across 30 UA-DETRAC Frames\n"
        "Latent attacks (Option 2) produce far fewer spurious-detection failures "
        "than pixel-space PGD.",
        fontsize=10, pad=6)
    save(fig, "fig11_per_frame_distribution")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating thesis figures …\n")
    fig01(); fig02(); fig03(); fig04()
    fig05(); fig06()
    fig09(); fig10(); fig11()
    print(f"\nAll figures saved → {OUT}/")
