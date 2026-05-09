# Improvement Roadmap — Object-Aware Latent Adversarial Attack on YOLOv8

**Author:** Mohamed Aziz Brahmi
**Status:** Living document, revised with each milestone
**Goal:** Evolve current proof-of-concept into publication-quality framework

---

## 1. Critical weak points in the current framework

Identified from `experiments.md` (50 frames, single sequence, May 2026):

1. **Metric implementation contradicts method description.** `evaluate.py` uses loose DFR/ASR; `method.tex` defines stricter versions. With strict metrics, current ASR=0.76 likely drops to ~0.30–0.45. **This is a publication blocker.**
2. **Comparisons are not iso-budget.** `eps_z=0.50` (latent L∞) is not commensurable with `eps=8/255` (pixel L∞). The "3.4× more effective than PGD" claim is partly an artifact of unequal budgets. No Pareto curve exists.
3. **Sample size too small.** 50 frames, single sequence. No bootstrap CI, no multi-sequence variance estimate.
4. **PSNR is a poor stealth metric for structured perturbations.** Latent perturbations produce coherent artifacts; pixel attacks produce high-frequency noise. At equal PSNR, perceptual ranking can flip.
5. **No ablation studies.** No defense for `lambda_p=0.05`, `lambda_r=1e-3`, Adam vs. PGD, bbox-restricted mask, or step count.
6. **No temporal coherence on a video dataset.** UA-DETRAC is video; current pipeline treats frames i.i.d. Flicker is a likely failure mode.
7. **Single detector.** Only YOLOv8n nano. No transferability evidence.
8. **VAE is out-of-distribution for surveillance footage.** Base reconstruction error ~0.33 max pixel diff at delta=0 is the dominant driver of low PSNR_mask.

---

## 2. Prioritized roadmap

Ordered to de-risk each phase before the next. Each phase produces a measurable deliverable.

### P0 — Foundation fixes (1–2 days)

These are blockers. Do them first; everything downstream depends on having correct metrics and a fair protocol.

| # | Task | Deliverable |
|---|---|---|
| P0.1 | Rewrite `evaluate.py` with strict DFR/ASR/mAP-drop matching `method.tex` | `eval/metrics.py` with unit tests |
| P0.2 | Define iso-budget evaluation protocol (sweep budgets, build Pareto curves) | `configs/iso_budget.yaml`, `eval/pareto.py` |
| P0.3 | Expand evaluation set to ≥3 DETRAC sequences, ≥500 frames | `data/eval_split.json` with documented selection |
| P0.4 | Add bootstrap CI computation to all reported metrics | `eval/bootstrap.py`, 95% CI on every number |

### P1 — Core scientific contributions (1–2 weeks)

These are the methodological improvements that constitute the paper's added value.

| # | Task | Deliverable |
|---|---|---|
| P1.1 | Replace masked L2 with masked LPIPS perceptual loss | `attack/losses/perceptual.py` |
| P1.2 | Add temporal consistency loss (optical-flow-warped latent regularization) | `attack/losses/temporal.py` |
| P1.3 | Improved latent regularization (smoothness term, Tikhonov-style) | `attack/losses/regularization.py` |
| P1.4 | Fine-tune SD-VAE on DETRAC (10–20 epochs) | `vae/finetune.py`, fine-tuned checkpoint |
| P1.5 | Full ablation grid: eps_z, lambda_p, lambda_temp, num_steps, restarts | Ablation table in `results/ablations/` |

### P2 — Rigor improvements (1 week)

| # | Task | Deliverable |
|---|---|---|
| P2.1 | Black-box transferability: YOLOv8s/m/l, RT-DETR, Faster R-CNN | `eval/transfer.py`, transfer table |
| P2.2 | Per-frame variance and worst-case analysis | Frame-level scatter plots |
| P2.3 | Runtime/FLOPs profiling for all attacks | Runtime comparison table |
| P2.4 | Statistical significance tests (paired Wilcoxon for matched-frame comparisons) | p-values in main table |

### P3 — Publication outputs (3–5 days)

| # | Task | Deliverable |
|---|---|---|
| P3.1 | Pareto plots: DFR vs. PSNR, DFR vs. LPIPS | Figures 3a, 3b |
| P3.2 | Qualitative comparison grids (clean / adv / diff × 3 attacks) | Figure 4 |
| P3.3 | Per-pixel perturbation heatmaps | Figure 5 |
| P3.4 | Detection overlay videos (clean vs. adversarial side-by-side) | Supplementary video |
| P3.5 | Temporal stability visualization (delta_t evolution) | Figure 6 |

---

## 3. Implementation details — P0

### P0.1 — Corrected metrics

Mathematical definitions (matching `method.tex`):

- **DFR (per-frame, then averaged):**
  Given clean detections D_c^(f) and adversarial detections D_a^(f) for frame f,
  DFR = (1/N_frames) · Σ_f [1 − |D_a^(f)| / max(|D_c^(f)|, 1)]

- **ASR (strict):**
  ASR = (1/N_frames) · Σ_f 1[every originally-detected class in D_c^(f) is absent from D_a^(f) at IoU ≥ 0.5]

- **mAP drop:**
  Treat D_c as pseudo-ground-truth. Compute mAP@0.5 and mAP@0.5:0.95 of D_a against D_c. Report drop from baseline (which is mAP=1.0 by construction).

**Implementation skeleton (`eval/metrics.py`):**

```python
import numpy as np
from torchvision.ops import box_iou

def per_frame_metrics(clean_boxes, clean_scores, clean_classes,
                      adv_boxes, adv_scores, adv_classes,
                      iou_thr=0.5, conf_thr=0.25):
    # Filter by confidence threshold
    c_keep = clean_scores >= conf_thr
    a_keep = adv_scores >= conf_thr

    n_clean = int(c_keep.sum())
    n_adv = int(a_keep.sum())

    # Per-frame DFR
    dfr = 1.0 - n_adv / max(n_clean, 1)

    # Strict ASR: every clean class must disappear
    survived = False
    if n_clean > 0:
        ious = box_iou(clean_boxes[c_keep], adv_boxes[a_keep]) if n_adv > 0 else None
        for i in range(n_clean):
            cls_i = clean_classes[c_keep][i].item()
            if ious is not None:
                # any adv detection of same class with IoU >= thr counts as survival
                matches = (adv_classes[a_keep] == cls_i) & (ious[i] >= iou_thr)
                if matches.any():
                    survived = True
                    break
    asr_frame = float(n_clean > 0 and not survived)

    # Confidence drop on matched detections
    conf_drop = compute_matched_conf_drop(...)

    return dict(dfr=dfr, asr=asr_frame, n_clean=n_clean, n_adv=n_adv,
                conf_drop=conf_drop)


def dataset_metrics(per_frame_results):
    valid = [r for r in per_frame_results if r['n_clean'] > 0]
    return dict(
        DFR=np.mean([r['dfr'] for r in valid]),
        ASR=np.mean([r['asr'] for r in valid]),
        n_frames=len(valid),
    )
```

Add a unit test that runs both old and new metrics on the existing 50-frame run; document the discrepancy in the paper transition.

### P0.2 — Iso-budget protocol

For each attack, sweep its native budget and record (DFR, LPIPS, PSNR_mask, runtime).

```yaml
# configs/iso_budget.yaml
sweeps:
  latent:
    eps_z: [0.10, 0.25, 0.50, 0.75, 1.00]
    fixed: {lr: 0.05, num_steps: 200, lambda_p: 0.05}
  pgd:
    eps: [2, 4, 6, 8, 10]   # divide by 255
    fixed: {alpha: 1, num_steps: 50}  # alpha in 1/255 units
  fgsm:
    eps: [2, 4, 6, 8, 10]
    fixed: {}
```

Plot DFR (y) vs. LPIPS (x) with one curve per method. Your method wins iff its curve is **dominantly above** the others — not just at one operating point. Report the area-under-Pareto as a single summary number.

### P0.3 — Evaluation set expansion

Pick 3–5 DETRAC sequences spanning diverse conditions:
- One urban dense (MVI_40171 or similar)
- One highway sparse (MVI_39271)
- One night/low-light (MVI_40991)
- One with heavy occlusion (MVI_40991 or MVI_40701)

Sample 100–200 frames per sequence. Document selection rule (e.g., every 5th frame to ensure motion variation). Report per-sequence metrics in a supplementary table.

### P0.4 — Bootstrap CI

```python
def bootstrap_ci(per_frame_values, n_boot=1000, ci=0.95):
    rng = np.random.default_rng(42)
    boots = []
    for _ in range(n_boot):
        sample = rng.choice(per_frame_values, size=len(per_frame_values), replace=True)
        boots.append(np.mean(sample))
    lo = np.quantile(boots, (1-ci)/2)
    hi = np.quantile(boots, 1-(1-ci)/2)
    return float(np.mean(per_frame_values)), float(lo), float(hi)
```

Report all metrics as `mean [lo, hi]`.

---

## 4. Implementation details — P1

### P1.1 — Masked LPIPS loss

Per-bbox crop LPIPS aggregated by mean:

```python
import lpips
import torch.nn.functional as F

class MaskedLPIPS(nn.Module):
    def __init__(self, net='alex', crop_size=64, device='cuda'):
        super().__init__()
        self.net = lpips.LPIPS(net=net).to(device).eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.crop_size = crop_size

    def forward(self, x_adv, x_clean, boxes):
        # x in [0,1]; LPIPS expects [-1,1]
        if len(boxes) == 0:
            return x_adv.new_zeros(())
        losses = []
        for (x1, y1, x2, y2) in boxes:
            x1, y1 = max(int(x1), 0), max(int(y1), 0)
            x2, y2 = min(int(x2), x_adv.shape[-1]), min(int(y2), x_adv.shape[-2])
            if x2 <= x1 + 4 or y2 <= y1 + 4:  # skip degenerate boxes
                continue
            ca = 2 * x_adv[..., y1:y2, x1:x2] - 1
            cc = 2 * x_clean[..., y1:y2, x1:x2] - 1
            ca = F.interpolate(ca, size=(self.crop_size, self.crop_size), mode='bilinear', align_corners=False)
            cc = F.interpolate(cc, size=(self.crop_size, self.crop_size), mode='bilinear', align_corners=False)
            losses.append(self.net(ca, cc))
        return torch.stack(losses).mean() if losses else x_adv.new_zeros(())
```

**Trade-off:** LPIPS adds ~50ms/step on L4 (~10s/image at 200 steps). Worth it for perceptual fidelity. Keep the AlexNet backbone (faster than VGG) unless you have a specific reason.

### P1.2 — Temporal consistency loss

Two variants. Start with V1 (free); upgrade to V2 if flicker remains.

**V1 — Warm-start initialization (no flow):**

```python
delta_t.data.copy_(delta_{t-1}.data * 0.7)  # decay slightly
```

This alone removes 60–70% of flicker at zero compute cost.

**V2 — Optical-flow-warped consistency loss:**

```python
# Once per frame pair (cache):
flow = raft_small(x_clean[t-1], x_clean[t])  # B x 2 x H x W
flow_lat = F.avg_pool2d(flow, 8) / 8           # downsample to latent grid

def warp_latent(z, flow_lat):
    # Backward warp via grid_sample
    B, C, H, W = z.shape
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    grid = torch.stack([xx, yy], dim=-1).float().to(z.device)  # H x W x 2
    grid = grid + flow_lat.permute(0, 2, 3, 1)
    grid[..., 0] = 2 * grid[..., 0] / (W-1) - 1
    grid[..., 1] = 2 * grid[..., 1] / (H-1) - 1
    return F.grid_sample(z, grid, mode='bilinear', align_corners=True)

L_temp = lambda_temp * (delta_t - warp_latent(delta_{t-1}.detach(), flow_lat)).pow(2).mean()
```

Use `RAFT-small` (~150ms/pair on L4) or Farnebäck (free, noisier). Document the choice.

**Temporal stability metrics:**

- **Latent jitter:** `mean_t ||delta_t − delta_{t-1}||_2`
- **Pixel temporal LPIPS gap:** `mean_t [LPIPS(x_adv_t, x_adv_{t-1}) − LPIPS(x_clean_t, x_clean_{t-1})]` (subtracts baseline scene motion).

### P1.3 — Improved latent regularization

Replace `lambda_r * ||delta||_2^2` with a hybrid:

```python
L_reg = lambda_r * (
    delta.pow(2).mean()                                                  # L2
    + 0.1 * delta.abs().mean()                                           # L1 sparsity
    + 0.01 * (delta[..., 1:, :] - delta[..., :-1, :]).pow(2).mean()      # vertical smoothness
    + 0.01 * (delta[..., :, 1:] - delta[..., :, :-1]).pow(2).mean()      # horizontal smoothness
)
```

The smoothness term tends to remove high-frequency latent noise that decodes into perceptible artifacts.

### P1.4 — VAE fine-tuning on DETRAC

Highest-ROI item for stealth. Standard VAE objective on DETRAC frames:

```python
# Train ~10–20 epochs, LR=1e-5, batch_size=8 on L4
optim = torch.optim.AdamW(vae.parameters(), lr=1e-5, weight_decay=1e-4)
for x in detrac_loader:
    posterior = vae.encode(x).latent_dist
    z = posterior.sample() * 0.18215
    x_recon = vae.decode(z / 0.18215).sample
    loss = (
        1.0 * F.mse_loss(x_recon, x)
        + 0.1 * lpips_net(x_recon, x).mean()
        + 1e-6 * posterior.kl().mean()
    )
    loss.backward()
    optim.step()
```

**Expected gain:** PSNR_mask 20.6 → 28–30 dB at zero algorithmic cost.

**Honest caveat:** breaks direct comparability with prior latent-attack work that uses stock SD-VAE. Report both stock and fine-tuned numbers.

### P1.5 — Ablation grid

```yaml
# configs/ablations/
eps_z:        [0.10, 0.25, 0.50, 0.75, 1.00]   # 5 conditions
lambda_p:     [0.0,  0.01, 0.05, 0.10, 0.50]   # 5 conditions
lambda_temp:  [0.0,  0.001, 0.01, 0.1]         # 4 conditions
num_steps:    [25, 50, 100, 150, 200, 300]     # 6 conditions
num_restarts: [1, 3, 5]                        # 3 conditions
```

Don't run the full Cartesian product (2700 conditions). Run **one-factor-at-a-time** sweeps from a base config, plus a small interaction grid for `(eps_z × lambda_p)` since these are coupled. ~30 conditions × 100 frames is manageable on L4 over a weekend.

---

## 5. Implementation details — P2

### P2.1 — Black-box transferability

```python
# Generate adv images using YOLOv8n. Evaluate detection on:
target_models = ['yolov8s', 'yolov8m', 'yolov8l', 'yolov8x',
                 'rtdetr-l', 'faster_rcnn_r50_fpn']
```

Report per-target DFR. Useful even if numbers are weak — gives reviewers a sense of scope.

### P2.3 — Runtime profiling

```python
import torch
torch.cuda.synchronize()
t0 = time.perf_counter()
delta = run_attack(x, ...)
torch.cuda.synchronize()
elapsed = time.perf_counter() - t0
```

Report: per-image runtime, GPU memory peak, FLOPs (use `fvcore`).

---

## 6. Modular architecture

```
src/
├── attack/
│   ├── losses/
│   │   ├── detection.py       # vanishing class-confidence
│   │   ├── perceptual.py      # masked LPIPS
│   │   ├── temporal.py        # flow-warped latent consistency
│   │   └── regularization.py  # L2 + L1 + smoothness
│   ├── optimizers/
│   │   ├── latent_adam.py
│   │   └── latent_pgd.py      # projected sign-gradient variant
│   └── runner.py              # composes losses from config
├── baselines/
│   ├── fgsm.py
│   └── pgd_pixel.py
├── vae/
│   ├── wrapper.py
│   └── finetune.py
├── eval/
│   ├── metrics.py             # corrected DFR/ASR/mAP
│   ├── bootstrap.py           # CI computation
│   ├── pareto.py              # iso-budget curves
│   ├── transfer.py            # cross-detector evaluation
│   └── temporal.py            # latent jitter, temporal LPIPS
├── viz/
│   ├── qualitative.py
│   ├── heatmaps.py
│   ├── detection_overlay.py
│   └── video_export.py
configs/
├── base.yaml
├── ablations/
│   ├── eps_z.yaml
│   ├── lambda_p.yaml
│   ├── lambda_temp.yaml
│   └── num_steps.yaml
├── iso_budget.yaml
└── final.yaml
```

Each loss is a `nn.Module` returning a scalar; the runner composes them with config-driven weights. Ablations become config sweeps, not code duplication.

---

## 7. Honest trade-offs to report

| Improvement | Cost | Trade-off |
|---|---|---|
| LPIPS perceptual loss | +3–4× compute per step (~10s/image overhead) | Better perceptual fidelity but slower |
| Temporal loss (V2 with flow) | +150ms/pair (RAFT) | Requires optical flow; sensitive to flow errors |
| VAE fine-tuning | One-time 1–2h on L4 | Breaks comparability with stock SD-VAE prior work |
| Multi-restart attack | Linear in num_restarts | ASR up but compute up too |
| Strict metrics | None | Numbers will look worse than current report |

---

## 8. Suggested timeline

| Week | Focus | Output |
|---|---|---|
| 1 | P0 (metrics, iso-budget, eval set, CI) | Corrected baseline numbers |
| 2 | P1.1 (LPIPS) + P1.4 (VAE finetune) | Better stealth numbers |
| 3 | P1.2 (temporal) + P1.3 (regularization) | Video-stable attacks |
| 4 | P1.5 (ablations) | Full ablation table |
| 5 | P2 (transfer, runtime, statistics) | Rigor table |
| 6 | P3 (figures, videos) | Paper draft figures |

---

## 9. What to write up first

If you have to choose a single artifact to ship:
1. The corrected metrics + iso-budget Pareto curves (P0). Without these, no claim is defensible.
2. Then LPIPS + VAE-finetune (P1.1, P1.4) — biggest stealth wins.
3. Then temporal (P1.2) — differentiator from any single-image attack paper.

Everything else supports these three pillars.
