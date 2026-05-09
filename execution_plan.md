# Compute-Aware Execution Plan

**Constraint:** Colab Pro (~75 compute units, ~20h L4 GPU)
**Companion to:** `roadmap.md` (strategic vision)
**Purpose:** Strategically selected experiments that produce a publication-quality submission within budget

---

## 1. Guiding principles

1. **Dev/eval split.** Ablate on a small frozen dev set; touch the full eval set only for headline numbers and Pareto points.
2. **Reverse-engineer from deliverables.** Three figures + three tables — every experiment must feed one of them.
3. **Free things first, expensive things last.** If the code is broken, you discover it on cheap runs.
4. **Cache once, reuse forever.** Clean detections, latents, optical flow, LPIPS features.
5. **Reserve a 22-unit buffer.** Colab will drop you. Bugs will appear. Plan for it.

---

## 2. Data splits

| Split | Frames | Sequences | Used for |
|---|---|---|---|
| Dev | 30 | 1 (one with motion + occlusion variety) | All ablations, hyperparameter selection, debugging |
| Eval | ~100 | 3 (urban dense, highway sparse, occlusion-heavy) | Headline metrics + Pareto curves only |
| Sanity | 1 (img00001) | — | Single-image sanity checks before each batch run |

Do the dev/eval selection once, freeze it in `data/splits.json`, never touch it again.

---

## 3. Compute budget (75 units)

| # | Phase | Frames | Wall time | Units |
|---|---|---|---|---|
| 1 | VAE fine-tune on DETRAC | — | 2.0h | ~8 |
| 2 | Headline eval (3 attacks × 100 frames) | 100 | 2.5h | ~10 |
| 3 | Pareto curves (3 attacks × 3 budgets × 30) | 30 | 2.0h | ~8 |
| 4 | Ablation eps_z (3 levels × 30) | 30 | 1.5h | ~6 |
| 5 | Ablation lambda_p (3 levels × 30) | 30 | 1.5h | ~6 |
| 6 | Ablation lambda_temp (3 levels × 30) | 30 | 1.5h | ~6 |
| 7 | VAE stock vs. finetuned (2 × 30) | 30 | 1.0h | ~4 |
| 8 | Transferability YOLOv8s (30 frames) | 30 | 0.5h | ~2 |
| 9 | Figure/video generation | — | 1.0h | ~3 |
| 10 | **Buffer for reruns / debugging / Colab drops** | — | — | **~22** |
|   | **Total** | | | **~75** |

---

## 4. Tier S — Must do (defines the paper)

These are the experiments that, if any one is missing, the paper has a hole.

### S.1 Corrected metrics (no GPU)
Rewrite `evaluate.py` with strict definitions matching `method.tex`:
- Per-frame DFR, then averaged
- Strict ASR (every clean class disappears at IoU ≥ 0.5)
- mAP@0.5 drop using clean dets as pseudo-GT
- Bootstrap CI on every metric

### S.2 Iso-budget Pareto curves (~8 units)
Sweep each attack's budget. 3 points per attack, 30 frames.
- Latent: `eps_z ∈ {0.25, 0.50, 1.00}`
- PGD: `eps ∈ {4, 8, 12}/255`
- FGSM: same as PGD
Plot DFR vs. LPIPS. **Headline figure.**

### S.3 Masked LPIPS loss (in code, ~free to integrate)
Replace `lambda_p * masked_l2` with masked LPIPS via per-bbox crops at 64×64. AlexNet backbone only.

### S.4 Temporal consistency loss (~free engineering, ~6 units to ablate)
- V1: warm-start `delta_t = 0.7 * delta_{t-1}` — no compute cost.
- V2: Farnebäck flow + flow-warped latent regularizer. Skip RAFT (saves dependency + compute).
- Metric: latent jitter `mean_t ||delta_t − delta_{t-1}||_2`.

### S.5 VAE fine-tuning on DETRAC (~8 units)
20 epochs, LR=1e-5, batch 8, mixed reconstruction loss (MSE + LPIPS + small KL). One-time cost. **Single biggest stealth gain.**

### S.6 Headline run (~10 units)
Full eval set, best config from ablations. Three attacks, single budget per attack chosen for matched LPIPS. Bootstrap CI on all metrics.

---

## 5. Tier A — Strong rigor at low compute

### A.1 Three-axis ablation on dev set
Run only one-factor-at-a-time, 3 levels per axis:

| Axis | Levels | Cost (units) |
|---|---|---|
| eps_z | 0.25, 0.50, 1.00 | 6 |
| lambda_p | 0.0, 0.05, 0.50 | 6 |
| lambda_temp | 0.0, 0.01, 0.10 | 6 |

Default config is the middle point of each axis. **No interactions tested.** That's a deliberate choice; report it as a limitation.

### A.2 VAE stock vs. fine-tuned (~4 units)
2 conditions × 30 frames. One row in ablation table.

### A.3 Convergence curve (free, from headline run)
Log L_det per step on 5 representative dev frames. One small plot, no extra compute.

---

## 6. Tier B — Minimal-cost rigor

### B.1 Transferability to YOLOv8s only (~2 units)
30 frames. One row in transferability table. **No RT-DETR, no Faster R-CNN.** State explicitly that broader transfer is left for future work.

### B.2 Runtime profile (free)
Single image timing for each attack. One row in complexity table. No distributions needed.

---

## 7. What's been cut and why

| Cut | Reason |
|---|---|
| Multi-restart experiments | Linear compute cost; expected gain modest; not core contribution |
| 5-level ablation grids | 3 levels is enough to show monotonic trend |
| Cartesian ablation interactions | 9× cost; not standard in similar papers |
| Multi-backbone LPIPS | Adds compute, no scientific value |
| RAFT optical flow | Farnebäck is free and good enough for stability |
| RT-DETR / Faster R-CNN transfer | Each = 2+ units, attack often weakens, distracts from main claim |
| num_steps ablation | Convergence curve from 1 run replaces it |
| Mask-type ablation | Use union, document choice, move on |
| Step-size / lr sweeps | Not interesting once convergence shown |

If a reviewer asks for any of these, you can run them post-hoc with the buffer or in a revision cycle.

---

## 8. Engineering wins (free time savings)

| Trick | Savings |
|---|---|
| Reduce attack steps 200 → 120 (with early stop) | ~40% on latent |
| fp16 VAE forward pass | ~1.7× speedup |
| Cache `z_clean`, optical flow, clean detections to disk | Avoids recompute in ablations |
| Save adversarial latents instead of pixel images | Smaller files, re-decodable |
| Checkpoint per sequence to Google Drive | Recover from Colab timeouts |
| L4 GPU only (avoid A100) | A100 burns 3× units for ~1.3× speed |
| Precompute baselines on full budget grid once | FGSM/PGD compute amortized |

---

## 9. The paper's three figures and three tables

Reverse-engineer your experiments from this list. If an experiment doesn't feed one of these, don't run it.

### Figures

**F1 — Pareto curve (headline figure).**
DFR (y) vs. LPIPS (x). One curve per attack, 3 points each. Annotate operating points.

**F2 — Qualitative grid.**
4 frames × 4 columns: clean / adv-latent / adv-PGD / adv-FGSM, with diff row underneath. Frames chosen for diversity (urban dense, occlusion, small objects, low light).

**F3 — Temporal stability.**
Left panel: latent jitter over 30 consecutive frames, ours vs. PGD-per-frame.
Right panel: side-by-side adversarial video sample.

### Tables

**T1 — Headline metrics.**
| Method | DFR | ASR | mAP-drop | LPIPS | PSNR_mask | Runtime |
| Latent (ours) | ... ± CI | ... | ... | ... | ... | ... |
| PGD | ... | ... | ... | ... | ... | ... |
| FGSM | ... | ... | ... | ... | ... | ... |

**T2 — Ablation.**
| Variant | DFR | LPIPS | Note |
| Full (ours) | ... | ... | — |
| − LPIPS loss | ... | ... | masked L2 only |
| − Temporal | ... | ... | i.i.d. frames |
| Stock VAE | ... | ... | no DETRAC finetune |
| eps_z = 0.25 | ... | ... | low budget |
| eps_z = 1.00 | ... | ... | high budget |

**T3 — Transferability + complexity.**
| Source → Target | DFR | FLOPs | Peak GPU mem |
| YOLOv8n → YOLOv8n (white-box) | ... | ... | ... |
| YOLOv8n → YOLOv8s (black-box) | ... | — | — |

---

## 10. Execution order (safe-to-fail ordering)

| Day | Phase | GPU-h | Cumulative units |
|---|---|---|---|
| 1–2 | Implement corrected metrics, bootstrap CI, iso-budget runner (no GPU) | 0 | 0 |
| 3 | Curate dev/eval splits. Cache clean detections, latents, flow. Run baselines on full budget grid. | 4 | ~16 |
| 4 | Implement + integrate masked LPIPS. Sanity run on dev. | 2 | ~24 |
| 5 | Fine-tune VAE on DETRAC. | 2 | ~32 |
| 6 | Ablations: eps_z, lambda_p, VAE stock vs. finetuned (dev set). | 4 | ~48 |
| 7 | Implement temporal loss V1 + V2. Ablation on dev. | 3 | ~60 |
| 8 | **Headline eval on full eval set with best config.** | 3 | ~72 |
| 9 | Transferability YOLOv8s. | 1 | ~76 |
| 10 | Figures, videos, tables (CPU). | 0 | ~76 |

Note that day 8's headline run is **after** all hyperparameter selection. This keeps the eval set untouched until the final commit. The 22-unit buffer ends up partially eaten by reruns; whatever remains gives you slack at any phase.

---

## 11. Recovery plan if you blow the budget

If at day 6 you've burned 50 units instead of 32:
1. Drop transferability entirely. Save 2 units.
2. Cut Pareto curves from 3 to 2 points per attack. Save 4 units.
3. Cut headline eval set from 100 → 60 frames. Save 4 units.
4. Drop one ablation axis (lambda_temp first — least surprising direction). Save 6 units.

In order. Do not cut S-tier items. Do not cut the headline run.

---

## 12. Reproducibility hygiene

- All seeds fixed in config (`seed: 42` for numpy, torch, optim init)
- Single `configs/final.yaml` for the headline run; ablation configs inherit from it
- Log every metric to `results/<exp_name>/metrics.json` with timestamps
- Save model checkpoints, fine-tuned VAE, optical flow, attack latents to Drive
- A single notebook `notebooks/reproduce.ipynb` that loads cached outputs and regenerates all figures/tables in <10 minutes (CPU)

This last point is the difference between "took 2 weeks" and "fully reproducible". Make it work.
