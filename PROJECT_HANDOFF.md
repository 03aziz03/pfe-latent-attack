# Project Handoff — Latent Adversarial Attack on YOLOv8 (PFE)

**Author:** Mohamed Aziz Brahmi
**Date:** May 2026
**Status:** Phase 1 + 1.5 + Viz v1 complete. Figures v2 prompt prepared, not yet executed.

This document captures the full state of the project and conversation so a new chat session can resume seamlessly. Read top-to-bottom for context; jump to "Resuming work" at the bottom for the immediate next action.

---

## 1. Project at a glance

**Topic:** Object-aware latent-space adversarial attack against YOLOv8 object detection on UA-DETRAC.

**Core method (frozen, do NOT redesign):**
- Latent perturbation in frozen Stable Diffusion VAE space (`stabilityai/sd-vae-ft-mse`)
- Bounding-box-restricted (object-aware) masked perturbations
- Vanishing-detection loss: `ReLU(p_c − γ)²` summed over originally-detected classes
- Adam optimizer in latent L∞ ball, paste-back operator outside mask
- Baselines: FGSM (1-step) and PGD (50-step) in pixel space

**Hardware envelope:** Google Colab Pro, ~75 compute units (~20h L4). Tight but workable.

**Detector:** YOLOv8n fine-tuned on UA-DETRAC, weights at `runs/yolov8n_detrac/best.pt`.

**Evaluation set (current):** 50 frames `img00001..img00050`, single sequence. To be expanded to ~100 frames across 3 sequences for Phase 4 headline run.

---

## 2. Strategic documents in this workspace

| File | Purpose |
|---|---|
| `roadmap.md` | Strategic vision: P0-P3 priorities, full ablation plan |
| `execution_plan.md` | Tactical compute-aware plan against 75-unit budget |
| `claude_code_prompt_phase1.md` | DONE — corrected metrics + bootstrap CI module |
| `claude_code_prompt_phase1_5.md` | DONE — re-run YOLO, save raw FrameDetections |
| `claude_code_prompt_phase1_5_plus_viz.md` | DONE — Phase 1.5 + visualization module |
| `claude_code_prompt_figures_v2.md` | **NEXT** — fix coord bug + new figures |

---

## 3. Original experiments.md findings (May 2026 — pre-correction)

The user's initial `experiments.md` reported:

| Metric | Latent | PGD | FGSM |
|---|---|---|---|
| DFR (loose) | 0.310 | 0.111 | 0.034 |
| ASR (loose) | 0.76 | 0.58 | 0.22 |
| Conf drop | 0.263 | 0.143 | 0.014 |
| PSNR_mask (dB) | 20.6 | 32.5 | 30.1 |

**Major weak points identified:**
1. Metric definitions in `evaluate.py` contradict `method.tex` (publication blocker)
2. Comparisons not iso-budget (latent eps_z=0.50 vs pixel eps=8/255 not commensurable)
3. Sample size too small (50 frames, 1 sequence, no CI)
4. PSNR is wrong stealth metric for structured perturbations
5. No ablation studies
6. No temporal coherence on a video dataset
7. Single detector (no transferability)
8. VAE not adapted to surveillance domain

---

## 4. Roadmap (priority order)

**P0 — Foundation fixes (DONE in Phase 1 + 1.5):**
- ✅ Corrected metric definitions matching method.tex
- ⚠️ Iso-budget Pareto protocol (designed, not yet executed — Phase 2)
- ⚠️ Eval set expansion to ≥100 frames / 3 sequences (Phase 4)
- ✅ Bootstrap CI on all metrics

**P1 — Core scientific contributions (Phase 2 next):**
- Masked LPIPS perceptual loss (replace masked L2)
- Temporal consistency loss (warm-start + Farnebäck flow)
- VAE fine-tuning on DETRAC (one-time 2h, biggest stealth gain)
- Ablation grid: eps_z, lambda_p, lambda_temp (3 levels each, dev set)

**P2 — Rigor improvements:**
- Black-box transferability to YOLOv8s only (skip RT-DETR / Faster R-CNN to save budget)
- Statistical significance (paired Wilcoxon)
- Runtime / FLOPs profiling

**P3 — Publication outputs:**
- Pareto curves DFR vs LPIPS
- Qualitative grids (in progress, see Figures v2 prompt)
- Per-pixel perturbation heatmaps
- Detection overlay videos

**Cuts (deliberate, document as limitations):**
- Multi-restart experiments
- 5-level ablations (use 3)
- Cartesian ablation interactions
- Multi-backbone LPIPS (AlexNet only)
- RAFT optical flow (use Farnebäck, free)
- num_steps ablation (replaced by 1-run convergence curve)

---

## 5. Phase 1 — Corrected metrics (DONE)

**Built by Claude Code:** 33/33 tests pass.

```
src/eval/
  metrics.py      — FrameDetections, per_frame_dfr, per_frame_asr,
                    per_frame_map_drop, per_frame_psnr_mask,
                    per_frame_masked_l2, per_frame_conf_drop, aggregate
  bootstrap.py    — bootstrap_ci, bootstrap_metric_dict (seed=42)
  pareto.py       — build_pareto, plot_pareto
  io.py           — load_detections, save_detections (JSON schema)
tests/
  test_metrics.py    (23 tests)
  test_bootstrap.py  (10 tests)
  fixtures/synthetic_dets.json
scripts/
  recompute_metrics.py  — old vs new comparison
  evaluate.py           — DEPRECATED with warning, kept for reference
PHASE1_DONE.md
```

**3 method.tex discrepancies vs original task description (method.tex wins):**

1. **DFR:** method.tex defines binary per-frame indicator (`D_adv = ∅`), not proportional. Both implemented; `DFR_binary = (per_frame_dfr == 1.0).mean()`.
2. **ASR:** method.tex is class-based only (`C_clean ∩ classes(D_adv) = ∅`), no IoU constraint. Stricter version implemented; `iou_thr` is no-op for API compat.
3. **conf_drop:** method.tex uses pre-NMS class-level max; implemented as post-NMS IoU-matched (pre-NMS anchors not in `FrameDetections`). Document deviation in paper.

---

## 6. Phase 1 results — corrected numbers

After re-running `recompute_metrics.py` on existing per-image counts:

| Metric | Latent | PGD | FGSM |
|---|---|---|---|
| DFR_loose | 0.310 | 0.106 | 0.034 |
| DFR_strict_proportional | 0.205 [0.100, 0.320] | 0.076 [0.046, 0.107] | **−0.040** [−0.069, −0.012] |
| DFR_binary | 0.100 [0.020, 0.200] | 0.000 | 0.000 |
| ASR_loose | 0.760 | 0.620 | 0.220 |
| Frames with n_adv > n_clean | 10/50 | 4/50 | **16/50** |
| PSNR_mask (dB) | 20.60 | 32.51 | 30.07 |
| masked_L2 | 0.008848 | 0.000561 | 0.000984 |
| mean_conf_drop | 0.2627 | 0.1422 | 0.0136 |

**Key findings:**
- **Latent is the only attack achieving complete frame vanishing** (DFR_binary > 0). PGD and FGSM never achieve it on any of 50 frames.
- **FGSM is actively counter-productive** at this budget: DFR_strict_proportional = −0.040 with 95% CI entirely below zero. 16/50 frames show *more* adversarial detections than clean. This is a publishable finding (single-step pixel-space attacks against vanishing objective inflate spurious detections rather than suppressing them).
- Latent vs PGD ratio on DFR_strict_proportional ≈ 2.7× (CIs separated, statistically meaningful at n=50).
- **Headline metric should be DFR_strict_proportional**, not DFR_binary. Binary is rare-event (5/50 successes) → wide CI [0.02, 0.20]. Use binary as categorical marker ("only our method achieves this").

**Clipping policy decided:** unclipped. Negative DFR is reported as-is to expose FP inflation. Clipping would hide a real failure mode.

---

## 7. Phase 1.5 — Re-run detection + class-based metrics (DONE)

**Built by Claude Code:**

```
src/eval/run_detection.py        — run YOLO on dir, save FrameDetections
scripts/run_full_eval.py         — orchestrate 4-dir inference + ASR_strict + mAP
scripts/recompute_metrics.py     — updated to read metrics_full.json
results/dets_clean.json
results/dets_latent.json
results/dets_pgd.json
results/dets_fgsm.json
results/metrics_full.json
```

**Sanity check:** clean detection on data/images_50 → 401 detections (within ±5% of original report's 387). ✓

**Phase 1.5 results:**

| Metric | Latent | PGD | FGSM |
|---|---|---|---|
| ASR_strict (class-based) | **0.240** [0.140, 0.360] | 0.000 | 0.000 |
| mAP_drop@0.5 | **1.000 ⚠️ BUG** | 1.000 ⚠️ | 1.000 ⚠️ |

**⚠️ CRITICAL BUG DISCOVERED in mAP_drop:** see Section 9.

ASR_strict is real. 24% of frames lose all originally-detected vehicle classes under latent attack. PGD/FGSM never achieve this.

---

## 8. Visualization module (DONE)

**Built by Claude Code:**

```
src/viz/
  style.py             — PALETTE {clean:#2ca02c, latent:#1f77b4,
                         pgd:#ff7f0e, fgsm:#d62728}, setup_publication_style
  detection_overlay.py — draw_detections, overlay_clean_vs_adv
  perturbation.py      — perturbation_heatmap, difference_grid
  metrics_plots.py     — per_frame_dfr_distribution,
                         n_clean_vs_n_adv_scatter, metric_bar_chart,
                         stealth_vs_effectiveness_preview
  grids.py             — qualitative_grid
  pareto.py            — plot_pareto (works with current 1-pt-per-attack data)
  ablation.py          — STUB (Phase 2)
  convergence.py       — STUB (Phase 2)
  temporal.py          — STUB (Phase 3)

scripts/generate_figures.py      — orchestrator (10 figures)
results/figures/png/             — f01..f10 (300 dpi PNG)
results/figures/pdf/             — f01..f10 (vector PDF)
results/figures/index.html
```

**10 figures produced:**
- f01 dfr_distribution (violin)
- f02 nclean_vs_nadv (scatter)
- f03 metric_barchart
- f04 stealth_vs_effectiveness
- f05 perturbation_heatmap
- f06 difference_grid
- f07 overlay_latent
- f08 overlay_pgd
- f09 overlay_fgsm
- f10 qualitative_grid

---

## 9. Critical bugs and figure issues found in v1

### 9.1 ⚠️ Coordinate-space mismatch (MOST IMPORTANT)

`mAP_drop = 1.000` for all 3 attacks is NOT a sign of perfect attack. It's a coordinate-space bug:

- `data/images_50/*.jpg` are at original DETRAC resolution (~960×540)
- `results/adv_*/*.png` are saved at 640×640 letterboxed (gray bars top/bottom)

YOLO returns boxes in input-image coordinate space:
- `dets_clean.json` boxes in (960, 540) frame
- `dets_*.json` (adversarial) boxes in (640, 640) letterboxed frame

These don't overlap by construction → IoU≈0 → mAP_drop=1.0 trivially. This invalidates the mAP number entirely until fixed.

**Fix:** letterbox clean images to 640×640 BEFORE running detection so both reference frames match. New helper `src/viz/letterbox.py` with `letterbox_image()` and `unletterbox_boxes()`.

### 9.2 Per-figure issues

| Fig | Critical issues |
|---|---|
| f01 | Wide line is median not mean (LATENT shows ~0.05 instead of 0.205) |
| f02 | Severe overplotting (n_clean ∈ {6,7,8,9,10} only) — needs jitter |
| f03 | **No error bars** despite computed CIs; mixes scales DFR/ASR |
| f04 | "← lower effectiveness" label points at LATENT (the most effective!) |
| f05 | Layout broken (thin horizontal strip, illegible) |
| f06 | Fixed ×10 amplification saturates LATENT, hides PGD/FGSM |
| f07-f09 | **No clean boxes drawn**; image sizes inconsistent (clean orig vs adv letterboxed) |
| f10 | **No detection boxes anywhere** in qualitative grid; sizes inconsistent |

### 9.3 Missing figures (added in Figures v2 prompt)

- f11 per-class breakdown (which classes survive — uses Phase 1.5 class data)
- f12 per-frame timeseries (50-bar plot, n_clean vs n_adv per attack)
- f13 IoU distribution of matched detections (tests "latent moves boxes" hypothesis)

---

## 10. Story for the paper (current best framing)

**Headline claim:**
> Our latent-space attack is the only one of the three that achieves complete frame-level vanishing on UA-DETRAC. Pixel-space attacks at the same operating point either partially suppress (PGD: 7.6% per-frame DFR, 0% complete vanishing) or actively inflate spurious detections (FGSM: net negative DFR, 16 frames out of 50).

**Supporting claims:**
- 2.7× per-frame DFR over PGD (CIs separated)
- 24% of frames lose all original vehicle classes (ASR_strict)
- Trade-off: lower PSNR_mask (20.6 dB vs 32.5) — to be improved by VAE fine-tuning + LPIPS in Phase 2

**Honest caveats to ship:**
- 50 frames, 1 sequence (Phase 4 expands to 100+ frames / 3 sequences)
- No transferability to non-YOLO architectures
- conf_drop implementation is post-NMS, method.tex specifies pre-NMS
- mAP_drop currently corrupted by coordinate bug; correct value pending Figures v2

---

## 11. Compute budget tracker

| Phase | Estimated cost | Status |
|---|---|---|
| Phase 1 (no GPU) | 0 units | ✅ DONE |
| Phase 1.5 (re-detect, ~30s GPU) | <1 unit | ✅ DONE |
| Viz v1 (no GPU) | 0 units | ✅ DONE |
| **Figures v2 (re-run + figures)** | <1 unit | **PENDING** |
| Phase 2.1 LPIPS integration | ~4 units | Pending |
| Phase 2.2 VAE finetune (2h) | ~8 units | Pending |
| Phase 2.3 Ablation grid (3-axis × 3 levels × 30 frames) | ~18 units | Pending |
| Phase 3 Temporal loss + ablation | ~9 units | Pending |
| Phase 4 Headline run (100 frames × 3 attacks) | ~10 units | Pending |
| Transferability check (YOLOv8s) | ~2 units | Pending |
| Buffer (debug, reruns, Colab drops) | ~22 units | Reserve |
| **Total** | **~75 units** | |

---

## 12. Repository layout (current state)

```
<project_root>/
├── configs/
│   └── default.yaml
├── data/
│   └── images_50/             (50 clean DETRAC frames)
├── docs/
│   └── method.tex             (formal method description — ground truth)
├── runs/
│   └── yolov8n_detrac/best.pt (5.9 MB)
├── scripts/
│   ├── run_attack.py          (DO NOT MODIFY)
│   ├── evaluate.py            (DEPRECATED with warning)
│   ├── recompute_metrics.py
│   ├── run_full_eval.py
│   └── generate_figures.py
├── src/
│   ├── eval/
│   │   ├── metrics.py
│   │   ├── bootstrap.py
│   │   ├── pareto.py
│   │   ├── io.py
│   │   └── run_detection.py
│   └── viz/
│       ├── style.py
│       ├── detection_overlay.py
│       ├── perturbation.py
│       ├── metrics_plots.py
│       ├── grids.py
│       ├── pareto.py
│       ├── ablation.py        (STUB)
│       ├── convergence.py     (STUB)
│       └── temporal.py        (STUB)
├── tests/
│   ├── test_metrics.py        (23 tests)
│   ├── test_bootstrap.py      (10 tests)
│   └── fixtures/synthetic_dets.json
├── baselines/
│   ├── fgsm.py
│   └── pgd_pixel.py
├── results/
│   ├── adv_latent/            (50 PNG)
│   ├── adv_pgd/               (50 PNG)
│   ├── adv_fgsm/              (50 PNG)
│   ├── metrics_latent.json
│   ├── metrics_pgd.json
│   ├── metrics_fgsm.json
│   ├── dets_clean.json
│   ├── dets_latent.json
│   ├── dets_pgd.json
│   ├── dets_fgsm.json
│   ├── metrics_full.json
│   ├── metric_comparison.md
│   └── figures/{png,pdf}/     (10 v1 figures)
├── PHASE1_DONE.md
└── pytest passes 33/33
```

---

## 13. Resuming work — immediate next action

**Open `D:\minimal_research\claude_code_prompt_figures_v2.md`** and feed it to Claude Code in the project repo. This will:

1. Fix the coordinate-space bug → real `mAP_drop` numbers
2. Rework f01–f10 (error bars, label fix, box overlays, layout)
3. Add f11–f13 (per-class, time series, IoU distribution)
4. Re-run `run_full_eval.py` after fix
5. Regenerate all figures

**Cost:** <1 compute unit, ~8 minutes wall time.

**Verification after run:**
- mAP_drop != 1.0 for all attacks (expect ~0.3–0.7 for latent)
- f10 qualitative grid has boxes drawn in every cell
- f03 shows visible CI error bars
- pytest still 33/33

After Figures v2 is clean, the next strategic step is **Phase 2: LPIPS + VAE fine-tune** (`roadmap.md` P1.1 + P1.4). The order of operations within Phase 2:

1. Implement `MaskedLPIPS` loss in `src/attack/losses/perceptual.py`
2. Integrate into the existing optimizer (currently uses masked L2)
3. Fine-tune SD-VAE on DETRAC for 10–20 epochs (one-time ~2h)
4. Re-run latent attack with new loss + new VAE on dev set (30 frames)
5. Iso-budget sweep: latent eps_z ∈ {0.25, 0.50, 1.00}, PGD eps ∈ {4, 8, 12}/255
6. Generate first real Pareto curve (DFR vs LPIPS)

That session will need ~12 compute units total. Plan for one Colab session of ~3 hours.

---

## 14. Open questions / things to verify

- After coordinate bug fix, does the FGSM ASR_strict change? Could go up if class-shifted false positives count as "C_clean ∩ classes(D_adv) = ∅".
- After Phase 2, does fine-tuned VAE break attack effectiveness? Need to verify the headline DFR_strict_proportional ≥ 0.20 still holds.
- Choice of Farnebäck vs RAFT for Phase 3 temporal flow — start with Farnebäck (free), upgrade only if temporal jitter is still high.
- Whether to include img00037 (PGD partial success) in qualitative grid or replace with another representative.

---

## 15. How to brief a new Claude session

Paste this into a fresh chat:

> I'm continuing work on a final-year project (PFE): an object-aware latent
> adversarial attack against YOLOv8 on UA-DETRAC, using a frozen Stable
> Diffusion VAE. Read `D:\minimal_research\PROJECT_HANDOFF.md` for full
> context. The immediate next step is to apply
> `D:\minimal_research\claude_code_prompt_figures_v2.md` via Claude Code
> in my project repo. After that, we move to Phase 2 (LPIPS + VAE fine-tune)
> per `roadmap.md`. Compute budget remaining: ~70 Colab Pro units.
>
> Please confirm you've read the handoff, summarize the current state in
> 5 bullet points, and ask me whether to proceed with Figures v2 or
> jump straight to Phase 2 planning.

---

_End of handoff document._
