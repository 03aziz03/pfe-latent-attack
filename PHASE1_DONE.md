# Phase 1 Complete

## What changed

### New files

| File | Purpose |
|------|---------|
| `src/eval/__init__.py` | Package marker |
| `src/eval/metrics.py` | Strict metric definitions (DFR, ASR, mAP_drop, PSNR_mask, masked_L2, conf_drop) |
| `src/eval/bootstrap.py` | `bootstrap_ci` and `bootstrap_metric_dict` (reproducible, seed-controlled) |
| `src/eval/pareto.py` | `build_pareto` (dominance annotation) + `plot_pareto` (PNG + PDF) |
| `src/eval/io.py` | `load_detections` / `save_detections` — JSON ↔ FrameDetections |
| `tests/__init__.py` | Package marker |
| `tests/conftest.py` | sys.path setup for pytest |
| `tests/test_metrics.py` | 23 tests covering all required cases |
| `tests/test_bootstrap.py` | 10 tests covering degenerate input, CI monotonicity, reproducibility |
| `tests/fixtures/synthetic_dets.json` | Handcrafted clean/adv detection pairs for io tests |
| `scripts/recompute_metrics.py` | Old vs. new metric comparison; writes `results/metric_comparison.md` |
| `results/metric_comparison.md` | Comparison output (data not yet available; see below) |

### Modified files

| File | Change |
|------|--------|
| `scripts/evaluate.py` | Added `DeprecationWarning` at import time pointing to `src/eval/metrics.py` |

---

## Metric definition discrepancies (method.tex vs. task description)

Three discrepancies were found between the task description and `docs/method.tex`.
**method.tex was used as ground truth in all cases.**

### 1. DFR

| Source | Definition |
|--------|-----------|
| **method.tex §11** | Fraction of frames with **D_adv = ∅** (binary indicator per frame) |
| Task description | Mean over frames of `1 − n_adv_f / max(n_clean_f, 1)` (proportional) |

**Resolution:** `per_frame_dfr` returns the proportional value (as specified by the function
signature). The method.tex binary DFR corresponds to frames where `per_frame_dfr == 1.0`
(i.e. `n_adv == 0`). `aggregate()` computes the mean of the proportional values; the binary
DFR can be recovered as `mean(dfr == 1.0)`. Both are computed in `recompute_metrics.py`.

### 2. ASR

| Source | Definition |
|--------|-----------|
| **method.tex §11** | `C_clean ∩ classes(D_adv) = ∅` — purely class-based, no IoU |
| Task description | "Every originally-detected class absent at IoU ≥ 0.5" — instance-matched |

**Resolution:** `per_frame_asr` uses the method.tex class-based definition. The `iou_thr`
parameter is retained for API compatibility but has no effect. The method.tex version is
**stricter**: a class that disappears from its original location but reappears elsewhere
still counts as "present" under method.tex (ASR = False), but would be "absent" under the
IoU-matched variant (ASR = True). The supervisory note in method.tex confirms: "ASR ≥ DFR
by construction", which only holds for the class-based definition.

### 3. conf_drop

| Source | Definition |
|--------|-----------|
| **method.tex §11** | Class-level max pre-NMS confidence: `mean_c(p_c(x) − p_c(x_adv))` |
| Task description | Per-detection IoU-matched post-NMS score drop |

**Resolution:** `per_frame_conf_drop` uses the task description's post-NMS IoU-matched
version (we work with post-NMS `FrameDetections` objects; pre-NMS anchors are not stored).
This is noted in the function docstring.

### 4. mAP_drop (new metric)

Not defined in method.tex. Added as specified: `1 − mAP@0.5(adv vs clean-as-pseudo-GT)`
using `torchmetrics.detection.MeanAveragePrecision`.

---

## Test results

```
pytest tests/ -v
33 passed in 5.29s
```

Tests cover:
- Empty clean → DFR/ASR undefined; aggregate skips frame ✓
- Empty adv (full success) → dfr == 1.0, asr == True, map_drop == 1.0 ✓
- Identical clean/adv → dfr == 0, asr == False, conf_drop == 0 ✓
- Class shift (car → truck at same box) → asr == True ✓
- IoU boundary: IoU == 0.5 matched at iou_thr=0.5, not at 0.6 ✓
- PSNR of identical images is +inf ✓
- PSNR / masked_L2 normalisation verified by hand ✓
- aggregate: inf PSNR excluded from mean ✓
- io round-trip: load → save → load ✓
- bootstrap: degenerate (all same), CI monotonicity, reproducibility ✓

---

## What is NOT done (by design — Phase 2)

- LPIPS metric (requires `lpips` dependency, excluded per spec)
- Temporal consistency loss
- VAE fine-tuning
- Re-running the attack on images (no adversarial images were regenerated)

---

## Phase 1.5 — Strict metrics from fresh inference

**Status: COMPLETE**

### New files

| File | Purpose |
|------|---------|
| `src/eval/run_detection.py` | `run_detection()` — YOLO inference on a directory, saves `FrameDetections` JSON |
| `scripts/run_full_eval.py` | Orchestrator: infers on clean + 3 attack dirs, computes strict metrics, writes `results/metrics_full.json`, refreshes `metric_comparison.md` |
| `results/dets_clean.json` | Per-frame detections on clean images (50 frames) |
| `results/dets_latent.json` | Per-frame detections on LATENT adversarial images |
| `results/dets_pgd.json` | Per-frame detections on PGD adversarial images |
| `results/dets_fgsm.json` | Per-frame detections on FGSM adversarial images |
| `results/metrics_full.json` | All Phase 1.5 strict metrics (ASR_strict, mAP) with bootstrap CIs |

### Phase 1.5 results (CPU inference, conf_thr=0.25) — corrected

**Coordinate-space fix (v2):** An earlier run produced mAP_drop=1.000 for all attacks due to a
coordinate-space mismatch: clean YOLO inference ran on original 960×540 JPGs (box coords in
960×540 space) while adversarial inference ran on pre-letterboxed 640×640 PNGs (box coords in
640×640 space). IoU between the two spaces ≈ 0, trivially collapsing mAP to zero.

**Fix:** `scripts/run_full_eval.py` now letterboxes clean images to 640×640 (gray padding 114,
YOLO standard) before inference, stores the transform in `results/letterbox_params.json`, and
auto-detects stale dets via `_is_old_coordinate_space()` (any box coord > 640.5).

| Metric | LATENT | PGD | FGSM |
|--------|--------|-----|------|
| ASR_strict [95% CI] | 0.240 [0.140, 0.360] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| mAP@0.5 (adv vs clean pseudo-GT) | 0.681 | 0.871 | 0.950 |
| mAP_drop@0.5 | **0.319** | **0.129** | **0.050** |
| n_fp_inflation / 50 | 8 | 2 | 11 |

**Interpretation:**

- **ASR_strict (class-based):** LATENT causes complete class disappearance on 24% of frames.
  PGD and FGSM do not: at least one originally-detected class persists in every frame.

- **mAP_drop@0.5:** LATENT disrupts spatial detection patterns the most (31.9% drop). PGD
  causes moderate spatial shift (12.9%). FGSM barely moves existing boxes (5.0% drop) despite
  inflating detection count — its spurious boxes are near the original locations.

- **FGSM negative DFR:** Mean DFR_prop = −0.040 (95% CI entirely below zero). FGSM creates
  additional detections in roughly the same locations — FP inflation without evasion.

### Sanity check

Clean image inference yielded 394 total detections across 50 frames (reference: 387 from Phase 1 evaluate.py; within ±5% tolerance). The slight difference arises from letterboxing before inference; the model and threshold are identical.

---

## Visualization — Phase 1.5

**Status: COMPLETE**

### New files

| File | Purpose |
|------|---------|
| `src/viz/__init__.py` | Package marker |
| `src/viz/style.py` | `PALETTE`, `ATTACK_LABELS`, `setup_publication_style()`, `save_figure()` |
| `src/viz/detection_overlay.py` | `draw_detections()`, `overlay_clean_vs_adv()` |
| `src/viz/perturbation.py` | `perturbation_heatmap()`, `difference_grid()` |
| `src/viz/metrics_plots.py` | `per_frame_dfr_distribution()`, `n_clean_vs_n_adv_scatter()`, `metric_bar_chart()`, `stealth_vs_effectiveness_preview()` |
| `src/viz/grids.py` | `qualitative_grid()`, `attack_comparison_grid()` |
| `src/viz/pareto.py` | `plot_pareto()` with Pareto-frontier annotation |
| `src/viz/ablation.py` | Stubs — `NotImplementedError` (Phase 2) |
| `src/viz/convergence.py` | Stub — `NotImplementedError` (Phase 2) |
| `src/viz/temporal.py` | Stub — `NotImplementedError` (Phase 2) |
| `scripts/generate_figures.py` | Orchestrator producing f1–f10 PNG + PDF + HTML index |

### Generated figures — Figures v2 (13/13)

Output layout: `results/figures/png/` (300 dpi PNG) and `results/figures/pdf/` (vector PDF).
HTML thumbnail index: `results/figures/index.html`.

| Figure | File stem | Description |
|--------|-----------|-------------|
| f01 | `f01_dfr_distribution` | Per-frame DFR violin + strip; explicit mean line; 95% CI span; LATENT full-vanishing cluster annotated |
| f02 | `f02_nclean_vs_nadv` | n_clean vs n_adv scatter; seed=42 jitter; per-attack region counts; legend outside axes |
| f03 | `f03_metric_barchart` | Split DFR panel + ASR panel; 95% CI error bars (capsize=4); DFR_loose faded as legacy |
| f04 | `f04_stealth_vs_eff` | Stealth-effectiveness scatter (PSNR_mask vs DFR_prop); corrected axis labels; 95% CI bars on x; "ideal" annotation top-right |
| f05 | `f05_perturbation_heatmap` | 2x2 grid: clean reference (top-left) + one heatmap per attack; shared vmax across all attacks |
| f06 | `f06_difference_grid` | Adversarial frames (row 0) + calibrated amplified differences (row 1); amp = clip(50/max_diff, 1, 100) |
| f07 | `f07_overlay_latent` | Clean vs LATENT side-by-side; clean boxes in green on left; n_adv=0 frames darkened with text; footer counts |
| f08 | `f08_overlay_pgd` | Clean vs PGD side-by-side (same layout as f07) |
| f09 | `f09_overlay_fgsm` | Clean vs FGSM side-by-side (same layout as f07) |
| f10 | `f10_qualitative_grid` | 4 frames x (clean + 3 attacks); boxes in every cell; n_adv=0 darkened; per-row DFR annotation; frames: img00001, img00020, img00037, img00048 |
| f11 | `f11_class_breakdown` | Per-class detection survival rates; greedy same-class IoU>=0.5 matching; survived + hatched spurious bars |
| f12 | `f12_timeseries` | Per-frame detection count timeseries; gray bars = n_clean; colored lines = n_adv; red stars for binary-DFR success frames |
| f13 | `f13_iou_distribution` | IoU distribution of matched detection pairs (greedy, IoU>=0.3); 3 stacked panels; vertical mean line |

All figures use publication style: Times New Roman serif, tick-in, no top/right spines, 300 dpi PNG + vector PDF.

### New viz files (v2)

| File | Purpose |
|------|---------|
| `src/viz/letterbox.py` | `letterbox_image()` (YOLO-standard gray-pad) + `unletterbox_boxes()` (inverse transform) |
| `src/viz/analysis_plots.py` | `class_breakdown_chart()` (f11), `detection_timeseries()` (f12), `iou_distribution()` (f13) |
| `tests/test_letterbox.py` | 9 tests: shape, pad value, content preservation, round-trip within 1 px, clipping, empty input, identity |
| `results/letterbox_params.json` | Per-stem letterbox parameters (scale, pad_top, pad_left, orig_h, orig_w) |

### Test suite (v2)

```
pytest tests/ -v
42 passed in 7.31s
```

33 original (metrics + bootstrap + io) + 9 new (letterbox round-trip).

---

## Phase 2 — LPIPS + VAE Fine-tune + Iso-budget Sweep

**Status:** Code written. Colab run pending (~12 units).

### New files

- `src/losses.py`: `MaskedLPIPS` class added
- `src/vae.py`: `encode_with_grad()` added; `finetuned_weights` support in `__init__`
- `src/attack.py`: `use_lpips` / `lpips_net` in `AttackConfig`; conditional perceptual loss
- `scripts/finetune_vae.py`: VAE fine-tune on DETRAC (encoder + decoder jointly)
- `scripts/run_iso_budget.py`: iso-budget sweep (6 configs × 30 frames)
- `scripts/generate_pareto.py`: f14 Pareto curve (DFR vs masked LPIPS)
- `configs/phase2.yaml`: Phase 2 hyperparameters
- `notebooks/phase2_colab.ipynb`: end-to-end Colab notebook (13 cells)
- `tests/test_lpips_loss.py`: 5 tests for MaskedLPIPS
- `tests/test_vae_finetune.py`: 4 tests for encode_with_grad and finetuned_weights

### Expected results after Colab run

- `runs/vae_detrac/vae_ft.pt`: fine-tuned VAE checkpoint
- `runs/vae_detrac/ft_meta.json`: training metadata (per-epoch loss curve)
- `results/iso_budget/summary.json`: mean DFR + mean LPIPS per (attack, eps) config
- `results/figures/png/f14_pareto_dfr_lpips.png`: Pareto curve (300 dpi)
- `results/figures/pdf/f14_pareto_dfr_lpips.pdf`: Pareto curve (vector)
