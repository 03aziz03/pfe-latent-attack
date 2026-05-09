# Claude Code — Phase 1.5 + Visualization Scaffolding

This prompt does two things in one Claude Code session:
1. **Phase 1.5**: re-run YOLOv8 on existing images, save raw `FrameDetections`, compute `ASR_strict` and `mAP_drop` (~30s GPU).
2. **Visualization module**: build `src/viz/` with all immediately-usable figures + scaffolding for future phases. No GPU.

---

## PROMPT

```
Two-part task: complete the strict metrics and build the visualization module.

# Context
Phase 1 + recompute_metrics.py are done. Strict DFR variants are populated:
- Latent: DFR_prop=0.205 [0.10, 0.32], DFR_bin=0.10, FP inflation 10/50
- PGD:    DFR_prop=0.076, DFR_bin=0.00, FP inflation 4/50
- FGSM:   DFR_prop=-0.040 (negative!), DFR_bin=0.00, FP inflation 16/50
ASR_strict and mAP_drop are still TODO because raw FrameDetections weren't saved.

This session: capture the missing metrics + build a complete viz module.

============================================================
PART A — Re-run YOLO detection (Phase 1.5)
============================================================

## A.1 New file: src/eval/run_detection.py

  def run_detection(
      image_dir: Path,
      model_path: Path,
      output_path: Path,
      conf_thr: float = 0.25,
      iou_nms: float = 0.45,
      img_size: int = 640,
      device: str = "cuda",
  ) -> dict[str, FrameDetections]:
      """Run YOLOv8 inference on every image in image_dir, save raw
      FrameDetections via src/eval/io.save_detections.

      Use ultralytics.YOLO. Letterbox to img_size=640 to match attack
      preprocessing. Frame ID = image stem. tqdm progress bar.

      Convert ultralytics Results to FrameDetections:
          boxes   = result.boxes.xyxy.cpu()   # (N, 4)
          scores  = result.boxes.conf.cpu()   # (N,)
          classes = result.boxes.cls.cpu().long()
      """

## A.2 New file: scripts/run_full_eval.py

CLI script:
1. run_detection on data/images_50  → results/dets_clean.json
2. run_detection on results/adv_latent → results/dets_latent.json
3. run_detection on results/adv_pgd    → results/dets_pgd.json
4. run_detection on results/adv_fgsm   → results/dets_fgsm.json
5. For each attack, compute per-frame ASR_strict (class-based, no IoU per
   method.tex §11) and dataset-level mAP@0.5 + mAP@0.5:0.95 via
   torchmetrics.detection.MeanAveragePrecision (clean as pseudo-GT).
6. Bootstrap CI (n_boot=1000, seed=42) on ASR_strict.
7. Update results/metrics_full.json with the new fields.
8. Re-run scripts/recompute_metrics.py at the end so the comparison table
   gets the new ASR_strict and mAP_drop rows filled.

Sanity check: assert that running detection on data/images_50 yields ~387
total detections (±5%). If not, abort with a clear error — preprocessing
has drifted from the original report.

## A.3 Update scripts/recompute_metrics.py

If results/dets_*.json files exist, replace the "TODO Phase 1.5" cells
with the real ASR_strict (with CI) and mAP_drop values.

## A.4 Special FGSM watch

FGSM has DFR_strict_prop = -0.040 (creates more detections than it removes).
Its ASR_strict could be either:
- High (~0.30+): if the inflated detections are of DIFFERENT classes than
  the originals, then by the class-disjoint rule of method.tex §11 the
  attack "succeeds" on those frames.
- Low (~0.05): if the inflated detections share classes with the originals.

Whatever it is, REPORT IT. This is a scientifically interesting finding,
not a bug. Add a one-paragraph note in metric_comparison.md interpreting
the result for FGSM specifically.

============================================================
PART B — Visualization module (src/viz/)
============================================================

## B.1 Directory structure

  src/viz/
    __init__.py
    style.py             # NEW — publication style + color palette
    detection_overlay.py # NEW — draw clean vs adv boxes on images
    perturbation.py      # NEW — diff heatmaps, mask visualizations
    metrics_plots.py     # NEW — bar charts, distributions, scatters
    grids.py             # NEW — multi-attack qualitative grids
    pareto.py            # NEW — Pareto curves (scaffolding for future)
    ablation.py          # NEW — ablation curve scaffolding (Phase 2)
    convergence.py       # NEW — convergence curve scaffolding (Phase 2)
    temporal.py          # NEW — temporal stability scaffolding (Phase 3)

  scripts/
    generate_figures.py  # NEW — orchestrator, calls every available viz

  results/
    figures/             # NEW — auto-created output dir
      png/
      pdf/

## B.2 src/viz/style.py

  PALETTE = {
      "clean":  "#2ca02c",   # green
      "latent": "#1f77b4",   # blue
      "pgd":    "#ff7f0e",   # orange
      "fgsm":   "#d62728",   # red
  }

  ATTACK_LABELS = {
      "clean":  "Clean",
      "latent": "Latent (ours)",
      "pgd":    "PGD",
      "fgsm":   "FGSM",
  }

  def setup_publication_style() -> None:
      """Configure matplotlib for paper-quality figures.

      - Serif font (cm/cmr10), TeX-friendly fallback if no LaTeX
      - Tick direction in
      - Linewidth 1.0
      - Figsize default 5x3.5 inches
      - Save 300 dpi PNG, vector PDF
      - No top/right spines by default
      """

  def save_figure(fig, name: str, out_dir: Path) -> None:
      """Save fig as PNG and PDF in out_dir/png and out_dir/pdf.
      Use bbox_inches='tight', pad_inches=0.05."""

## B.3 src/viz/detection_overlay.py

  def draw_detections(
      image: np.ndarray,             # H x W x 3 uint8
      boxes: torch.Tensor,           # (N, 4) xyxy
      classes: torch.Tensor,         # (N,)
      scores: torch.Tensor,          # (N,)
      color: tuple[int, int, int],
      class_names: list[str] | None = None,
  ) -> np.ndarray:
      """Draw boxes with class:score labels on image. Use cv2.rectangle
      and cv2.putText. Return a copy."""

  def overlay_clean_vs_adv(
      clean_image_path: Path,
      adv_image_path: Path,
      clean_dets: FrameDetections,
      adv_dets: FrameDetections,
      class_names: list[str],
      out_path: Path,
      title: str | None = None,
  ) -> None:
      """Side-by-side: [clean image + clean boxes (green)] | [adv image + adv boxes (red)].
      Save with style.save_figure."""

## B.4 src/viz/perturbation.py

  def perturbation_heatmap(
      clean_path: Path,
      adv_path: Path,
      mask: np.ndarray | None,    # boolean H x W; None = full image
      out_path: Path,
      log_scale: bool = True,
  ) -> None:
      """Per-pixel |x_adv - x_clean| (max over channels), log-scaled,
      colormap='inferno'. Optionally restrict to mask with grey-out outside."""

  def difference_grid(
      clean_path: Path,
      adv_paths: dict[str, Path],   # {attack_name: path}
      out_path: Path,
      log_scale: bool = True,
  ) -> None:
      """One row of heatmaps, one per attack, with shared colorbar.
      Useful for showing which attack creates the loudest perturbation."""

## B.5 src/viz/metrics_plots.py

This is the biggest module. Implement all of these:

  def per_frame_dfr_distribution(
      per_image_data: dict[str, list[dict]],   # {attack: per_image list from JSONs}
      out_path: Path,
      include_loose: bool = False,
  ) -> None:
      """Strip plot (or violin if you prefer) of per-frame DFR_strict_prop
      values, one column per attack. Mark mean with a wide horizontal line.
      Annotate the n_adv > n_clean count below each column.
      Use PALETTE colors. Y-axis: 'Per-frame DFR (unclipped)'. Add zero line."""

  def n_clean_vs_n_adv_scatter(
      per_image_data: dict[str, list[dict]],
      out_path: Path,
  ) -> None:
      """Scatter with x = n_clean_f, y = n_adv_f. One color per attack.
      Plot diagonal y=x. Points BELOW diagonal = suppression, ABOVE = inflation.
      Annotate quadrant counts. Highlight the (n_clean=7, n_adv=0) cluster
      for the latent attack — these are the binary-DFR successes."""

  def metric_bar_chart(
      results: dict[str, dict],   # {attack: {metric: (mean, lo, hi)}}
      metrics: list[str],         # which metric keys to plot
      out_path: Path,
      ylabel: str = "Value",
      title: str | None = None,
  ) -> None:
      """Grouped bar chart: x = metric name, color = attack, error bars = CI.
      Useful for the headline metrics figure."""

  def stealth_vs_effectiveness_preview(
      results: dict[str, dict],   # {attack: {DFR: ..., PSNR_mask: ..., LPIPS: ...}}
      out_path: Path,
      x_metric: str = "PSNR_mask",
      y_metric: str = "DFR_strict_prop",
  ) -> None:
      """Single point per attack on a (stealth, effectiveness) plane.
      One point per attack with attack-color and label. This is a PREVIEW —
      the full Pareto with multiple budget levels comes from src/viz/pareto.py."""

## B.6 src/viz/grids.py

  def qualitative_grid(
      frame_ids: list[str],         # e.g., ['img00001', 'img00020', 'img00037', 'img00048']
      clean_dir: Path,
      adv_dirs: dict[str, Path],    # {attack: dir of adv images}
      out_path: Path,
      include_diff_row: bool = True,
      max_pixel_value: int = 255,
  ) -> None:
      """N_frames rows × (1 + N_attacks) columns. Optionally add a row of
      diff heatmaps below. Tight layout. Column titles. Row labels (frame ID)."""

## B.7 src/viz/pareto.py (scaffolding — works with current data + future)

  def plot_pareto(
      runs: list[dict],   # [{'name': str, 'budget': float, 'DFR': ..., 'LPIPS': ..., ...}]
      out_path: Path,
      x_metric: str = "LPIPS",
      y_metric: str = "DFR_strict_prop",
  ) -> None:
      """One curve per 'name', one marker per budget level. Annotate budgets.
      If runs has only 1 point per attack (current state), still produce a
      scatter — it's a valid degenerate Pareto."""

## B.8 Scaffolding stubs (Phase 2/3 — implement signatures only, raise
        NotImplementedError("Available after Phase 2/3") in body)

  src/viz/ablation.py:
      def plot_ablation_curve(axis: str, results: list[dict], out_path: Path) -> None: ...

  src/viz/convergence.py:
      def plot_loss_convergence(history: dict[str, list[float]], out_path: Path) -> None: ...

  src/viz/temporal.py:
      def plot_latent_jitter(jitters: dict[str, list[float]], out_path: Path) -> None: ...
      def export_video_comparison(clean_dir, adv_dirs, out_path) -> None: ...

These can stay as stubs for now. Later phases will fill them in. Document
clearly in the docstring what data they expect.

## B.9 scripts/generate_figures.py

Orchestrator that:
1. Detects which data is available in results/ (count JSONs, det JSONs, ablation runs, etc.)
2. Calls every viz function whose data IS available
3. Skips (with a printed notice) those whose data is not yet available
4. Saves all output to results/figures/png/ and results/figures/pdf/

After Phase 1.5 + this work, generate_figures.py should produce AT LEAST:

  results/figures/
    png/
      f1_per_frame_dfr_distribution.png
      f2_n_clean_vs_n_adv.png
      f3_qualitative_grid_4frames.png
      f4_difference_heatmap_grid.png
      f5_metric_bar_chart_dfr.png
      f6_metric_bar_chart_asr.png
      f7_stealth_vs_effectiveness_preview.png
      f8_detection_overlay_img00001.png
      f9_detection_overlay_img00037.png   # one success, one failure
      f10_detection_overlay_img00048.png
    pdf/
      <same files .pdf>

Frame selection for grids/overlays: pick at least one binary-DFR success
(latent: img00001-04 or img00008), one moderate (img00020), one PGD partial
(img00037), and one FGSM failure (any from the 16 with n_adv > n_clean).

## B.10 Update PHASE1_DONE.md

Append a "Phase 1.5 + Visualization" section listing:
- Final ASR_strict and mAP_drop numbers per attack
- Path to results/figures/png/index.html (optional: a simple HTML index that
  shows all PNGs inline — handy for quick review)
- Note on FGSM ASR_strict interpretation

============================================================
Constraints and verification
============================================================

# Constraints
- DO NOT modify scripts/run_attack.py or attack code
- DO NOT regenerate adversarial images
- New deps allowed: ultralytics (already there), torchmetrics (Phase 1),
  matplotlib (likely already there). NO seaborn, NO plotly.
- All figures must be reproducible: matplotlib's default state must be
  restored after each save (use plt.close(fig) religiously)
- Total runtime < 8 minutes including viz generation
- Use pathlib.Path everywhere, no string paths
- Type hints everywhere

# Verification
1. python scripts/run_full_eval.py runs end-to-end, GPU < 5 min
2. results/dets_*.json files exist and load via src/eval/io.load_detections
3. results/metric_comparison.md has ASR_strict + mAP_drop populated with CI
4. Sanity check: clean detection ~387 total (±5%) — abort with error otherwise
5. python scripts/generate_figures.py produces all f1..f10 PNG/PDF
6. pytest tests/ -v still passes 33/33
7. Open one PNG and verify it has correct attack colors per PALETTE

# Honest expectations
- ASR_strict for FGSM may surprise: the class-based criterion can be
  satisfied by class-shifts even when DFR is negative. Report the actual
  number; this is a real finding either way.
- mAP_drop will probably show LATENT > PGD > FGSM cleanly because mAP
  penalizes both false negatives AND false positives. FGSM's FP inflation
  will show up as low mAP on adv (high mAP_drop is good for the attacker).

Start Part A. When detections are saved, run Part B. The viz module should
be FULLY USABLE with current Phase 1.5 data — don't leave any function
broken.
```

---

## What you'll have after this session

**Real numbers** for the headline metrics table:
- ASR_strict (class-based) for all 3 attacks with bootstrap CI
- mAP_drop@0.5 and mAP@0.5:0.95 dataset-level

**Working visualization pipeline** that produces these figures from current data:

| Figure | Purpose | Status after this session |
|---|---|---|
| Per-frame DFR distribution | Shows variance, FP inflation visually | ✓ generated |
| n_clean vs n_adv scatter | Reveals FGSM inflation pattern | ✓ generated |
| Qualitative 4-frame grid | Visual proof of attack | ✓ generated |
| Difference heatmap grid | Shows perturbation localization | ✓ generated |
| Metric bar charts (DFR, ASR) | Headline figure with CI | ✓ generated |
| Stealth-vs-effectiveness preview | Single-budget scatter | ✓ generated |
| Detection overlay (3 frames) | Side-by-side success/failure | ✓ generated |

**Scaffolding for future phases:**
- `pareto.py` — works now (1 point per attack), will work with full Pareto data later
- `ablation.py`, `convergence.py`, `temporal.py` — stubs that raise NotImplementedError with clear data-shape docstrings, ready to fill in during Phases 2-3
