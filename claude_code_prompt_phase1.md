# Claude Code — Phase 1 Kickoff Prompt

Copy-paste the block below into Claude Code at the root of your project repository.

---

## PROMPT

```
You are helping me professionalize an adversarial attack research framework for a final-year project (PFE). The current codebase implements an object-aware latent-space adversarial attack on YOLOv8 using a frozen Stable Diffusion VAE on UA-DETRAC frames.

# Current state of the codebase

Existing files (do not delete or break them):
- scripts/run_attack.py        — main latent attack optimizer
- scripts/evaluate.py          — metric computation (HAS A BUG, see below)
- baselines/fgsm.py            — FGSM baseline
- baselines/pgd_pixel.py       — PGD pixel baseline
- configs/default.yaml         — base config (eps_z=0.50, lr=0.05, num_steps=200, lambda_p=0.05, lambda_r=1e-3, gamma=0.05)
- runs/yolov8n_detrac/best.pt  — fine-tuned YOLOv8n weights
- data/images_50/              — 50 DETRAC frames (img00001..img00050)
- results/metrics_latent.json, metrics_pgd.json, metrics_fgsm.json — raw eval output from the broken evaluator
- docs/method.tex              — formal method description with the CORRECT metric definitions

# Critical bug to fix

`scripts/evaluate.py` uses LOOSE metric definitions that contradict `docs/method.tex`. This is a publication blocker. You will not modify `evaluate.py` directly — you will write a new module that supersedes it, and add a deprecation warning to the old one.

LOOSE (current, in scripts/evaluate.py):
  DFR_loose = 1 - n_kept_total / n_clean_total                # aggregated over all detections
  ASR_loose = fraction of frames with >=1 removed detection

STRICT (correct, from docs/method.tex):
  DFR_strict = mean over frames of (1 - n_adv_f / max(n_clean_f, 1))
  ASR_strict = fraction of frames where EVERY originally-detected class is absent
               from adv detections at IoU >= 0.5
  mAP_drop@0.5  = mAP@0.5(adv vs clean-as-pseudo-GT) drop from 1.0
  mAP_drop@0.5:0.95 = same but averaged over IoU thresholds 0.5..0.95 step 0.05

# Phase 1 deliverables (no GPU required)

You will implement the following in a clean modular layout. Create new files; do not move or rename existing files yet.

## 1. New directory structure

Create these directories and __init__.py files:

  src/
    __init__.py
    eval/
      __init__.py
      metrics.py            # NEW
      bootstrap.py          # NEW
      pareto.py             # NEW
      io.py                 # NEW — load/save detection results, frame metadata
  tests/
    __init__.py
    test_metrics.py         # NEW
    test_bootstrap.py       # NEW
    fixtures/
      synthetic_dets.json   # NEW — small handcrafted ground-truth/adv pairs

## 2. src/eval/metrics.py

Implement these functions with full type hints and docstrings:

  @dataclass
  class FrameDetections:
      boxes:   torch.Tensor   # (N, 4) xyxy
      scores:  torch.Tensor   # (N,)
      classes: torch.Tensor   # (N,) int

  def per_frame_dfr(clean: FrameDetections, adv: FrameDetections,
                    conf_thr: float = 0.25) -> float:
      """1 - n_adv / max(n_clean, 1) for one frame."""

  def per_frame_asr(clean: FrameDetections, adv: FrameDetections,
                    iou_thr: float = 0.5, conf_thr: float = 0.25) -> bool:
      """True iff every clean detection's class is absent from adv at IoU >= iou_thr."""

  def per_frame_map_drop(clean: FrameDetections, adv: FrameDetections,
                         iou_thr: float = 0.5) -> float:
      """mAP drop with clean as pseudo-GT. Reuse torchmetrics.detection.MeanAveragePrecision."""

  def per_frame_psnr_mask(clean_img: torch.Tensor, adv_img: torch.Tensor,
                          mask: torch.Tensor) -> float:
      """PSNR computed only inside the mask (boolean H x W)."""

  def per_frame_masked_l2(clean_img: torch.Tensor, adv_img: torch.Tensor,
                          mask: torch.Tensor) -> float:
      """RMS pixel error inside mask, normalized by mask area * 3 channels."""

  def per_frame_conf_drop(clean: FrameDetections, adv: FrameDetections,
                          iou_thr: float = 0.5) -> float:
      """Mean (score_clean - score_adv) over matched detections; 0 if no match."""

  def aggregate(per_frame: list[dict]) -> dict:
      """Mean of each per-frame metric, ignoring frames with n_clean == 0 for DFR/ASR."""

Use torchvision.ops.box_iou for IoU. Avoid manual IoU code. Match by greedy IoU within same class.

## 3. src/eval/bootstrap.py

  def bootstrap_ci(values: np.ndarray, n_boot: int = 1000,
                   ci: float = 0.95, seed: int = 42) -> tuple[float, float, float]:
      """Returns (mean, lo, hi). Uses np.random.default_rng(seed) for reproducibility."""

  def bootstrap_metric_dict(per_frame: list[dict], keys: list[str],
                            n_boot: int = 1000, seed: int = 42) -> dict[str, tuple]:
      """Returns {key: (mean, lo, hi)} for every requested metric."""

## 4. src/eval/pareto.py

  def build_pareto(runs: list[dict]) -> list[dict]:
      """Each run is {'name': str, 'budget': float, 'metrics': dict}.
      Returns list sorted by stealth metric ascending; computes dominance flags."""

  def plot_pareto(runs, x_key='LPIPS', y_key='DFR', save_path=None):
      """Matplotlib plot, one curve per attack name. Save PNG + PDF."""

## 5. src/eval/io.py

Helpers to read/write detection results in a standard JSON schema:

  def load_detections(path: Path) -> dict[str, FrameDetections]:
      """Load per-frame dets keyed by frame stem."""

  def save_detections(dets: dict[str, FrameDetections], path: Path) -> None: ...

Schema: {"frame_id": {"boxes": [[x1,y1,x2,y2],...], "scores": [...], "classes": [...]}}

## 6. tests/test_metrics.py

Cover at minimum:
- Empty clean (no objects) → DFR/ASR undefined; aggregate skips frame
- Empty adv (full success) → per_frame_dfr == 1.0, asr == True
- Identical clean/adv → DFR == 0, ASR == False, conf_drop == 0
- Class shift (clean=car, adv=truck same box) → ASR == True (class disappeared)
- IoU edge case: matched at exactly iou_thr boundary
- Numerical: PSNR_mask of identical images is +inf, handle via large finite

Use pytest fixtures with synthetic FrameDetections.

## 7. tests/test_bootstrap.py

- bootstrap_ci on degenerate input (all same value) returns (v, v, v)
- CI bracket monotonic in n_boot
- Reproducible with fixed seed

## 8. Backward-compat shim

In scripts/evaluate.py, add at top:

  import warnings
  warnings.warn(
      "scripts/evaluate.py uses loose DFR/ASR definitions. "
      "Use src/eval/metrics.py for strict definitions matching docs/method.tex.",
      DeprecationWarning, stacklevel=2)

Do not delete or rewrite the rest.

## 9. Reproduce existing results + diff

Write a small script `scripts/recompute_metrics.py` that:
1. Loads existing results/metrics_*.json (which contain raw per-frame detections)
2. Computes BOTH old and new metrics
3. Prints a side-by-side table
4. Saves results/metric_comparison.md with the gap

Expected output: the new strict ASR will be lower than 0.76 for the latent attack (likely 0.30-0.45). Document the gap honestly.

# Code style
- Python 3.10+, full type hints (use `from __future__ import annotations` and PEP 604 unions)
- Black formatting, 100-char lines
- Docstrings: Google style, with Args/Returns/Raises
- NO new dependencies beyond: torch, torchvision, torchmetrics, numpy, matplotlib, pytest
- Do NOT install pycocotools (use torchmetrics.detection)

# Verification before you finish
1. `pytest tests/ -v` passes all tests
2. `python scripts/recompute_metrics.py` produces results/metric_comparison.md
3. The script imports from src.eval.* without errors
4. No GPU is required for any of the above

# What NOT to do in this phase
- Do NOT modify scripts/run_attack.py or any attack optimization
- Do NOT add LPIPS, temporal loss, or VAE finetuning yet (those are Phase 2)
- Do NOT change configs/default.yaml
- Do NOT regenerate adversarial images
- Do NOT install heavy deps (lpips, ultralytics extras)

# Final deliverables
- src/eval/{metrics,bootstrap,pareto,io}.py with full implementation
- tests/ with passing pytest suite
- scripts/recompute_metrics.py
- results/metric_comparison.md showing old vs new numbers on existing data
- A short PHASE1_DONE.md summary at repo root listing what changed

Start by reading docs/method.tex to confirm the metric definitions match what I described, then proceed with the implementation. If the definitions in method.tex differ from what I wrote above, USE METHOD.TEX as the ground truth and flag the difference.
```

---

## How to use this prompt

1. Open Claude Code in your project directory: `cd ~/your-attack-repo && claude`
2. Paste the entire prompt above (everything between the triple backticks)
3. Claude Code will read `docs/method.tex` first, then implement
4. Review each file change before approving
5. Run the verification step (`pytest tests/ -v`) yourself before moving on

## When to use the next prompt

After Phase 1 is merged and `metric_comparison.md` is committed, ask me for the **Phase 2 prompt** which will cover:
- Masked LPIPS loss integration
- VAE fine-tuning script
- Iso-budget runner glue
- Caching layer for clean detections / latents / flow

That phase will use GPU, so we'll do it after the foundation is solid.
