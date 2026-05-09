# Claude Code — Phase 1.5: Unblock Metrics

Goal: get real numbers into `results/metric_comparison.md` by re-running YOLOv8 detection on existing clean and adversarial images. **This does not re-run the attack.** It only re-runs detection to capture raw `FrameDetections` (boxes, scores, classes) instead of just counts.

Cost: ~30 seconds of GPU time. Run on Colab L4 or any machine with a CUDA GPU.

---

## PROMPT

```
Phase 1.5: populate raw detection JSON so recompute_metrics.py produces real numbers.

# Context
Phase 1 is complete (33/33 tests passing). The blocker is that existing results/metrics_*.json files store only counts, not raw FrameDetections. We need to re-run YOLOv8n detection on existing images (clean + already-generated adversarial) and save the raw outputs. The attack itself does NOT need to run again — adversarial images already exist on disk.

Existing artifacts on disk:
- data/images_50/                 — 50 clean DETRAC frames
- results/adv_latent/             — 50 latent-attack adversarial images
- results/adv_pgd/                — 50 PGD adversarial images
- results/adv_fgsm/               — 50 FGSM adversarial images
- runs/yolov8n_detrac/best.pt     — fine-tuned YOLOv8n weights

# Deliverables

## 1. New file: src/eval/run_detection.py

  def run_detection(
      image_dir: Path,
      model_path: Path,
      output_path: Path,
      conf_thr: float = 0.25,
      iou_nms: float = 0.45,
      img_size: int = 640,
      device: str = "cuda",
  ) -> dict[str, FrameDetections]:
      """Run YOLOv8 detection on every image in image_dir.
      Save raw FrameDetections via src.eval.io.save_detections.
      Returns the dict for in-memory use too."""

Implementation notes:
- Use ultralytics.YOLO. Load weights from model_path.
- Iterate images with sorted(image_dir.glob('*.jpg')) for stable ordering.
- Letterbox to img_size=640 (the same preprocessing the attack used).
- Frame ID = image stem (e.g., "img00001").
- Convert ultralytics Results to FrameDetections:
    boxes   = result.boxes.xyxy.cpu()      # (N, 4)
    scores  = result.boxes.conf.cpu()      # (N,)
    classes = result.boxes.cls.cpu().long()
- Apply conf threshold AFTER receiving results (consistent with what evaluate.py did).
- tqdm progress bar.
- Save_detections handles serialization.

## 2. New file: scripts/run_full_eval.py

CLI script that:
1. Runs run_detection on data/images_50 → results/dets_clean.json
2. Runs run_detection on results/adv_latent → results/dets_latent.json
3. Runs run_detection on results/adv_pgd → results/dets_pgd.json
4. Runs run_detection on results/adv_fgsm → results/dets_fgsm.json
5. Computes per-frame metrics for each (clean, attack) pair using src/eval/metrics.py
6. Computes bootstrap CI (n_boot=1000) for DFR_proportional, DFR_binary, ASR, conf_drop
7. Writes results/metrics_full.json with structure:
     {
       "latent": {
         "DFR_proportional": {"mean": ..., "ci_low": ..., "ci_high": ...},
         "DFR_binary":       {...},
         "ASR":              {...},
         "conf_drop":        {...},
         "n_frames": 50,
         "n_clean_dets": ...,
         "n_adv_dets": ...
       },
       "pgd": {...},
       "fgsm": {...}
     }
8. Updates results/metric_comparison.md with a clean side-by-side:

   | Metric             | Latent (old) | Latent (new) | PGD (old) | PGD (new) | FGSM (old) | FGSM (new) |
   | DFR_proportional   | 0.310        | x.xxx ± CI   | 0.111     | ...        | 0.034      | ...        |
   | DFR_binary         | —            | x.xxx ± CI   | —         | ...        | —          | ...        |
   | ASR (strict class) | 0.760        | x.xxx ± CI   | 0.580     | ...        | 0.220      | ...        |
   | conf_drop          | 0.263        | x.xxx ± CI   | 0.143     | ...        | 0.014      | ...        |

   Add a paragraph explaining the gap honestly: which numbers went down, by how much, and why the new ones are correct.

## 3. mAP drop

Add per-frame mAP@0.5 and mAP@0.5:0.95 to the metrics dict via torchmetrics.detection.MeanAveragePrecision. Report dataset-level values (not bootstrapped — torchmetrics doesn't trivially support per-frame aggregation for mAP, so just report the global value computed across all frames at once for adv vs. clean-as-pseudo-GT).

## 4. Quick sanity check

Add a simple assertion in run_full_eval.py: clean detection should reproduce the 387 total detections from the original report. If it doesn't (off by more than ±5%), abort with a clear error — something has changed (preprocessing, NMS thresholds, etc.).

## 5. Update PHASE1_DONE.md

Append a "Phase 1.5 — Real numbers" section with the new headline metrics (mean ± CI for all three attacks) and a one-line interpretation of how the gap with the old report changes the attack story.

# Constraints
- Do NOT modify scripts/run_attack.py
- Do NOT regenerate adversarial images (they exist on disk)
- Do NOT install new heavy deps. Use ultralytics (already installed for run_attack.py) and torchmetrics (added in Phase 1)
- The run must complete in under 5 minutes on a single GPU
- All outputs go to results/, never overwrite existing files — append timestamps if needed (e.g., results/dets_clean_20260507.json) but the canonical names should be results/dets_<source>.json

# Verification
1. python scripts/run_full_eval.py runs end-to-end
2. results/metric_comparison.md is populated with real numbers (not "TODO")
3. results/dets_*.json files are valid JSON loadable by src/eval/io.load_detections
4. New ASR is < old ASR (the strict version should be more demanding)
5. Existing pytest suite still passes

# Honest expectation
The new strict ASR will likely be 0.30-0.50 for the latent attack (vs. 0.76 in the old report). The new DFR_binary will likely be 0.10-0.25. These are the correct numbers; the old ones were inflated by loose definitions. Document the gap matter-of-factly in metric_comparison.md.
```

---

## After this runs

You'll have actual ground-truth numbers for the latent / PGD / FGSM comparison, with bootstrap CI. From there:

1. Update `experiments.md` with the corrected numbers (or write a new `experiments_v2.md` and deprecate the old one)
2. Decide whether the corrected gap latent-vs-PGD is still meaningful for the paper (likely yes, but the framing changes from "3.4× better" to something more nuanced)
3. Move to **Phase 2**: LPIPS loss, VAE finetune, iso-budget runner

If the corrected ASR ends up below ~0.30 for the latent attack, we'll need to discuss whether your current `eps_z=0.50` budget is too low, or whether the loss formulation needs adjustment before LPIPS gets layered on top. Better to learn that now than after spending compute on Phase 2.
