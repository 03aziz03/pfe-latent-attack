# Experiments — Object-Aware Latent Adversarial Attack on YOLOv8

**Author:** Mohamed Aziz Brahmi  
**Dataset:** UA-DETRAC (50 frames, img00001–img00050)  
**Detector:** YOLOv8n fine-tuned on DETRAC (`runs/yolov8n_detrac/best.pt`)  
**Hardware:** Google Colab Pro — NVIDIA L4 GPU  
**Date:** May 2026

---

## 1. Setup

### Detector
- Architecture: YOLOv8n (nano)
- Training: 80 epochs on UA-DETRAC, mosaic augmentation, early stopping on val mAP
- Weights: `runs/yolov8n_detrac/best.pt` (5.9 MB)
- Inference threshold: `conf_thr = 0.25`, `iou_nms = 0.45`

### VAE
- Model: `stabilityai/sd-vae-ft-mse` (frozen, ~335 MB)
- Spatial downsampling: ×8 (latent size: 4 × H/8 × W/8)
- Latent scale: 0.18215 (standard SD value)

### Images
- 50 JPEG frames from UA-DETRAC, resized and letterboxed to 640×640
- Total clean detections across all 50 frames: **387**
- Average detections per frame: **7.74**

---

## 2. Attack configurations

All three attacks use the same:
- Bounding-box-restricted perturbation region (union of clean boxes)
- Vanishing detection loss: `ReLU(p_c − gamma)^2` summed over originally-detected classes
- YOLOv8 confidence threshold: `conf_thr = 0.25`

| Parameter | Latent (ours) | PGD | FGSM |
|---|---|---|---|
| Perturbation space | SD-VAE latent | Pixel | Pixel |
| Budget | `eps_z = 0.50` (latent L-inf) | `eps = 8/255` (pixel L-inf) | `eps = 8/255` (pixel L-inf) |
| Steps | 200 (early stop at ~150) | 50 | 1 |
| Step size | Adam `lr = 0.05` | `alpha = 1/255` | — |
| `gamma` | 0.05 | 0.05 | 0.05 |
| `lambda_p` | 0.05 | — | — |
| `lambda_r` | 1e-3 | — | — |

---

## 3. Results

### 3.1 Metrics table

| Metric | Latent (ours) | PGD | FGSM |
|---|---|---|---|
| **DFR** | **0.310** | 0.111 | 0.034 |
| **ASR** | **0.76** | 0.58 | 0.22 |
| **Mean confidence drop** | **0.263** | 0.143 | 0.014 |
| **PSNR_mask (dB) ↑** | 20.6 | **32.5** | 30.1 |
| **Masked L2 ↓** | 0.00885 | 0.000561 | 0.000984 |
| Clean detections | 387 | 387 | 387 |
| Kept after attack | 267 | 344 | 374 |
| Removed | 120 | 43 | 13 |

### 3.2 Metric definitions

| Metric | Definition |
|---|---|
| **DFR** (detection removal rate) | `1 − n_kept / n_clean` — fraction of individual clean detections removed across all frames |
| **ASR** (attack success rate) | Fraction of frames in which at least one clean detection was removed |
| **Mean confidence drop** | Average of `score_clean − score_adv` over all clean detections |
| **PSNR_mask** | Peak signal-to-noise ratio computed only over pixels inside the bounding-box mask (dB); higher = stealthier |
| **Masked L2** | RMS pixel error inside the mask, normalized by mask area and RGB channels |

> **Note:** The DFR and ASR definitions above match the implementation in
> `scripts/evaluate.py`. They differ from the stricter formal definitions in
> `docs/method.tex` (DFR = fraction of frames with zero detections; ASR =
> fraction of frames where every originally-detected class disappears). This
> will be corrected in a future version of `evaluate.py`.

---

## 4. Analysis

### 4.1 Attack effectiveness

The latent attack is the most effective by every detection-level metric:
- Removes **3.4×** more detections than PGD (DFR: 0.310 vs 0.111)
- Removes **9.2×** more detections than FGSM (DFR: 0.310 vs 0.034)
- Achieves ASR of **0.76** — in 76% of frames at least one detection disappears

The gap between ASR (0.76) and DFR (0.31) reflects that the attack reliably
removes the dominant detections in most frames but occasionally leaves a few
low-confidence boxes. The 24% failure rate (ASR) corresponds to frames where
the detector is particularly confident or the bounding boxes are very large,
making it harder to suppress confidence within the latent budget.

### 4.2 Stealth (visual quality)

The latent attack has lower PSNR_mask (20.6 dB) and higher masked L2 (0.00885)
compared to both pixel-space methods. Two factors explain this:

1. **VAE reconstruction error.** The SD VAE produces a base reconstruction
   error of ~0.33 max pixel difference even at `delta = 0` on DETRAC frames.
   This is because `stabilityai/sd-vae-ft-mse` was trained on natural images
   (LAION) and does not compress surveillance footage as efficiently. This base
   error is baked into the PSNR_mask of every adversarial image regardless of
   `delta`.

2. **Structured perturbation vs. noise.** Despite the lower PSNR_mask, the
   latent attack produces structured, texture-like distortions inside boxes
   (the decoder spreads the latent perturbation over coherent regions), whereas
   FGSM produces high-frequency sign-gradient noise. Whether structured
   distortion is more or less perceptible to a human observer than noise at
   the same L2 magnitude is debatable — LPIPS would be a better stealth metric
   than PSNR_mask for this comparison.

### 4.3 Sanity check results (single image, img00001)

| Check | Result |
|---|---|
| Outside mask `|x' − x|_max` | `0.000e+00` — paste-back is exact |
| Inside mask `|x' − x|_max` (delta = 0) | `3.322e-01` — VAE reconstruction error |
| Final `p_max` after attack | `0.031` (target `< gamma = 0.05`) |
| L_det: initial → final | `0.723 → 0.001` (−99.9%) |
| Steps to convergence | 150 / 200 (early stop) |

---

## 5. Potential improvements

### 5.1 Fix metric definitions (do this first)

`scripts/evaluate.py` computes DFR and ASR with looser definitions than
`docs/method.tex`. This must be corrected before any improved results are
reported, otherwise comparisons are against an inconsistent baseline.

### 5.2 Fine-tune the VAE on DETRAC (stealth)

The single biggest driver of low PSNR_mask is the VAE reconstruction error
on out-of-distribution traffic frames. Fine-tuning `stabilityai/sd-vae-ft-mse`
for 10–20 epochs on DETRAC would reduce the base error and raise PSNR_mask
significantly without any change to the attack algorithm.

**Expected gain:** PSNR_mask from ~20 dB to ~28–30 dB.

### 5.3 Multi-scale vanishing loss (effectiveness)

YOLOv8 runs detection heads at strides 8, 16, and 32. The current loss
suppresses all scales implicitly through the class-confidence max over all
anchors. An explicit per-scale term would force the attack to suppress
detections at every stride simultaneously, preventing recovery at coarser scales.

**Expected gain:** DFR and ASR increase on frames where the detector currently
recovers a coarse-scale detection after fine-scale ones are suppressed.

### 5.4 Latent PGD instead of Adam (convergence speed)

Replace Adam with projected sign-gradient descent in the latent L-inf ball.
PGD is theoretically more suited to L-inf constrained problems and typically
converges in fewer steps, reducing the per-image cost without losing
effectiveness.

**Expected gain:** Same DFR/ASR in ~100 steps instead of ~150, halving
compute cost (~1 min/image on L4).

### 5.5 LPIPS perceptual loss (stealth)

Replace the masked L2 term with a masked LPIPS loss. LPIPS correlates better
with human perception and penalizes structured artifacts more than raw pixel
MSE, giving the optimizer a better-calibrated stealth budget.

**Expected gain:** Lower perceptual distortion inside boxes at the same DFR,
and a more meaningful stealth metric for the comparison table.

### 5.6 Warm restarts (effectiveness on hard frames)

The attack currently fails on ~24% of frames. Running 3–5 random restarts with
different `delta` initializations and keeping the best result by final `p_max`
would push ASR toward 0.90+ at a linear compute cost.

**Expected gain:** ASR from 0.76 to ~0.88–0.92.

---

## 6. Reproduction

```bash
# 1. install
pip install -r requirements.txt

# 2. run latent attack on 50 frames
python scripts/run_attack.py \
    --input data/images_50 \
    --output results/adv_latent

# 3. run baselines
python baselines/fgsm.py \
    --input data/images_50 \
    --output results/adv_fgsm

python baselines/pgd_pixel.py \
    --input data/images_50 \
    --output results/adv_pgd

# 4. evaluate all three
python scripts/evaluate.py --clean data/images_50 --adv results/adv_latent --out results/metrics_latent.json
python scripts/evaluate.py --clean data/images_50 --adv results/adv_fgsm   --out results/metrics_fgsm.json
python scripts/evaluate.py --clean data/images_50 --adv results/adv_pgd    --out results/metrics_pgd.json
```

Config used: `configs/default.yaml` with `eps_z: 0.50`, `lr: 0.05`,
`num_steps: 200`, `device: cuda`.

Raw metric JSON files: `results/metrics_latent.json`, `results/metrics_fgsm.json`,
`results/metrics_pgd.json`.
