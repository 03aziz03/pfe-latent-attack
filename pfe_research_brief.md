# PFE Research Brief — Object-Aware Latent Adversarial Attack on YOLOv8

**Author**: Mohamed Aziz Brahmi
**Email**: mohamedazizbrahmi29@gmail.com
**Session date**: 2026-05-04
**Purpose of this file**: Self-contained brief to bootstrap a fresh Cowork session whose goal is to write the PFE (final-year engineering internship) report in English. Drop this file into a new chat and ask: *"Help me write the PFE report in English using this brief."*

---

## 0. One-Sentence Pitch

We propose a minimal, reproducible adversarial attack that perturbs only the bounding-box latents of a frozen Stable-Diffusion VAE under a vanishing-confidence objective, causing all originally-detected objects to disappear from a YOLOv8 detector's output while keeping perturbations imperceptible inside the masked region.

---

## 1. Project Overview

### Context
- **Type**: Final-year engineering internship project (PFE — Projet de Fin d'Études)
- **Domain**: Computer vision, adversarial machine learning
- **Time budget**: ~1 month (fast-track of an originally 3–4 month plan)
- **Detector under attack**: YOLOv8 (single architecture; we focus depth, not breadth)
- **Attack type**: Digital-only (no physical-world adversarial patches)
- **Application domain**: Vehicle detection in surveillance video (UA-DETRAC dataset)

### Research Question
*Can a perturbation restricted to the bounding-box latent footprint of a pre-trained generative VAE suppress all originally-detected objects in a YOLOv8 output while remaining imperceptible — and if so, how does this compare to standard pixel-space attacks at matched perturbation budget?*

### Hypothesis
Restricting perturbations to (a) object regions in pixel space and (b) the latent manifold of a Stable-Diffusion VAE yields an attack that is at least as effective as pixel-space PGD on detection failure rate, while producing perturbations that are biased toward natural-image directions — measured by lower LPIPS / higher PSNR inside the perturbed region.

---

## 2. Methodology

### 2.1 Formal Problem Statement

Let:
- `x ∈ R^(3×H×W)` — clean image
- `f_θ` — frozen YOLOv8 detector
- `D_clean = f_θ(x)` — set of detections at confidence threshold τ
- `C_clean` — set of unique class IDs present in `D_clean`
- `E, D` — frozen Stable-Diffusion VAE encoder and decoder; latent shape `(4, H/8, W/8)`
- `M ∈ {0,1}^(H×W)` — pixel mask = union of bounding boxes from `D_clean`
- `M_z ∈ {0,1}^(4, H/8, W/8)` — latent mask = `MaxPool₈(M)` broadcast to 4 channels

We seek a latent perturbation `δ` such that:

```
δ* = argmin_δ   L_det(f_θ(x'), C_clean) + λ_p · L_perc(x', x; M) + λ_r · ‖δ‖²
   s.t.        ‖δ‖_∞ ≤ ε_z       and       δ = M_z ⊙ δ
```

where the adversarial image is reconstructed as:

```
z       = E(x)
z_adv   = z + M_z ⊙ δ
x_dec   = D(z_adv)
x'      = M ⊙ x_dec + (1 − M) ⊙ x      (paste-back)
```

### 2.2 Loss Components

**Vanishing detection loss (class-level, no IoU matching):**
```
p_c(x') = max over anchors a of  conf_{a,c}(x')                  for c ∈ C_clean
L_det   = (1/|C_clean|) · Σ_c  ReLU(p_c − γ)²
```
For each class originally present, we drive its maximum class-confidence anywhere in the image below threshold γ. Once `p_c < γ`, the term saturates (no further gradient) — this is anti-overshoot behavior.

**Why class-level instead of per-instance IoU matching?** Simpler, more stable in autograd, no post-NMS decoding during the attack loop, and aligned with the false-negative goal: if `p_c < γ` for every clean class, NMS at threshold γ produces zero detections of those classes.

**Masked perceptual loss (pixel-space L2):**
```
L_perc = ‖M ⊙ (x' − x)‖₂² / (‖M‖₁ · C)
```
Normalized by foreground pixel count × channel count so `λ_p` is comparable across images of different object sizes.

**Latent regularizer:**
```
L_reg  = mean(δ²)
```

### 2.3 Why Latent Space?

1. **Manifold prior.** SD VAE's decoder is biased toward natural-image directions; perturbations along latent dimensions tend to look more natural than in pixel space.
2. **Object-aware mask.** Latent mask `M_z = MaxPool₈(M)` ensures the perturbation lives only on object footprints in latent space.
3. **Paste-back guarantee.** Even if the VAE decoder leaks artifacts globally, the paste-back operator restores untouched pixels outside `M`, giving a hard pixel-level locality constraint.

### 2.4 Optimization

- **Optimizer**: Adam (NOT signed PGD). Latent gradients are anisotropic; signed updates overshoot in low-sensitivity directions. Adam's per-coordinate normalization handles the wide value range.
- **Projection**: After each Adam step, with `torch.no_grad()`:
  ```
  delta.clamp_(-ε_z, ε_z)         # L_∞ projection
  delta.mul_(M_z)                 # zero non-object latents
  ```
- **Default hyperparameters**: ε_z = 0.10, γ = 0.05, λ_p = 0.05, λ_r = 1e-3, lr = 0.01, T = 80 steps.
- **Early stopping**: stop when `max_c p_c < γ` (attack succeeded).

---

## 3. System Architecture

```
                       ┌──────────────────────────────┐
   x (image) ────────► │ YOLOv8 (frozen)              │ ──► D_clean (boxes)
                       └──────────────────────────────┘
                                        │
                                        ▼
                       ┌──────────────────────────────┐
                       │ Mask builder: boxes → M, M_z │
                       └──────────────────────────────┘
                                        │
                                        ▼
   x ──► [ VAE Encoder E ] ──► z ──► z + M_z⊙δ ──► [ VAE Decoder D ] ──► x_dec
                                                                         │
                                                                         ▼
                                                       x' = M⊙x_dec + (1−M)⊙x
                                                                         │
                                                                         ▼
                                                              [ YOLOv8 (frozen) ]
                                                                         │
                                                                         ▼
                                              L_det + λ_p·L_perc + λ_r·‖δ‖²
                                                                         │
                                                                  Adam step on δ
                                                                  + L_∞ projection
```

**Five modules, all decoupled:**
1. **Detection** — `src/detector.py`: `YOLOv8Wrapper.forward_raw()` exposes pre-NMS logits for autograd; `detect_nms()` builds `D_clean`.
2. **Latent encoding** — `src/vae.py`: `SDVAE` from `stabilityai/sd-vae-ft-mse`, frozen.
3. **Masking** — `src/masks.py`: `boxes_to_pixel_mask()` + `pixel_mask_to_latent_mask()` (max-pool stride 8).
4. **Attack** — `src/attack.py`: `LatentObjectAttack` class (Adam loop, projection, history).
5. **Losses** — `src/losses.py`: `vanishing_loss`, `masked_l2`, `latent_l2`.

---

## 4. Implementation

### File Structure

```
minimal_research/
├── configs/
│   └── default.yaml         # all hyperparameters
├── src/
│   ├── detector.py          # YOLOv8 wrapper
│   ├── vae.py               # SD VAE wrapper
│   ├── masks.py             # bbox → pixel mask → latent mask
│   ├── losses.py            # vanishing + L2 + reg
│   ├── attack.py            # LatentObjectAttack
│   ├── data.py              # folder image loader
│   └── utils.py             # IoU, viz, IO
├── baselines/
│   ├── fgsm.py              # pixel-space FGSM (same vanishing loss)
│   └── pgd_pixel.py         # pixel-space PGD (full / mask-restricted)
├── scripts/
│   ├── sanity_check.py      # 3-check verification
│   ├── run_attack.py        # batch attack with metadata
│   └── evaluate.py          # DFR, ASR, mean confidence drop, PSNR_mask
├── tools/
│   ├── detrac_to_yolo.py    # DETRAC XML → YOLO converter
│   └── train_yolov8.py      # local training wrapper
├── notebooks/
│   └── train_yolov8_detrac.ipynb   # Colab Pro training notebook
└── runs/yolov8n_detrac/best.pt     # trained detector weights (after Phase 1)
```

### Stack
- PyTorch ≥ 2.1 + CUDA 12.1
- `ultralytics` ≥ 8.2 (YOLOv8 + training pipeline)
- `diffusers` (SD VAE)
- `pycocotools` (mAP evaluation)

---

## 5. Dataset — UA-DETRAC

### Why DETRAC?
- **Domain-specific**: traffic surveillance video, vehicle classes (car, bus, van, others). Justifies domain-specific YOLOv8 fine-tuning rather than just attacking COCO weights.
- **Practical relevance**: surveillance failure modes (missing vehicles) have real-world safety implications, which strengthens the false-negative framing.
- **Frame-level annotations**: simplifies the conversion to YOLO format.

### Conversion (DETRAC XML → YOLO format)
- DETRAC ships with one XML per `MVI_*` sequence; each XML contains per-frame bounding boxes and `vehicle_type` attributes.
- Custom converter (`tools/detrac_to_yolo.py`) maps:
  - `vehicle_type` → class id: `car=0, bus=1, van=2, others=3`
  - drops boxes with `truncation_ratio > 0.5` (heavily occluded)
  - splits BY sequence (10% val) to avoid leakage between adjacent frames
  - keeps every 5th frame to reduce ~140k frames to ~28k (DETRAC is 25 fps; consecutive frames are ~99% redundant)

### Stats (this run)
- **Train**: 14 170 frames / 51 sequences
- **Val**: 2 224 frames / 9 sequences
- **Split**: 86% / 14% by sequence
- **Classes**: 4 (car, bus, van, others)

### Detector Training
- Starting weights: `yolov8n.pt` (COCO-pretrained, transfer learning)
- Architecture: yolov8n (3.0M parameters, 8.2 GFLOPs)
- Training: SGD, lr0 = 0.01, cos_lr, 60–80 epochs, patience 15, batch 32 on L4 GPU
- Augmentations: mosaic (off last 10 epochs), `degrees=0` (cameras don't rotate), `flipud=0`
- **Target metrics**: mAP@50 ≥ 0.75 overall

### Status
Training was launched on 2026-05-04 on Colab Pro L4. Expected ~2–3 hours. Drive-sync callback persists `last.pt` and `best.pt` after every epoch. See `colab_training_debug_log.md` for the bring-up issues (numpy ABI, wandb integration, etc.) and their fixes.

---

## 6. Evaluation Protocol

### Setup
- **Test images**: held-out subset of DETRAC val frames (those not used for detector training).
- **Detector**: our DETRAC fine-tuned YOLOv8n (`runs/yolov8n_detrac/best.pt`).
- **Attack**: applied per-image; 80 Adam steps; ε_z = 0.10 default.

### Metrics

| Metric | Formula | Interpretation |
|---|---|---|
| **DFR** (Detection Failure Rate) | `1 − #kept_after_attack / #clean_detections` | % of original detections that disappear |
| **ASR** (Attack Success Rate) | `#images_with_at_least_one_disappeared / #images` | per-image success |
| **Mean confidence drop** | `mean(score_clean − score_adv)` | how much the survivors weakened |
| **mAP@50 drop** | `mAP_clean − mAP_adv` | end-to-end detection degradation |
| **PSNR_mask (dB)** | `10·log₁₀(1/MSE_inside_M)` | image quality inside perturbed region |
| **Masked L2** | `‖M⊙(x'−x)‖₂² / (‖M‖₁·3)` | raw perturbation magnitude |

### Baselines (matched budget)

| Method | Domain | Budget | Notes |
|---|---|---|---|
| **FGSM** | pixel | `‖x'−x‖_∞ ≤ 8/255` | 1-step |
| **PGD (full image)** | pixel | `‖x'−x‖_∞ ≤ 8/255` | 50 steps, α = 1/255 |
| **PGD (mask-restricted)** | pixel, inside M only | `‖x'−x‖_∞ ≤ 8/255` | isolates the contribution of latent space vs spatial restriction |
| **Ours (latent)** | latent, inside M_z | `‖δ‖_∞ ≤ ε_z` | 80 Adam steps |

All baselines use the **same** `vanishing_loss` for fair comparison.

### Required Tables / Figures for the Paper

1. **Main result table** — DFR / ASR / mAP_drop / PSNR_mask / Masked_L2 for FGSM, PGD-full, PGD-mask, Ours.
2. **Ablation 1** — `ε_z` sweep ∈ {0.05, 0.1, 0.15, 0.2}.
3. **Ablation 2** — `λ_p` sweep ∈ {0.0, 0.01, 0.05, 0.2}.
4. **Ablation 3** — with vs. without paste-back (to isolate the contribution of the locality enforcement).
5. **Qualitative grid** — for 5 sample images: clean | adv | difference (×10) | YOLO output overlay.
6. **Convergence plot** — `p_max` and `L_det` vs. attack step for one image.

---

## 7. Novelty / Contribution (for the paper)

We argue three concrete contributions, each defensible against reviewers:

### (N1) Object-aware latent perturbation
Existing latent / VAE-based attacks (e.g., AdvLatent, generative perturbations) perturb the *whole* latent and target classification. We restrict the perturbation to the latent footprint of object bounding boxes, paired with a paste-back operator that guarantees pixel-level locality. To our knowledge, this is the first work to combine spatial bounding-box constraints with latent-space optimization for a detector attack.

### (N2) Vanishing-only objective for object detection
Most detector attacks (e.g., DAG, RAP) mix objectness, classification, and box-regression terms. We isolate a single class-level *max-confidence-with-floor* term that targets exclusively the false-negative regime — simpler to analyze, easier to compare against baselines under matched budget, and directly aligned with safety-critical failure modes (missed pedestrians, missed vehicles).

### (N3) Manifold-constrained stealth without diffusion
By optimizing in the SD-VAE latent space (no denoising, no text conditioning, no full diffusion sampling), the perturbation is implicitly biased toward natural-image-manifold directions, yielding lower LPIPS / higher PSNR at equal attack strength compared to pixel PGD — at a fraction of the compute cost of diffusion-based attacks (ACE, DiffAttack).

### Suggested abstract sentence
> *We propose a minimal latent-space adversarial attack that perturbs only the bounding-box latents of a frozen Stable-Diffusion VAE under a vanishing-confidence objective, matching pixel-PGD's detection failure rate on YOLOv8 fine-tuned on UA-DETRAC while reducing perceptual distortion inside the masked region by X dB PSNR.*

---

## 8. Suggested Report Structure (English, ~30–40 pages)

```
1. Abstract                                            (1 page)
2. Acknowledgements                                    (1 page)
3. Introduction                                        (3–4 pages)
   3.1 Context: adversarial ML and object detection
   3.2 Motivation: false negatives in surveillance / autonomous driving
   3.3 Research question and contributions
   3.4 Outline of the report

4. Background and Related Work                         (5–6 pages)
   4.1 Object detection: YOLOv8 architecture
   4.2 Adversarial attacks: FGSM, PGD, C&W
   4.3 Generative-model-based attacks: AdvLatent, ACE, DiffAttack
   4.4 Detector attacks: DAG, RAP, vanishing attacks
   4.5 Stable Diffusion VAE as a manifold prior

5. Methodology                                         (5–6 pages)
   5.1 Formal problem statement
   5.2 Object-aware latent perturbation
   5.3 Vanishing detection loss (class-level formulation)
   5.4 Perceptual constraint and regularization
   5.5 Optimization: Adam with L_∞ projection
   5.6 Paste-back operator

6. Implementation                                      (3–4 pages)
   6.1 System architecture and module breakdown
   6.2 PyTorch implementation details
   6.3 Hyperparameters and computational cost
   6.4 Reproducibility (code structure, configs, seeds)

7. Experimental Setup                                  (3–4 pages)
   7.1 Dataset: UA-DETRAC, conversion to YOLO format
   7.2 Detector: YOLOv8n fine-tuned on DETRAC
   7.3 Baselines: FGSM, PGD (full and mask-restricted)
   7.4 Metrics: DFR, ASR, mAP drop, PSNR_mask
   7.5 Hardware and runtime

8. Results                                             (5–6 pages)
   8.1 Main comparison vs. FGSM and PGD baselines
   8.2 Ablation: ε_z sweep
   8.3 Ablation: λ_p sweep
   8.4 Ablation: with vs. without paste-back
   8.5 Qualitative results
   8.6 Convergence analysis

9. Discussion                                          (2–3 pages)
   9.1 Why latent-space wins on perceptual metrics
   9.2 Failure cases
   9.3 Threat model and limitations (digital-only, white-box)
   9.4 Possible defenses and future work

10. Conclusion                                         (1 page)

11. References                                         (2–3 pages)
12. Appendices                                         (as needed)
    A. Hyperparameter tuning details
    B. Additional qualitative examples
    C. Code listings of key algorithms
```

---

## 9. Required Citations (initial list — not exhaustive)

### Object detection
- Redmon, Divvala, Girshick, Farhadi (2016) — *YOLO: You Only Look Once*
- Jocher et al. (Ultralytics) — *YOLOv8 documentation* (or arXiv if available)

### Adversarial attacks (foundational)
- Goodfellow, Shlens, Szegedy (2014) — *Explaining and Harnessing Adversarial Examples* (FGSM)
- Madry et al. (2018) — *Towards Deep Learning Models Resistant to Adversarial Attacks* (PGD)
- Carlini & Wagner (2017) — *Towards Evaluating the Robustness of Neural Networks* (C&W)

### Detector-specific attacks
- Xie, Wang, Zhang, Zhou, Xie, Yuille (2017) — *Adversarial Examples for Semantic Segmentation and Object Detection* (DAG)
- Li, Tian, Ramamurthy, Ho, Long (2020) — *Robust Adversarial Perturbation on Deep Proposal-based Models* (RAP)

### Generative-model-based / manifold attacks
- Song et al. (2018) — *Constructing Unrestricted Adversarial Examples with Generative Models*
- Xiao et al. (2018) — *Generating Adversarial Examples with Adversarial Networks* (AdvGAN)
- Chen et al. (2024) — *Diffusion-based Adversarial Examples* (relevant context, not a baseline)

### Stable Diffusion VAE
- Rombach et al. (2022) — *High-Resolution Image Synthesis with Latent Diffusion Models* (LDM)
- Original VAE paper: Kingma & Welling (2013) — *Auto-Encoding Variational Bayes*

### Dataset
- Wen et al. (2020) — *UA-DETRAC: A New Benchmark and Protocol for Multi-Object Detection and Tracking*

### Perceptual metrics
- Zhang et al. (2018) — *The Unreasonable Effectiveness of Deep Features as a Perceptual Metric* (LPIPS)

---

## 10. Status & Next Steps (as of 2026-05-04)

### Done
- Full method definition and mathematical formulation (refined twice).
- Repository scaffolded with all modules (`src/`, `baselines/`, `scripts/`, `tools/`, `notebooks/`).
- DETRAC dataset converted to YOLO format (14 170 train / 2 224 val).
- Colab notebook for resilient training (per-epoch Drive-sync, resume capability).

### In Progress
- YOLOv8n fine-tuning on DETRAC (60 epochs, ~2–3h on L4 GPU).
- Resolving wandb integration error (currently blocking training start; fix documented in `colab_training_debug_log.md`).

### Pending
- Download `best.pt` from Drive to local repo at `runs/yolov8n_detrac/best.pt`.
- Update `configs/default.yaml` to point to the trained weights.
- Run sanity check (`scripts/sanity_check.py`) with DETRAC weights on 1 frame.
- Run latent attack on a 200-image DETRAC val subset.
- Run FGSM and PGD baselines on the same subset.
- Compute and tabulate metrics (DFR, ASR, mAP_drop, PSNR_mask).
- Run ablation studies (ε_z, λ_p, paste-back).
- Generate qualitative figures.
- Write the report (see Section 8).

### Files in This Repo (relevant for the report)
- `README.md` — repo overview and quick-start
- `colab_training_debug_log.md` — Colab issues + fixes (tactical reference)
- `pfe_research_brief.md` — *this file* (strategic / methodology reference)
- `notebooks/train_yolov8_detrac.ipynb` — full training pipeline
- `src/attack.py` — attack implementation (algorithm pseudocode in docstring)
- `configs/default.yaml` — all hyperparameters with their meanings

---

## 11. How to Use This File in a New Cowork Session

1. **Open a fresh Cowork conversation.**
2. **Drag-drop or attach this file** (`pfe_research_brief.md`).
3. **Initial prompt example**:
   > *I want to start writing my PFE internship report in English. I'm attaching a research brief that summarizes the project. Please read it, then help me draft the **Introduction** chapter (Section 3 in the suggested structure). Aim for ~3 pages, academic tone, with placeholders for citations I'll fill in. Use the contributions from Section 7 as the framing.*

4. **Iterate chapter by chapter**:
   - Introduction → Background → Methodology → Implementation → Setup → Results → Discussion → Conclusion
   - Wait until experimental results are in before writing the **Results** and **Discussion** chapters.

5. **Optional companion files to attach**:
   - `colab_training_debug_log.md` — if you want help analyzing the training process for the *Implementation* or *Setup* chapter.
   - `configs/default.yaml` — for exact hyperparameter listings.
   - `src/attack.py` — if you want algorithmic pseudocode generated for the *Methodology* chapter.

---

*End of brief. Total ~9 pages of structured context. Designed to give a fresh Cowork session everything it needs to help write the PFE report from scratch.*
