# Object-Aware Latent Adversarial Attack on YOLOv8

Minimal, reproducible implementation of a latent-space adversarial attack against
YOLOv8 that increases false negatives (objects disappear from detection) while
constraining perturbations to bounding-box regions in the Stable-Diffusion VAE
latent space.

## Method (one-paragraph summary)

Given a clean image `x` and a frozen YOLOv8 detector `f`, we encode the image
into the latent space of a frozen Stable-Diffusion VAE, add a learnable
perturbation `delta` restricted to the latent footprint of detected bounding
boxes, decode back to pixel space, and paste the modified region back over the
clean image. The perturbation is optimized with Adam under an L-infinity
constraint so that, for every class originally detected, the maximum
class-confidence anywhere in the image drops below a threshold gamma — i.e. all
originally-detected objects vanish under YOLOv8 inference.

See `docs/method.md` (or the chat transcript that accompanied this repo) for
the full mathematical formulation.

## Workflow

```
[1] Train YOLOv8 on DETRAC (Colab Pro)              ─┐
        notebooks/train_yolov8_detrac.ipynb          │   produces best.pt
                                                     ▼
[2] Place best.pt under runs/yolov8n_detrac/best.pt  and set it in
    configs/default.yaml -> detector.weights
                                                     │
                                                     ▼
[3] Sanity-check the attack pipeline locally
        python scripts/sanity_check.py --image data/images/<frame>.jpg
                                                     │
                                                     ▼
[4] Run the latent attack + baselines on a folder of frames
        python scripts/run_attack.py    --input data/images --output results/adv_latent
        python baselines/fgsm.py        --input data/images --output results/adv_fgsm
        python baselines/pgd_pixel.py   --input data/images --output results/adv_pgd
                                                     │
                                                     ▼
[5] Evaluate (DFR, ASR, mean confidence drop, PSNR_mask)
        python scripts/evaluate.py --clean data/images --adv results/adv_latent \
                                   --out results/metrics_latent.json
```

## Quick start (assuming weights already trained)

```bash
# 1. install (Python 3.10+ recommended; CUDA-capable GPU strongly recommended)
pip install -r requirements.txt

# 2. drop a few test frames in data/images/ (jpg or png)

# 3. sanity check (verifies detector + VAE + masks + paste-back are wired up)
python scripts/sanity_check.py --image data/images/your_image.jpg

# 4. run the latent attack on a folder of images
python scripts/run_attack.py \
    --input data/images \
    --output results/adv_latent \
    --config configs/default.yaml

# 5. run the pixel-space baselines for comparison
python baselines/fgsm.py --input data/images --output results/adv_fgsm
python baselines/pgd_pixel.py --input data/images --output results/adv_pgd

# 6. evaluate (DFR, ASR, mAP drop)
python scripts/evaluate.py \
    --clean data/images \
    --adv results/adv_latent \
    --out results/metrics_latent.json
```

## Training YOLOv8 on UA-DETRAC

If your dataset is UA-DETRAC (XML annotations per `MVI_*` sequence), the
recommended path is Colab Pro (the dataset is large and benefits from a
T4/L4/A100 GPU). See `notebooks/train_yolov8_detrac.ipynb` for the full
workflow. Local-GPU users can use `tools/train_yolov8.py` instead.

The pipeline is:

1. Convert DETRAC → YOLO format with `tools/detrac_to_yolo.py` (one-time).
2. Train `yolov8n.pt` for 80 epochs with mosaic augmentation, early stopping
   on val mAP.
3. Save `best.pt` and load it in `configs/default.yaml` for the attack.

See `dataset/README.md` for the dataset folder layout.

## Repository layout

```
minimal_research/
├── configs/
│   └── default.yaml         # all hyperparameters
├── src/
│   ├── detector.py          # YOLOv8 wrapper (frozen, differentiable head)
│   ├── vae.py               # Stable-Diffusion VAE wrapper (frozen)
│   ├── masks.py             # bbox -> pixel mask -> latent mask
│   ├── losses.py            # vanishing detection loss + L2 perceptual + reg
│   ├── attack.py            # LatentObjectAttack (Adam + projection)
│   ├── data.py              # folder image loader
│   └── utils.py             # IoU, visualization, IO helpers
├── baselines/
│   ├── fgsm.py              # pixel-space FGSM
│   └── pgd_pixel.py         # pixel-space PGD (full image and mask-restricted)
├── scripts/
│   ├── sanity_check.py      # 3-check verification before scaling up
│   ├── run_attack.py        # batch latent attack
│   └── evaluate.py          # DFR, ASR, mAP drop
├── tools/
│   ├── detrac_to_yolo.py    # DETRAC XML -> YOLO format (with train/val split)
│   └── train_yolov8.py      # local-GPU training wrapper
├── notebooks/
│   └── train_yolov8_detrac.ipynb   # Colab Pro training notebook
├── dataset/                 # YOLO-format dataset (after conversion)
├── data/images/             # frames to attack go here
├── runs/                    # training runs and best.pt land here
└── results/                 # adversarial images + metrics land here
```

## Hyperparameters (defaults)

| Symbol | Value | Meaning |
|---|---|---|
| `eps_z` | 0.10 | L-inf budget on latent perturbation |
| `gamma` | 0.05 | confidence floor — once `p_c < gamma`, class c stops contributing |
| `lambda_p` | 0.05 | weight of masked L2 perceptual loss |
| `lambda_r` | 1e-3 | weight of latent-magnitude regularizer |
| `lr` | 0.01 | Adam learning rate |
| `num_steps` | 80 | optimization steps |
| `conf_thr` | 0.25 | YOLOv8 confidence threshold for `D_clean` |

All adjustable in `configs/default.yaml`.

## Notes

- The Stable-Diffusion VAE is loaded from `stabilityai/sd-vae-ft-mse` via
  `diffusers`. First run downloads ~335 MB.
- YOLOv8 weights (`yolov8n.pt`) are downloaded by `ultralytics` on first run.
- This repository implements **only digital attacks** on **YOLOv8**. No physical
  attacks, no other detectors, no diffusion denoising.
