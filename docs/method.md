# Method — Object-Aware Latent Adversarial Attack on YOLOv8

This document gives the full mathematical formulation of the attack
implemented in `src/attack.py`. The README contains a one-paragraph
summary; this is the formal version used in the report.

---

## 1. Notation

| Symbol | Space | Meaning |
|---|---|---|
| `x` | `R^{3 x H x W}`, values in `[0, 1]` | clean RGB image |
| `f` | `f : R^{3 x H x W} -> R^{N x (4 + C)}` | frozen YOLOv8 detector (pre-NMS head) |
| `E`, `D` | SD-VAE encoder / decoder | frozen Stable-Diffusion VAE |
| `z = E(x)` | `R^{4 x H/8 x W/8}` | latent encoding of `x` |
| `delta` | `R^{4 x H/8 x W/8}` | learnable latent perturbation |
| `M` | `{0, 1}^{H x W}` | pixel mask = union of detected boxes |
| `M_z` | `{0, 1}^{H/8 x W/8}` | latent mask (pooled `M`) |
| `D_clean` | set of detections | YOLOv8 outputs on `x` after NMS, conf > `tau` |
| `C_clean` | subset of class indices `{0, ..., C-1}` | classes that appear in `D_clean` |
| `eps_z` | scalar | L-infinity budget on `delta` |
| `gamma` | scalar in `(0, 1)` | confidence floor in the vanishing loss |
| `tau` | scalar | NMS confidence threshold (default `0.25`) |

Image tensors are in `[0, 1]`; the VAE expects inputs in `[-1, 1]`, so
`E` and `D` internally apply the affine `2x - 1` and `(y + 1)/2`.

---

## 2. Threat model

- **White-box, digital, untargeted (per class).** The attacker has full
  access to the frozen detector `f` and the frozen VAE `(E, D)`. No
  weights are updated. The attacker observes `D_clean` once and tries to
  drive every class in `C_clean` below the detection threshold.
- **Image-conditional, instance-agnostic.** The perturbation `delta` is
  optimized per image. We do not require persistent identities across
  frames; success is measured frame-by-frame.
- **Locality.** The pixel-space change is restricted to `M` (the union
  of clean boxes) by a paste-back operator. The latent change is
  restricted to `M_z` by elementwise masking of `delta`.
- **No physical realizability.** This is a digital attack; we make no
  claim about printability, illumination invariance, or perspective
  robustness.

---

## 3. Bounding-box masks

Let `D_clean = { (b_i, c_i, p_i) }_{i=1..K}` with `b_i = (x1, y1, x2, y2)`
in pixel coordinates. The pixel mask is the union of clean boxes:

```
M[h, w] = 1   if exists i s.t. (h, w) in b_i
          0   otherwise
```

The latent mask is obtained by an 8x8 max-pool with stride 8 — i.e. a
latent cell is "inside" the perturbable region iff *any* of its 8x8
pixel cells lies in `M`:

```
M_z = MaxPool2d(kernel=8, stride=8)(M)
```

This guarantees that every pixel inside a clean box is covered by at
least one perturbable latent cell, while keeping the perturbable region
as tight as possible.

`M` and `M_z` are computed once per image and held constant for the rest
of the optimization.

---

## 4. Forward pass (clean and adversarial)

Encoding is done once, with no gradient:

```
z = E(x)        # detached
```

Given a perturbation `delta`, the adversarial latent and decoded image
are

```
z_adv     = z + (M_z .* delta)
x_dec     = D(z_adv)                            # autograd-friendly
x_adv     = M .* x_dec + (1 - M) .* x           # paste-back
```

The paste-back operator on the last line is the key locality device:
pixels outside the union of clean boxes are byte-identical to the input.
Only pixels inside `M` can change, regardless of how `D` reconstructs
the rest of the image.

The adversarial detector output (pre-NMS) is

```
y = f(x_adv)  in  R^{N x (4 + C)}
```

where the last `C` channels of each anchor are class confidences in
`[0, 1]` (post-sigmoid in YOLOv8's head).

---

## 5. Vanishing loss (class-level max-confidence)

For each class `c` originally present in `C_clean`, define its image-level
max confidence

```
p_c(x_adv) = max_{n in 1..N} y[n, 4 + c]
```

The vanishing loss penalizes any class whose max confidence is still
above the floor `gamma`:

```
L_vanish(x_adv) = sum_{c in C_clean}  max( p_c(x_adv) - gamma, 0 )
```

This formulation is intentionally **class-level** rather than
**instance-level**: we do not match adversarial anchors to clean boxes
via IoU, we simply require that no anchor anywhere in the image
predicts class `c` with confidence above `gamma`. This sidesteps the
non-differentiability of NMS and the brittleness of IoU matching, and
empirically suffices to remove every detection of every targeted class
under standard YOLOv8 NMS at threshold `tau >= gamma`.

The hinge with floor `gamma` (rather than a plain `p_c` term) prevents
the optimizer from wasting budget pushing already-suppressed classes
further down.

---

## 6. Perceptual and regularization terms

To keep the perturbation visually unobtrusive we add a masked L2 term in
pixel space, normalized by the masked area:

```
L_perc(x_adv, x) = ||M .* (x_adv - x)||_2^2  /  (3 * ||M||_1)
```

`||M||_1` is the number of foreground pixels; the factor `3` accounts
for RGB channels. Normalizing makes `lambda_p` interpretable
independently of how much of the frame is covered by boxes.

We also penalize the magnitude of `delta` directly — a soft prior that
keeps latents near the clean encoding even when the budget is loose:

```
L_reg(delta) = ||M_z .* delta||_2^2 / ||M_z||_1
```

---

## 7. Objective

```
L(delta) = L_vanish(x_adv)
         + lambda_p * L_perc(x_adv, x)
         + lambda_r * L_reg(delta)
```

with `x_adv` defined in section 4 as a function of `delta`.

Default weights (see `configs/default.yaml`): `lambda_p = 0.05`,
`lambda_r = 1e-3`. The vanishing term is the only term that drives the
attack; the other two are taste-makers.

---

## 8. Constraint and projection

The perturbation lives in an L-infinity ball of radius `eps_z` in latent
space:

```
delta in B_inf(eps_z) = { d  :  ||d||_inf <= eps_z }
```

We additionally restrict `delta` to the latent mask `M_z` (zero outside
the box footprint, all the time).

After every Adam step we project:

```
delta <- M_z .* clip(delta, -eps_z, +eps_z)
```

We do **not** clip `x_adv` to `[0, 1]` between steps — the loss already
penalizes deviation from `x` and the paste-back operator only affects
pixels inside `M`. We do clip `x_adv` to `[0, 1]` once at the end before
saving.

---

## 9. Optimization

```
Initialize delta = 0
Encode z = E(x), detach
Compute M, M_z, D_clean, C_clean
Optimizer = Adam(delta, lr = lr)

repeat for t = 1..T:
    z_adv = z + M_z * delta
    x_dec = D(z_adv)
    x_adv = M * x_dec + (1 - M) * x
    y     = f(x_adv)

    L = L_vanish(y) + lambda_p * L_perc(x_adv, x) + lambda_r * L_reg(delta)

    L.backward()
    Adam.step()
    delta <- M_z * clip(delta, -eps_z, +eps_z)

    if all p_c < gamma for c in C_clean:
        break               # early stop on full vanishing
```

Defaults: `T = 80` steps, `lr = 0.01`, `eps_z = 0.10`, `gamma = 0.05`.
Adam is preferred over SGD because the gradient magnitude through
`f . D` varies sharply across latent channels.

---

## 10. Why latent-space perturbation?

Three reasons, each motivated by a known weakness of pixel-space
attacks (FGSM, PGD):

1. **Built-in image prior.** The decoder `D` is a learned manifold of
   natural-looking images. Updates that move `z` in a generic direction
   tend to decode to plausible textures, not high-frequency salt-and-
   pepper noise. This is what gives latent attacks their characteristic
   smoother appearance at comparable detection-failure rates.
2. **Object-locality at the right granularity.** YOLOv8's stride at the
   first detection scale is 8, exactly matching the SD-VAE downscaling
   factor. A single latent cell corresponds to a single 8x8 image patch
   — the scale at which YOLO actually makes its decisions. Restricting
   `delta` to `M_z` therefore restricts the attack to the exact
   receptive-field cells responsible for the boxes we want to remove.
3. **Decoupling the budget from pixel intensities.** Pixel-space `eps`
   is a blunt instrument: at `eps = 8/255` even a "small" perturbation
   over a large box is visually obvious as noise. A latent budget with
   `eps_z = 0.10` produces pixel-space changes that the decoder spreads
   over a structured texture, and the masked L2 term keeps the
   visible part bounded explicitly.

---

## 11. Differences from the IoU-matched formulation

An earlier draft of this method matched each adversarial anchor to the
nearest clean box by IoU and minimized that anchor's confidence. We
replaced this by the class-level max for three reasons:

- **Differentiability.** `argmax` over anchors is sub-differentiable;
  taking the max over the per-class confidence channel is the standard
  trick used in YOLO adversarial work and gives clean gradients.
- **NMS independence.** The class-level loss does not depend on which
  anchors survive NMS, so the attack target does not drift between
  optimization steps as different anchors light up.
- **Empirical equivalence.** On DETRAC, both formulations converged to
  the same DFR (1.0) and similar PSNR-mask, but the class-level version
  converges in fewer steps and is materially simpler.

---

## 12. Evaluation metrics (defined here, computed in `scripts/evaluate.py`)

| Metric | Definition |
|---|---|
| **DFR** (detection-failure rate) | fraction of frames with `D_adv = empty set`, where `D_adv = NMS(f(x_adv), tau, iou_thr)` |
| **ASR** (attack success rate) | fraction of frames in which every class originally present is fully removed (i.e. `C_clean ∩ classes(D_adv) = empty set`) |
| **mean conf drop** | average over `c in C_clean` of `p_c(x) - p_c(x_adv)` |
| **PSNR_mask** | PSNR computed only over pixels inside `M`, in dB; higher = stealthier |
| **masked L2** | `||M .* (x_adv - x)||_2 / sqrt(3 * ||M||_1)` — average per-pixel L2 inside the mask |

For aggregate reporting on a set of frames we report the mean of each
metric and the standard error, plus a pixel-budget-matched comparison
against FGSM and PGD baselines (`baselines/fgsm.py`,
`baselines/pgd_pixel.py`).

---

## 13. Reproducibility

The full pipeline is deterministic up to CUDA non-determinism in the
detector backbone:

- Seeds are set in `src/utils.set_seed` for `torch`, `numpy`, and
  Python's `random`.
- VAE weights: `stabilityai/sd-vae-ft-mse` (frozen).
- Detector weights: `runs/yolov8n_detrac/best.pt` (trained on DETRAC,
  see `notebooks/train_yolov8_detrac.ipynb`).
- All hyperparameters live in `configs/default.yaml`.
- Per-image attack metadata (initial detections, final detections,
  number of steps, final loss components) is written to
  `results/<run>/metadata/<image>.json` by `scripts/run_attack.py`.

---

## 14. Limitations

- **No physical-world claims.** The attack is digital; printing
  `x_adv - x` and pasting it onto a real vehicle is out of scope.
- **No transfer claims.** We attack the specific YOLOv8n detector
  trained on DETRAC. Transfer to other detectors (e.g. YOLOv5,
  RT-DETR) or other domains is not evaluated.
- **NMS dependence.** "Detection failure" is defined relative to the
  configured `(tau, iou_thr)`. A defender who lowers `tau` may recover
  some objects.
- **Per-image cost.** Optimization is ~80 backward passes through
  `f . D`, which is two orders of magnitude more expensive than FGSM.
  Real-time use is not the goal.
