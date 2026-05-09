# Claude Code — Phase 2: MaskedLPIPS + VAE Fine-tune + Iso-budget Sweep

Phase 2 adds the perceptual loss, fine-tunes the SD-VAE on DETRAC footage,
runs an iso-budget sweep across latent and pixel budgets, and produces the
first real Pareto curve (DFR_strict_proportional vs masked LPIPS).

All GPU-heavy work (fine-tune + sweep) runs in Colab Pro via a notebook
produced here. All source code changes are local (no GPU needed to write them).

---

## PROMPT

```
Phase 2: MaskedLPIPS perceptual loss, SD-VAE fine-tune on DETRAC,
iso-budget sweep, Pareto curve.

# Codebase snapshot

Relevant files (read these before writing anything):

  src/attack.py          — LatentObjectAttack, AttackConfig, AttackResult
  src/losses.py          — vanishing_loss, masked_l2, latent_l2
  src/vae.py             — SDVAE (frozen encoder+decoder, float32)
  configs/default.yaml   — all hyperparameters
  scripts/run_attack.py  — DO NOT MODIFY (batch runner, references AttackConfig)

Key facts:
  - L_perc is currently masked_l2(x_adv, x, M) in src/attack.py line ~60
  - AttackConfig has lambda_p (weight) and lambda_r (latent regularizer)
  - SDVAE.encode() is @torch.no_grad(); SDVAE.decode() allows grad flow
  - VAE is frozen: all params requires_grad_(False) in __init__
  - images are (1, 3, 640, 640) in [0, 1], device=cuda
  - dev set for sweep = data/images_50/img00001..img00030 (first 30 frames)

============================================================
PART A — MaskedLPIPS loss
============================================================

## A.1 Install dependency

The `lpips` package is needed. Add to requirements.txt:
    lpips>=0.1.4

Do NOT install it now (no GPU shell here). It will be installed in the
Colab notebook. Just add it to requirements.txt.

## A.2 src/losses.py — add MaskedLPIPS class

Add the following class at the bottom of src/losses.py. Do NOT remove or
modify existing functions (vanishing_loss, masked_l2, latent_l2).

```python
class MaskedLPIPS(torch.nn.Module):
    """Masked perceptual loss using LPIPS (AlexNet backbone).

    Computes LPIPS between x_adv and x after zeroing pixels outside the
    bounding-box mask. The result is normalized by the mask area fraction
    so that lambda_p stays comparable across images with different object sizes.

    Args:
        net:    LPIPS backbone, one of 'alex' (recommended), 'vgg', 'squeeze'.
        device: torch device string.
    """

    def __init__(self, net: str = "alex", device: str = "cuda"):
        super().__init__()
        import lpips  # lazy import so the rest of the codebase works without it
        self._fn = lpips.LPIPS(net=net, verbose=False).to(device)
        for p in self._fn.parameters():
            p.requires_grad_(False)

    def forward(
        self,
        x_adv: torch.Tensor,   # (1, 3, H, W) in [0, 1]
        x: torch.Tensor,        # (1, 3, H, W) in [0, 1]
        M: torch.Tensor,        # (1, 1, H, W) binary mask
    ) -> torch.Tensor:
        """Return masked LPIPS scalar."""
        # zero out non-mask regions in both images
        x_adv_m = x_adv * M
        x_m = x * M
        # LPIPS expects [-1, 1]
        x_adv_m = x_adv_m * 2.0 - 1.0
        x_m = x_m * 2.0 - 1.0
        loss = self._fn(x_adv_m, x_m)          # scalar or (1,1,1,1)
        loss = loss.squeeze()
        # normalize by mask fraction so loss scale is independent of bbox size
        mask_frac = M.mean().clamp(min=1e-4)
        return loss / mask_frac
```

## A.3 src/attack.py — integrate MaskedLPIPS

Modify AttackConfig to add two fields:
    use_lpips: bool = False          # if True, replace masked_l2 with MaskedLPIPS
    lpips_net: str = "alex"          # backbone passed to MaskedLPIPS

Modify LatentObjectAttack.__init__ to conditionally build the loss:
    from .losses import MaskedLPIPS
    self._lpips: MaskedLPIPS | None = None
    if self.cfg.use_lpips:
        self._lpips = MaskedLPIPS(net=self.cfg.lpips_net,
                                   device=str(vae.device))

In the optimization loop, replace:
    L_perc = masked_l2(x_adv, x, M)
with:
    if self._lpips is not None:
        L_perc = self._lpips(x_adv, x, M)
    else:
        L_perc = masked_l2(x_adv, x, M)

No other changes to the attack loop. The rest of AttackConfig, AttackResult,
and the optimization logic stays identical.

## A.4 tests/test_lpips_loss.py — at least 4 tests

Write tests that:
1. MaskedLPIPS(net='alex') constructs without error (skip if lpips not installed
   using pytest.importorskip('lpips'))
2. Forward pass on random (1,3,64,64) inputs returns a scalar tensor
3. x_adv == x → loss ≈ 0 (within 1e-3)
4. x_adv != x → loss > 0
5. Mask of all-zeros → loss ≈ 0 (no foreground → no penalty)

Use pytest.importorskip('lpips') at module level so the tests are skipped
gracefully on machines without lpips installed.

============================================================
PART B — VAE fine-tune script
============================================================

## B.1 scripts/finetune_vae.py

Write a self-contained fine-tune script. Read this section fully before
writing — there are several non-obvious choices documented here.

### Goal
Fine-tune the SD-VAE (stabilityai/sd-vae-ft-mse) on DETRAC surveillance
frames so that reconstruction quality on letterboxed 640×640 frames improves.
This reduces the base reconstruction error that currently dominates PSNR_mask.

### What to fine-tune
Fine-tune ENCODER + DECODER jointly. Reasoning: the attack caches z from
encode() with @no_grad, so the encoded latent changes if the encoder changes.
After Phase 2, the fine-tuned VAE is frozen for the Phase 4 headline run.
The checkpoint captures both encoder and decoder so the attack can reload it.
Document this choice as a comment in the script.

### Data
Use data/images_50/*.jpg (50 frames) for fine-tuning. This is intentionally
small — we want fast domain adaptation, not full retraining.
Letterbox each image to 640×640 using src/viz/letterbox.py::letterbox_image
before feeding to the VAE (same preprocessing as the attack).

### Loss
Reconstruction loss = 0.7 * MSE(x_recon, x) + 0.3 * LPIPS(x_recon, x)
(where LPIPS is unmasked — the full image).
This mixed loss is standard for VAE domain adaptation and avoids the
high-frequency artifacts of pure MSE.

### Training loop
    optimizer = AdamW(vae.parameters(), lr=1e-5, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    for epoch in range(num_epochs):          # default 15
        for x in loader:
            x_recon = vae.decode(vae.encode_with_grad(x))
            loss = 0.7 * F.mse_loss(x_recon, x) + 0.3 * lpips_fn(x_recon, x)
            ...

You will need to add a SDVAE.encode_with_grad() method to src/vae.py
(identical to encode() but without @torch.no_grad). The existing encode()
must stay untouched (attack uses it).

### Output
Save checkpoint to runs/vae_detrac/vae_ft.pt using:
    torch.save(vae.vae.state_dict(), "runs/vae_detrac/vae_ft.pt")

Also save a JSON sidecar runs/vae_detrac/ft_meta.json with:
    {"num_epochs": N, "final_loss": X, "per_epoch_loss": [...]}

### CLI
    python scripts/finetune_vae.py \
        --data data/images_50 \
        --output runs/vae_detrac \
        --epochs 15 \
        --lr 1e-5 \
        --batch_size 4

Batch size 4 fits within 16 GB VRAM on a Colab L4. Add a note in the
help string that users with less VRAM should use --batch_size 2.

### Expected runtime
~2 hours on Colab L4 for 15 epochs × 50 images. ~8 compute units.

## B.2 src/vae.py — add encode_with_grad

Add alongside the existing encode():

    def encode_with_grad(self, x: torch.Tensor) -> torch.Tensor:
        """Like encode() but allows gradient flow (for VAE fine-tuning only).
        Do NOT use in the attack loop — encode() with @no_grad is correct there.
        """
        x = x.to(self.device, self.dtype) * 2.0 - 1.0
        latent = self.vae.encode(x).latent_dist.mean
        return latent * self.scale

## B.3 configs/phase2.yaml — new config file

Create configs/phase2.yaml inheriting defaults and overriding:

    # Phase 2 — LPIPS loss + fine-tuned VAE
    # Copy all fields from default.yaml, then override:

    vae:
      model_id: "stabilityai/sd-vae-ft-mse"
      scale: 0.18215
      finetuned_weights: "runs/vae_detrac/vae_ft.pt"   # loaded if file exists

    attack:
      eps_z: 0.50          # sweep range: 0.25, 0.50, 1.00
      gamma: 0.05
      lambda_p: 0.05
      lambda_r: 1.0e-3
      lr: 0.01
      num_steps: 80
      early_stop: true
      early_stop_margin: 0.0
      use_lpips: true       # Phase 2 default: LPIPS on
      lpips_net: "alex"

    baselines:
      eps_pixel: 0.0314
      pgd_steps: 50
      pgd_alpha: 0.00392
      mask_restricted: true

    runtime:
      device: "cuda"
      dtype: "float32"
      seed: 0

Also update src/vae.py SDVAE.__init__ to accept an optional
finetuned_weights argument. If provided and the path exists, load the
state dict into self.vae after freezing all parameters (so the fine-tuned
weights replace base weights but the model stays frozen for the attack):

    if finetuned_weights and Path(finetuned_weights).exists():
        state = torch.load(finetuned_weights, map_location=self.device)
        self.vae.load_state_dict(state)
        print(f"[SDVAE] Loaded fine-tuned weights from {finetuned_weights}")

Update load_config / AttackConfig / run_attack.py argument parsing to
pass finetuned_weights through if present in the config.

============================================================
PART C — Iso-budget sweep script
============================================================

## C.1 scripts/run_iso_budget.py

This is the main Phase 2 experiment. It sweeps budgets for latent and PGD
attacks on the 30-frame dev set and records DFR + LPIPS for each config.

### Budget grid
    LATENT_EPS  = [0.25, 0.50, 1.00]         # eps_z values
    PGD_EPS     = [4/255, 8/255, 12/255]      # pixel L∞

### Dev set
    data/images_50/img00001.jpg ... img00030.jpg  (first 30 frames only)

### Per config, for each frame:
    - Run the attack (latent or PGD) to get x_adv
    - Compute DFR_strict_proportional (use src/eval/metrics.py)
    - Compute masked LPIPS (use MaskedLPIPS from src/losses.py)
    - Record {eps, dfr, lpips, n_clean, n_adv}

### Output
    results/iso_budget/
        latent_eps0.25.json
        latent_eps0.50.json
        latent_eps1.00.json
        pgd_eps4.json
        pgd_eps8.json
        pgd_eps12.json
        summary.json            <- aggregated mean DFR + mean LPIPS per config

### CLI
    python scripts/run_iso_budget.py \
        --config configs/phase2.yaml \
        --data data/images_50 \
        --n_frames 30 \
        --output results/iso_budget

### Expected runtime on Colab L4
~4 hours total (6 configs × 30 frames × ~80 steps). ~4 compute units.
Add a --resume flag: if a per-config JSON already exists, skip that config.
This makes the script re-runnable after Colab disconnects.

## C.2 scripts/generate_pareto.py

Read results/iso_budget/summary.json and produce:

f14_pareto_dfr_lpips.png/pdf
    x: mean_lpips (lower = more imperceptible)  → label: "Masked LPIPS (↓ better)"
    y: mean_dfr_strict_proportional             → label: "DFR_strict_proportional (↑ better)"
    Points: each budget config, colored by attack type (latent=blue, pgd=orange)
    Markers: circle for latent, square for PGD
    Connect latent points with a dashed line (Pareto frontier, ordered by eps)
    Connect PGD points with a dotted line
    Ideal corner annotation: "↗ ideal" in top-left of plot
    Error bars: 95% CI on DFR from bootstrap (reuse src/eval/bootstrap.py)
    Annotate each point with its eps value

Save to results/figures/png/f14_pareto_dfr_lpips.png (300 dpi) and
results/figures/pdf/f14_pareto_dfr_lpips.pdf.

============================================================
PART D — Colab notebook
============================================================

## D.1 notebooks/phase2_colab.ipynb

Create a complete Colab-ready notebook. Use nbformat to generate it
programmatically (safer than writing raw JSON). Each cell must be clearly
labeled with a markdown header. The notebook must be self-contained and
runnable top-to-bottom without any modification except the REPO_PATH variable.

### Cell structure

Cell 0 — Markdown: "# Phase 2: LPIPS + VAE Fine-tune + Iso-budget Sweep"
  - One-sentence description of each phase
  - Estimated runtime: ~6h total, ~12 Colab Pro compute units
  - Prerequisites: Colab Pro, L4 GPU, Google Drive with project repo

Cell 1 — Code: Mount Google Drive
    from google.colab import drive
    drive.mount('/content/drive')
    REPO_PATH = "/content/drive/MyDrive/YOUR_REPO_NAME"  # <-- edit this
    import os; os.chdir(REPO_PATH)
    print("Working directory:", os.getcwd())

Cell 2 — Code: Install dependencies
    !pip install lpips>=0.1.4 -q
    !pip install -r requirements.txt -q
    # verify GPU
    import torch
    print("CUDA available:", torch.cuda.is_available())
    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

Cell 3 — Markdown: "## Step 1: VAE Fine-tune (~2h, ~8 units)"

Cell 4 — Code: Run VAE fine-tune
    !python scripts/finetune_vae.py \
        --data data/images_50 \
        --output runs/vae_detrac \
        --epochs 15 \
        --lr 1e-5 \
        --batch_size 4
    # Checkpoint saved to runs/vae_detrac/vae_ft.pt

Cell 5 — Code: Plot fine-tune loss curve
    import json, matplotlib.pyplot as plt
    meta = json.load(open("runs/vae_detrac/ft_meta.json"))
    plt.plot(meta["per_epoch_loss"])
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("VAE fine-tune loss")
    plt.tight_layout(); plt.show()

Cell 6 — Markdown: "## Step 2: Iso-budget sweep (~4h, ~4 units)"

Cell 7 — Code: Run sweep
    !python scripts/run_iso_budget.py \
        --config configs/phase2.yaml \
        --data data/images_50 \
        --n_frames 30 \
        --output results/iso_budget
    # Use --resume to continue after a Colab disconnect

Cell 8 — Code: Quick sanity check — print summary table
    import json, pandas as pd
    summary = json.load(open("results/iso_budget/summary.json"))
    df = pd.DataFrame(summary).T
    print(df.to_string())

Cell 9 — Markdown: "## Step 3: Generate Pareto curve"

Cell 10 — Code: Generate Pareto figure
    !python scripts/generate_pareto.py
    from IPython.display import Image
    Image("results/figures/png/f14_pareto_dfr_lpips.png")

Cell 11 — Markdown: "## Done — copy results back to Drive"

Cell 12 — Code: Verify outputs
    import os
    for f in ["runs/vae_detrac/vae_ft.pt",
              "results/iso_budget/summary.json",
              "results/figures/png/f14_pareto_dfr_lpips.png"]:
        status = "✓" if os.path.exists(f) else "✗ MISSING"
        print(f"{status}  {f}")

============================================================
PART E — Tests
============================================================

## E.1 tests/test_lpips_loss.py

As described in A.4. All tests use pytest.importorskip('lpips').

## E.2 tests/test_vae_finetune.py (optional but recommended)

If lpips is importorskip'd:
1. encode_with_grad() returns a tensor with requires_grad-able computation
   (i.e., gradients flow back through it — check with autograd.grad)
2. SDVAE loads fine-tuned weights when finetuned_weights path is valid
3. SDVAE silently ignores finetuned_weights when path doesn't exist

============================================================
PART F — Documentation
============================================================

## F.1 Update PHASE1_DONE.md

Append a "Phase 2 (code-only — Colab run pending)" section:

    ## Phase 2 — LPIPS + VAE Fine-tune + Iso-budget Sweep

    **Status:** Code written. Colab run pending (~12 units).

    ### New files
    - src/losses.py: MaskedLPIPS class added
    - src/vae.py: encode_with_grad() added; finetuned_weights support
    - src/attack.py: use_lpips / lpips_net in AttackConfig; conditional loss
    - scripts/finetune_vae.py: VAE fine-tune on DETRAC
    - scripts/run_iso_budget.py: iso-budget sweep (6 configs × 30 frames)
    - scripts/generate_pareto.py: f14 Pareto curve
    - configs/phase2.yaml: Phase 2 hyperparameters
    - notebooks/phase2_colab.ipynb: end-to-end Colab notebook
    - tests/test_lpips_loss.py

    ### Expected results after Colab run
    - runs/vae_detrac/vae_ft.pt: fine-tuned VAE checkpoint
    - results/iso_budget/summary.json: DFR + LPIPS per config
    - results/figures/png/f14_pareto_dfr_lpips.png: Pareto curve

============================================================
PART G — Constraints and verification
============================================================

# Constraints
- Do NOT modify scripts/run_attack.py
- Do NOT modify the core attack algorithm (optimization loop logic in src/attack.py)
- Do NOT modify existing tests or existing functions in src/losses.py
- MaskedLPIPS must be importable without lpips installed (lazy import inside __init__)
- All new scripts must run to completion on CPU (slower but functional) for local testing
- Notebook cells must be self-contained — no cross-cell variable dependencies
  except REPO_PATH and standard imports

# Verification checklist (run these locally, no GPU needed)
1. python -c "from src.losses import MaskedLPIPS" — no ImportError
2. python -c "from src.attack import LatentObjectAttack, AttackConfig; c = AttackConfig(use_lpips=True); print(c)" — works
3. python -c "from src.vae import SDVAE" — no import error
4. python scripts/finetune_vae.py --help — prints usage
5. python scripts/run_iso_budget.py --help — prints usage
6. python scripts/generate_pareto.py --help — prints usage (or runs and warns about missing data gracefully)
7. pytest tests/test_lpips_loss.py -v — all tests pass or skip (if lpips absent)
8. pytest tests/ -q --ignore=tests/test_lpips_loss.py — 33/33 pass (existing tests unaffected)
9. jupyter nbconvert --to script notebooks/phase2_colab.ipynb — converts without error
10. configs/phase2.yaml is valid YAML (python -c "import yaml; yaml.safe_load(open('configs/phase2.yaml'))")
```

---

## What this builds

| Deliverable | Where | GPU needed? |
|---|---|---|
| MaskedLPIPS loss | src/losses.py | No |
| AttackConfig.use_lpips | src/attack.py | No |
| encode_with_grad() | src/vae.py | No |
| finetune_vae.py | scripts/ | **Yes (Colab)** |
| run_iso_budget.py | scripts/ | **Yes (Colab)** |
| generate_pareto.py | scripts/ | No |
| phase2.yaml | configs/ | No |
| phase2_colab.ipynb | notebooks/ | No (runs in Colab) |
| test_lpips_loss.py | tests/ | No |
| f14 Pareto curve | results/figures/ | No (post-sweep) |

## Budget for the Colab session

| Step | Time | Units |
|---|---|---|
| VAE fine-tune (15 epochs, 50 frames) | ~2h | ~8 |
| Iso-budget sweep (6 configs × 30 frames) | ~4h | ~4 |
| Figure generation | <5 min | <0.5 |
| **Total** | **~6h** | **~12–13** |

Remaining after Phase 2: ~57–58 units (Phase 3 temporal ~9, Phase 4 headline ~10,
transferability ~2, buffer ~22).

## After the Colab run — what to check

1. Fine-tune loss curve is decreasing (not diverging). Final loss < initial loss.
2. summary.json shows latent at eps_z=1.00 dominates PGD at eps=12/255 on DFR.
3. Pareto curve shows latent configs form a frontier above PGD configs.
4. If fine-tuned VAE breaks attack effectiveness (DFR_strict at eps_z=0.50 drops
   below 0.10), roll back to base VAE and note it as a limitation.
