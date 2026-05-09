"""Fine-tune the SD-VAE on DETRAC surveillance frames.

Goal: improve reconstruction quality on letterboxed 640x640 frames, reducing
the base reconstruction error that currently dominates PSNR_mask.

Design choice -- fine-tune ENCODER + DECODER jointly:
  The attack caches z = encode(x) with @no_grad, so changing encoder weights
  changes the cached latent. After Phase 2, the fine-tuned checkpoint is frozen
  for the Phase 4 headline run. Saving vae.vae.state_dict() captures both
  encoder and decoder so the attack can reload it via SDVAE(finetuned_weights=).

Recommended data layout (3 sequences, completely separate from attack eval):

    data/finetune_seqs/
        MVI_XXXXX/          <- sequence A, 60 frames
            img00001.jpg
            ...
            img00060.jpg
        MVI_YYYYY/          <- sequence B, 60 frames
        MVI_ZZZZZ/          <- sequence C, 60 frames

    data/images/            <- attack EVAL set (never used here)
        img00001.jpg
        ...
        img00050.jpg

Train/val split: stratified by sequence -- last val_frac frames of each
sequence go to validation, the rest to training. This ensures val frames
come from all three cameras, giving a meaningful early-stopping signal.

Default: val_frac=0.15 -> ~9 val frames per sequence, ~51 train per sequence.

Augmentation: random horizontal flip + mild colour jitter.
Loss: 0.7 * MSE + 0.3 * LPIPS.
Early stopping: patience=3 epochs on validation loss.

Usage:
    python scripts/finetune_vae.py \\
        --data data/finetune_seqs \\
        --output runs/vae_detrac \\
        --epochs 20 \\
        --lr 1e-5 \\
        --batch_size 4

Expected runtime: ~45 min on Colab L4 for 20 epochs x 180 frames (~3 units).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.vae import SDVAE
from src.viz.letterbox import letterbox_image


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class DETRACDataset(Dataset):
    """Loads DETRAC JPG/PNG frames, letterboxes to 640x640, returns float32 [0,1].

    Supports optional augmentation: horizontal flip + mild colour jitter.
    """

    def __init__(self, paths: list[Path], augment: bool = False, seed: int = 42):
        self.paths = paths
        self.augment = augment
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = cv2.imread(str(self.paths[idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_lb, _, _ = letterbox_image(img, target=(640, 640))
        arr = img_lb.astype(np.float32) / 255.0   # (H, W, 3) in [0,1]

        if self.augment:
            # Random horizontal flip
            if self._rng.random() < 0.5:
                arr = arr[:, ::-1, :].copy()
            # Mild brightness/contrast jitter
            factor = self._rng.uniform(0.85, 1.15)
            arr = np.clip(arr * factor, 0.0, 1.0)

        return torch.from_numpy(arr).permute(2, 0, 1)  # (3, H, W)


def stratified_split(
    data_dir: str | Path,
    val_frac: float = 0.15,
    seed: int = 42,
) -> tuple[list[Path], list[Path]]:
    """Split images into (train, val) stratified by subdirectory (sequence).

    For each subdirectory (one DETRAC sequence), the last val_frac fraction
    of frames (sorted by name) go to val; the rest go to train.

    If data_dir is flat (no subdirectories), all images are treated as one group.
    """
    data_dir = Path(data_dir)

    # Collect per-sequence groups
    subdirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    if not subdirs:
        # Flat directory -- treat as a single sequence
        subdirs = [data_dir]

    train_paths: list[Path] = []
    val_paths: list[Path] = []

    for seq_dir in subdirs:
        frames = sorted(
            list(seq_dir.rglob("*.jpg")) + list(seq_dir.rglob("*.png"))
        )
        if not frames:
            continue
        n_val = max(1, round(len(frames) * val_frac))
        val_paths.extend(frames[-n_val:])   # last n_val frames -> val
        train_paths.extend(frames[:-n_val]) # everything else -> train

    if not train_paths and not val_paths:
        raise FileNotFoundError(f"No image files found under {data_dir}")

    rng = random.Random(seed)
    rng.shuffle(train_paths)

    return train_paths, val_paths


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- split ---
    train_paths, val_paths = stratified_split(args.data, val_frac=args.val_frac)

    print(f"Fine-tune data : {args.data}")
    print(f"Train frames   : {len(train_paths)}")
    print(f"Val frames     : {len(val_paths)}")
    print(f"Attack eval    : separate (data/images_50) -- not touched here")

    train_ds = DETRACDataset(train_paths, augment=True)
    val_ds   = DETRACDataset(val_paths,   augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=2,
                              pin_memory=(device == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0,
                              pin_memory=(device == "cuda"))

    # --- model ---
    vae = SDVAE(device=device, dtype=torch.float32)
    for p in vae.vae.parameters():
        p.requires_grad_(True)
    vae.vae.train()

    # --- LPIPS ---
    import lpips as lpips_lib
    lpips_fn = lpips_lib.LPIPS(net="alex", verbose=False).to(device)
    for p in lpips_fn.parameters():
        p.requires_grad_(False)

    # --- optimiser ---
    optimizer = AdamW(vae.vae.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    per_epoch_train_loss: list[float] = []
    per_epoch_val_loss:   list[float] = []
    best_val_loss = float("inf")
    patience_counter = 0
    best_ckpt_path = out_dir / "vae_ft.pt"

    for epoch in range(args.epochs):
        # --- train ---
        vae.vae.train()
        epoch_loss, n_batches = 0.0, 0
        for x in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [train]",
                      leave=False):
            x = x.to(device)
            x_recon = vae.decode(vae.encode_with_grad(x))
            x_01        = x.clamp(0, 1)
            x_recon_01  = x_recon.clamp(0, 1)
            mse      = F.mse_loss(x_recon_01, x_01)
            lp_val   = lpips_fn(x_recon_01 * 2 - 1, x_01 * 2 - 1).mean()
            loss     = 0.7 * mse + 0.3 * lp_val

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            n_batches  += 1

        scheduler.step()
        mean_train = epoch_loss / max(n_batches, 1)
        per_epoch_train_loss.append(mean_train)

        # --- validate ---
        vae.vae.eval()
        val_loss, n_val = 0.0, 0
        with torch.no_grad():
            for x in val_loader:
                x = x.to(device)
                x_recon = vae.decode(vae.encode(x))
                x_01       = x.clamp(0, 1)
                x_recon_01 = x_recon.clamp(0, 1)
                mse    = F.mse_loss(x_recon_01, x_01)
                lp_val = lpips_fn(x_recon_01 * 2 - 1, x_01 * 2 - 1).mean()
                val_loss += float((0.7 * mse + 0.3 * lp_val).item())
                n_val    += 1
        mean_val = val_loss / max(n_val, 1)
        per_epoch_val_loss.append(mean_val)

        print(f"  Epoch {epoch+1:3d}/{args.epochs}  "
              f"train={mean_train:.6f}  val={mean_val:.6f}  "
              f"best={best_val_loss:.6f}")

        # --- early stopping ---
        if mean_val < best_val_loss:
            best_val_loss = mean_val
            patience_counter = 0
            torch.save(vae.vae.state_dict(), best_ckpt_path)
            print(f"    --> Best val loss. Checkpoint saved.")
        else:
            patience_counter += 1
            print(f"    --> No improvement ({patience_counter}/{args.patience})")
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch+1}.")
                break

    print(f"\nBest checkpoint: {best_ckpt_path}  (val={best_val_loss:.6f})")

    meta = {
        "num_epochs_run": len(per_epoch_train_loss),
        "best_val_loss": best_val_loss,
        "final_train_loss": per_epoch_train_loss[-1],
        "per_epoch_train_loss": per_epoch_train_loss,
        "per_epoch_val_loss": per_epoch_val_loss,
        "n_train_frames": len(train_paths),
        "n_val_frames": len(val_paths),
        "data_dir": str(args.data),
    }
    with open(out_dir / "ft_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata: {out_dir / 'ft_meta.json'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Fine-tune SD-VAE on multiple DETRAC sequences. "
            "Expected layout: --data dir with one subdirectory per sequence. "
            "Use --batch_size 2 if VRAM < 16 GB."
        )
    )
    ap.add_argument("--data", default="data/finetune_seqs",
                    help="Root dir containing one subdir per DETRAC sequence")
    ap.add_argument("--output", default="runs/vae_detrac")
    ap.add_argument("--epochs",     type=int,   default=20)
    ap.add_argument("--lr",         type=float, default=1e-5)
    ap.add_argument("--batch_size", type=int,   default=4,
                    help="Use 2 if VRAM < 16 GB")
    ap.add_argument("--val_frac",   type=float, default=0.15,
                    help="Fraction of each sequence held out for validation")
    ap.add_argument("--patience",   type=int,   default=3,
                    help="Early stopping patience in epochs")
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
