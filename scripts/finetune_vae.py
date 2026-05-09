"""Fine-tune the SD-VAE decoder on DETRAC surveillance frames.

Goal: improve reconstruction quality on letterboxed 640x640 frames, reducing
the base reconstruction error that currently dominates PSNR_mask.

Design choice -- DECODER ONLY fine-tuning (encoder stays frozen):
  - The attack caches z = encode(x) with @no_grad. Keeping the encoder frozen
    means cached latents remain valid after fine-tuning -- no need to re-run
    the attack from scratch.
  - Decoder-only fine-tuning uses ~half the GPU memory of joint fine-tuning
    (no gradients through encoder activations).
  - Empirically, decoder-only domain adaptation is sufficient: the encoder
    already maps surveillance frames to reasonable latents; the decoder needs
    to learn to reconstruct surveillance-style textures and colours.
  - Use --finetune_encoder to enable joint fine-tuning if needed (requires
    more VRAM; use batch_size=1).

Mixed precision: bfloat16 forward pass, float32 optimizer state.
  Cuts activation memory by ~2x. bfloat16 preferred over float16 for
  diffusers models (avoids inf/nan in GroupNorm).

Data layout (3 sequences, separate from attack eval set):

    data/finetune_seqs/
        MVI_XXXXX/    <- 60 frames from sequence A
        MVI_YYYYY/    <- 60 frames from sequence B
        MVI_ZZZZZ/    <- 60 frames from sequence C

    data/images_50/   <- attack eval set, never touched here

Train/val split: stratified by sequence (val_frac=0.15 -> ~9 val per seq).
Early stopping: patience=3 epochs on validation loss.

Usage:
    python scripts/finetune_vae.py \\
        --data data/finetune_seqs \\
        --output runs/vae_detrac \\
        --epochs 20 \\
        --lr 1e-5 \\
        --batch_size 2

Expected runtime: ~25 min on Colab L4, ~2 Colab Pro units.
"""
from __future__ import annotations

import argparse
import json
import os
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
    """Loads DETRAC frames, letterboxes to 640x640, returns float32 [0,1]."""

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
        arr = img_lb.astype(np.float32) / 255.0

        if self.augment:
            if self._rng.random() < 0.5:
                arr = arr[:, ::-1, :].copy()
            factor = self._rng.uniform(0.85, 1.15)
            arr = np.clip(arr * factor, 0.0, 1.0)

        return torch.from_numpy(arr).permute(2, 0, 1)  # (3, H, W)


def stratified_split(
    data_dir: str | Path,
    val_frac: float = 0.15,
    seed: int = 42,
) -> tuple[list[Path], list[Path]]:
    """Split per subdirectory (sequence), last val_frac frames -> val."""
    data_dir = Path(data_dir)
    subdirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    if not subdirs:
        subdirs = [data_dir]

    train_paths: list[Path] = []
    val_paths:   list[Path] = []

    for seq_dir in subdirs:
        frames = sorted(
            list(seq_dir.rglob("*.jpg")) + list(seq_dir.rglob("*.png"))
        )
        if not frames:
            continue
        n_val = max(1, round(len(frames) * val_frac))
        val_paths.extend(frames[-n_val:])
        train_paths.extend(frames[:-n_val])

    if not train_paths and not val_paths:
        raise FileNotFoundError(f"No image files found under {data_dir}")

    random.Random(seed).shuffle(train_paths)
    return train_paths, val_paths


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def compute_loss(
    x_recon: torch.Tensor,
    x: torch.Tensor,
    lpips_fn,
) -> torch.Tensor:
    """0.7 * MSE + 0.3 * LPIPS, both in [0,1] pixel space."""
    x_01       = x.clamp(0, 1)
    x_recon_01 = x_recon.clamp(0, 1)
    mse    = F.mse_loss(x_recon_01, x_01)
    lp_val = lpips_fn(x_recon_01 * 2 - 1, x_01 * 2 - 1).mean()
    return 0.7 * mse + 0.3 * lp_val


def train(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = (device == "cuda")
    print(f"Device: {device}  |  Mixed precision (bfloat16): {use_amp}")

    # Clear any leftover cache
    if device == "cuda":
        torch.cuda.empty_cache()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- split ---
    train_paths, val_paths = stratified_split(args.data, val_frac=args.val_frac)
    print(f"Fine-tune data : {args.data}")
    print(f"Train frames   : {len(train_paths)}")
    print(f"Val frames     : {len(val_paths)}")
    print(f"Decoder only   : {not args.finetune_encoder}")

    train_ds = DETRACDataset(train_paths, augment=True)
    val_ds   = DETRACDataset(val_paths,   augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=2,
                              pin_memory=(device == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0,
                              pin_memory=(device == "cuda"))

    # --- model: load in float32, freeze everything first ---
    vae = SDVAE(device=device, dtype=torch.float32)
    for p in vae.vae.parameters():
        p.requires_grad_(False)

    # Unfreeze decoder (always) and optionally encoder
    for p in vae.vae.decoder.parameters():
        p.requires_grad_(True)
    if args.finetune_encoder:
        for p in vae.vae.encoder.parameters():
            p.requires_grad_(True)
        print("Fine-tuning encoder + decoder jointly.")
    else:
        print("Fine-tuning decoder only (encoder frozen).")

    vae.vae.train()

    # --- LPIPS (always float32, always frozen) ---
    import lpips as lpips_lib
    lpips_fn = lpips_lib.LPIPS(net="alex", verbose=False).to(device)
    for p in lpips_fn.parameters():
        p.requires_grad_(False)
    lpips_fn.eval()

    # --- optimiser ---
    trainable = [p for p in vae.vae.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler    = torch.amp.GradScaler(enabled=use_amp)

    per_epoch_train_loss: list[float] = []
    per_epoch_val_loss:   list[float] = []
    best_val_loss    = float("inf")
    patience_counter = 0
    best_ckpt_path   = out_dir / "vae_ft.pt"

    for epoch in range(args.epochs):
        # --- train ---
        vae.vae.train()
        epoch_loss, n_batches = 0.0, 0

        for x in tqdm(train_loader,
                      desc=f"Epoch {epoch+1}/{args.epochs} [train]",
                      leave=False):
            x = x.to(device)

            with torch.amp.autocast(device_type=device, dtype=torch.bfloat16,
                                    enabled=use_amp):
                # Encoder: frozen -> no_grad (even if finetune_encoder=False)
                if args.finetune_encoder:
                    z = vae.encode_with_grad(x)
                else:
                    z = vae.encode(x)          # @no_grad inside encode()
                x_recon = vae.decode(z)        # gradients flow through decoder
                loss = compute_loss(x_recon.float(), x.float(), lpips_fn)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

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
                with torch.amp.autocast(device_type=device, dtype=torch.bfloat16,
                                        enabled=use_amp):
                    z      = vae.encode(x)
                    x_recon = vae.decode(z)
                    loss = compute_loss(x_recon.float(), x.float(), lpips_fn)
                val_loss += float(loss.item())
                n_val    += 1

        mean_val = val_loss / max(n_val, 1)
        per_epoch_val_loss.append(mean_val)

        print(f"  Epoch {epoch+1:3d}/{args.epochs}  "
              f"train={mean_train:.6f}  val={mean_val:.6f}  "
              f"best={best_val_loss:.6f}")

        # --- early stopping ---
        if mean_val < best_val_loss:
            best_val_loss    = mean_val
            patience_counter = 0
            torch.save(vae.vae.state_dict(), best_ckpt_path)
            print(f"    --> Best val loss. Checkpoint saved.")
        else:
            patience_counter += 1
            print(f"    --> No improvement ({patience_counter}/{args.patience})")
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch+1}.")
                break

    print(f"\nBest checkpoint : {best_ckpt_path}  (val={best_val_loss:.6f})")

    meta = {
        "num_epochs_run":       len(per_epoch_train_loss),
        "best_val_loss":        best_val_loss,
        "final_train_loss":     per_epoch_train_loss[-1],
        "per_epoch_train_loss": per_epoch_train_loss,
        "per_epoch_val_loss":   per_epoch_val_loss,
        "n_train_frames":       len(train_paths),
        "n_val_frames":         len(val_paths),
        "data_dir":             str(args.data),
        "decoder_only":         not args.finetune_encoder,
    }
    with open(out_dir / "ft_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata : {out_dir / 'ft_meta.json'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Fine-tune SD-VAE decoder on DETRAC frames. "
            "Expected --data layout: one subdirectory per DETRAC sequence. "
            "Use --batch_size 1 if still OOM."
        )
    )
    ap.add_argument("--data",             default="data/finetune_seqs")
    ap.add_argument("--output",           default="runs/vae_detrac")
    ap.add_argument("--epochs",           type=int,   default=20)
    ap.add_argument("--lr",               type=float, default=1e-5)
    ap.add_argument("--batch_size",       type=int,   default=2,
                    help="Default 2 (safe for 22 GB). Use 1 if OOM.")
    ap.add_argument("--val_frac",         type=float, default=0.15)
    ap.add_argument("--patience",         type=int,   default=3)
    ap.add_argument("--finetune_encoder", action="store_true",
                    help="Also fine-tune encoder (more VRAM; use batch_size=1)")
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
