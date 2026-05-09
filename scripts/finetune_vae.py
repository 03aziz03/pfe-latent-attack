"""Fine-tune the SD-VAE on DETRAC surveillance frames.

Goal: improve reconstruction quality on letterboxed 640×640 frames, reducing
the base reconstruction error that currently dominates PSNR_mask.

Design choice — fine-tune ENCODER + DECODER jointly:
  The attack caches z = encode(x) with @no_grad, so changing encoder weights
  changes the cached latent. After Phase 2, the fine-tuned checkpoint is frozen
  for the Phase 4 headline run. Saving vae.vae.state_dict() captures both
  encoder and decoder so the attack can reload it via SDVAE(finetuned_weights=…).

Loss: 0.7 * MSE + 0.3 * LPIPS  (standard VAE domain adaptation recipe).

Usage:
    python scripts/finetune_vae.py \\
        --data data/images_50 \\
        --output runs/vae_detrac \\
        --epochs 15 \\
        --lr 1e-5 \\
        --batch_size 4

    Note: --batch_size 4 requires ~16 GB VRAM (Colab L4).
          Use --batch_size 2 if you have less than 16 GB.

Expected runtime: ~2 hours on Colab L4 for 15 epochs × 50 images (~8 units).
"""
from __future__ import annotations

import argparse
import json
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
    """Loads DETRAC JPG frames, letterboxes to 640×640, returns float32 [0,1]."""

    def __init__(self, data_dir: str | Path):
        self.paths = sorted(Path(data_dir).glob("*.jpg"))
        if not self.paths:
            raise FileNotFoundError(f"No .jpg files found in {data_dir}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = cv2.imread(str(self.paths[idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_lb, _, _ = letterbox_image(img, target=(640, 640))
        arr = img_lb.astype(np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)  # (3, H, W)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load VAE (base weights; fine-tune both encoder and decoder jointly)
    vae = SDVAE(device=device, dtype=torch.float32)
    # Unfreeze all parameters for fine-tuning
    for p in vae.vae.parameters():
        p.requires_grad_(True)
    vae.vae.train()

    # LPIPS loss (unmasked, full image)
    import lpips as lpips_lib
    lpips_fn = lpips_lib.LPIPS(net="alex", verbose=False).to(device)
    for p in lpips_fn.parameters():
        p.requires_grad_(False)

    dataset = DETRACDataset(args.data)
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=0, pin_memory=(device == "cuda"))

    optimizer = AdamW(vae.vae.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    per_epoch_loss: list[float] = []

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        n_batches = 0
        for x in tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}", leave=False):
            x = x.to(device)
            x_recon = vae.decode(vae.encode_with_grad(x))

            # rescale to [-1,1] for LPIPS
            x_01 = x.clamp(0, 1)
            x_recon_01 = x_recon.clamp(0, 1)
            x_11 = x_01 * 2.0 - 1.0
            x_recon_11 = x_recon_01 * 2.0 - 1.0

            mse = F.mse_loss(x_recon_01, x_01)
            lpips_val = lpips_fn(x_recon_11, x_11).mean()
            loss = 0.7 * mse + 0.3 * lpips_val

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            n_batches += 1

        scheduler.step()
        mean_loss = epoch_loss / max(n_batches, 1)
        per_epoch_loss.append(mean_loss)
        print(f"  Epoch {epoch + 1:3d}/{args.epochs}  loss={mean_loss:.6f}")

    # Save checkpoint
    ckpt_path = out_dir / "vae_ft.pt"
    torch.save(vae.vae.state_dict(), ckpt_path)
    print(f"Checkpoint saved to {ckpt_path}")

    # Save training metadata sidecar
    meta = {
        "num_epochs": args.epochs,
        "final_loss": per_epoch_loss[-1] if per_epoch_loss else None,
        "per_epoch_loss": per_epoch_loss,
    }
    meta_path = out_dir / "ft_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved to {meta_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fine-tune SD-VAE on DETRAC frames. "
                    "Use --batch_size 2 if VRAM < 16 GB."
    )
    ap.add_argument("--data", default="data/images_50",
                    help="Directory of .jpg DETRAC frames")
    ap.add_argument("--output", default="runs/vae_detrac",
                    help="Output directory for checkpoint and metadata")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch_size", type=int, default=4,
                    help="Batch size (default 4 for 16 GB VRAM; use 2 for <16 GB)")
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
