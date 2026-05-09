"""Stable-Diffusion VAE wrapper (frozen).

We use the encoder/decoder from `stabilityai/sd-vae-ft-mse`, ignoring the
text encoder and U-Net entirely. This gives us a ~8x spatial downsampling
into a 4-channel latent space whose decoder is biased toward natural images.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from diffusers import AutoencoderKL


class SDVAE(nn.Module):
    """Frozen Stable-Diffusion VAE.

    Conventions:
        * Pixel input is in [0, 1]; we internally rescale to [-1, 1] for
          the SD VAE (which was trained on that range).
        * Latents are scaled by ``scale`` (default 0.18215, the SD value)
          so that downstream perturbation budgets are on the standard scale.
    """

    def __init__(self,
                 model_id: str = "stabilityai/sd-vae-ft-mse",
                 scale: float = 0.18215,
                 device: str = "cuda",
                 dtype: torch.dtype = torch.float32,
                 finetuned_weights: str | None = None):
        super().__init__()
        self.device = torch.device(device)
        self.dtype = dtype
        self.scale = scale
        self.vae: AutoencoderKL = AutoencoderKL.from_pretrained(model_id).to(
            self.device, dtype=dtype
        )
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad_(False)
        if finetuned_weights and Path(finetuned_weights).exists():
            state = torch.load(finetuned_weights, map_location=self.device)
            self.vae.load_state_dict(state)
            print(f"[SDVAE] Loaded fine-tuned weights from {finetuned_weights}")

    # ------------------------------------------------------------------
    # encode / decode
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Pixel image in [0,1] -> scaled latent (B, 4, H/8, W/8)."""
        x = x.to(self.device, self.dtype)
        x = x * 2.0 - 1.0
        latent = self.vae.encode(x).latent_dist.mean
        return latent * self.scale

    def encode_with_grad(self, x: torch.Tensor) -> torch.Tensor:
        """Like encode() but allows gradient flow (for VAE fine-tuning only).

        Do NOT use in the attack loop -- encode() with @no_grad is correct there.
        """
        x = x.to(self.device, self.dtype) * 2.0 - 1.0
        latent = self.vae.encode(x).latent_dist.mean
        return latent * self.scale

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Scaled latent -> pixel image in [0,1] (autograd-friendly).

        We do NOT wrap in no_grad here because the attack needs gradients
        to flow back through the decoder.
        """
        z = z.to(self.device, self.dtype) / self.scale
        x = self.vae.decode(z).sample
        x = (x + 1.0) / 2.0
        return x.clamp(0.0, 1.0)
