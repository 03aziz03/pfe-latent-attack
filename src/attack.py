"""Object-aware latent adversarial attack on YOLOv8.

Algorithm (single image, batch size 1):

    D_clean = NMS( f(x), conf=conf_thr )
    if no detections: return x
    C       = unique classes in D_clean
    M       = boxes_to_pixel_mask(D_clean)            (1,1,H,W)
    Mz      = pool8(M).expand(-1,4,-1,-1)             (1,4,H/8,W/8)
    z       = E(x)                                     (cached, no grad)
    delta   = zeros_like(z, requires_grad=True)
    optim   = Adam([delta], lr)
    for t in 1..T:
        z_adv   = z + Mz * delta
        x_dec   = D(z_adv)
        x_adv   = M * x_dec + (1-M) * x
        raw     = f(x_adv)                             (B, A, 4 + nc)
        cls_conf = raw[..., 4:]
        L_det   = mean_c [ ReLU(max_a conf[a,c] - gamma)^2 ]
        L_perc  = masked_l2(x_adv, x, M)
        L_reg   = mean(delta^2)
        L = L_det + lambda_p L_perc + lambda_r L_reg
        L.backward()
        optim.step()
        with no_grad: delta.clamp_(-eps_z, eps_z); delta *= Mz
        if early_stop and max_c p_c < gamma: break
    return x_adv

Phase 3 extensions (all backward-compatible, disabled by default):
    3A: objectness-aware loss  (use_objectness, obj_weight)
    3B: MI-Adam momentum       (use_momentum, momentum_decay)
    3C: multi-restart          (n_restarts, restart_noise)
    3D: LPIPS + SSIM           (use_ssim, ssim_weight)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch

from .detector import YOLOv8Wrapper
from .vae import SDVAE
from .masks import boxes_to_pixel_mask, pixel_mask_to_latent_mask
from .losses import (vanishing_loss, objectness_loss, masked_l2, latent_l2,
                     MaskedLPIPS, ssim_loss, perceptual_combined)
from .utils import Detection


@dataclass
class AttackConfig:
    # ---- core ----
    eps_z: float = 0.10
    gamma: float = 0.05
    lambda_p: float = 0.05
    lambda_r: float = 1e-3
    lr: float = 0.01
    num_steps: int = 80
    early_stop: bool = True
    early_stop_margin: float = 0.0
    conf_thr: float = 0.25
    iou_nms: float = 0.45
    use_lpips: bool = False   # if True, replace masked_l2 with MaskedLPIPS
    lpips_net: str = "alex"   # backbone passed to MaskedLPIPS

    # ---- Phase 3A: objectness-aware loss ----
    use_objectness: bool = False
    # Weight of anchor-level objectness term relative to the class-level
    # vanishing term.  A value of 1.0 gives equal weight; 0.5 is the default
    # recommended starting point.
    obj_weight: float = 0.5

    # ---- Phase 3B: MI-Adam momentum (gradient accumulation) ----
    use_momentum: bool = False
    # Decay factor mu for the running gradient average (MI-FGSM style).
    # Applied to the raw gradient *before* it is passed to Adam.
    # Typical range: 0.85-0.95.
    momentum_decay: float = 0.9

    # ---- Phase 3C: multi-restart ----
    # Number of independent starts.  Start 0 initialises delta = 0 (standard).
    # Starts 1..n_restarts-1 draw delta0 from U(-restart_noise*eps_z,
    # +restart_noise*eps_z) masked by Mz.  Best result is returned.
    n_restarts: int = 1
    restart_noise: float = 0.5   # fraction of eps_z used for random init

    # ---- Phase 3D: SSIM perceptual constraint ----
    use_ssim: bool = False
    # Weight of SSIM term inside the combined perceptual loss.  Only used
    # when use_lpips=True (the SSIM term is added on top of LPIPS).
    ssim_weight: float = 0.3


@dataclass
class AttackResult:
    x_adv: torch.Tensor                    # (1, 3, H, W) in [0, 1]
    delta: torch.Tensor                    # (1, 4, H/8, W/8)
    M: torch.Tensor                        # (1, 1, H, W)
    detections_clean: list[Detection]
    classes_clean: list[int]
    history: dict[str, list[float]] = field(default_factory=dict)
    steps_taken: int = 0


class LatentObjectAttack:
    """Object-aware latent adversarial attack.

    Both the detector and VAE are expected to be already loaded; this class
    is stateless apart from the cached references to those models.
    """

    def __init__(self,
                 detector: YOLOv8Wrapper,
                 vae: SDVAE,
                 config: AttackConfig | None = None):
        self.det = detector
        self.vae = vae
        self.cfg = config or AttackConfig()
        self._lpips: MaskedLPIPS | None = None
        if self.cfg.use_lpips:
            self._lpips = MaskedLPIPS(net=self.cfg.lpips_net,
                                      device=str(vae.device))

    # ------------------------------------------------------------------
    # main entry point
    # ------------------------------------------------------------------

    def attack(self, x: torch.Tensor) -> AttackResult:
        """Run the latent attack on a single image.

        Supports Phase 3 extensions: objectness-aware loss (3A), MI-Adam
        momentum (3B), multi-restart (3C), and SSIM perceptual constraint (3D).
        All extensions are disabled by default (backward-compatible).

        Args:
            x: (1, 3, H, W) tensor in [0, 1]; H, W must be multiples of 8.

        Returns:
            AttackResult with the best result across restarts.
        """
        cfg = self.cfg
        device = self.vae.device
        x = x.to(device)
        B, _, H, W = x.shape
        assert B == 1, "Attack expects a single-image batch."
        assert H % 8 == 0 and W % 8 == 0, "H and W must be multiples of 8."

        # ----- 1. clean detections -----
        D_clean = self.det.detect_nms(x, conf_thr=cfg.conf_thr,
                                       iou_thr=cfg.iou_nms)
        if len(D_clean) == 0:
            empty_delta = torch.zeros((1, 4, H // 8, W // 8), device=device)
            empty_M = torch.zeros((1, 1, H, W), device=device)
            return AttackResult(x_adv=x.clone(), delta=empty_delta, M=empty_M,
                                 detections_clean=[], classes_clean=[],
                                 steps_taken=0)
        C_clean = sorted({d.cls for d in D_clean})

        # ----- 2. masks -----
        M = boxes_to_pixel_mask(D_clean, H=H, W=W, device=device)
        Mz = pixel_mask_to_latent_mask(M, latent_channels=4, stride=8)

        # ----- 3. encode (cached, no grad) -----
        z = self.vae.encode(x).detach()

        # ----- 4. multi-restart loop (Phase 3C) -----
        best_result: AttackResult | None = None

        for restart_idx in range(max(cfg.n_restarts, 1)):
            result = self._single_run(
                x=x, z=z, M=M, Mz=Mz,
                D_clean=D_clean, C_clean=C_clean,
                restart_idx=restart_idx,
            )
            # Pick the best start: fewest adversarial detections first,
            # then lowest final detection loss as tie-breaker.
            if best_result is None:
                best_result = result
            else:
                n_adv_best = len(self.det.detect_nms(
                    best_result.x_adv, conf_thr=cfg.conf_thr, iou_thr=cfg.iou_nms))
                n_adv_curr = len(self.det.detect_nms(
                    result.x_adv, conf_thr=cfg.conf_thr, iou_thr=cfg.iou_nms))
                last_loss_best = (best_result.history["L_det"][-1]
                                  if best_result.history["L_det"] else float("inf"))
                last_loss_curr = (result.history["L_det"][-1]
                                  if result.history["L_det"] else float("inf"))
                if (n_adv_curr < n_adv_best or
                        (n_adv_curr == n_adv_best and
                         last_loss_curr < last_loss_best)):
                    best_result = result

        return best_result  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # internal single-run (one restart)
    # ------------------------------------------------------------------

    def _single_run(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        M: torch.Tensor,
        Mz: torch.Tensor,
        D_clean: list,
        C_clean: list[int],
        restart_idx: int,
    ) -> AttackResult:
        """One optimization trajectory from a fixed or random delta initialisation."""
        cfg = self.cfg
        device = self.vae.device

        # ----- init perturbation (Phase 3C: random init for restarts > 0) -----
        if restart_idx == 0 or cfg.n_restarts <= 1:
            delta_data = torch.zeros_like(z)
        else:
            noise_scale = cfg.eps_z * cfg.restart_noise
            delta_data = (torch.rand_like(z) * 2 - 1) * noise_scale
            delta_data = delta_data * Mz  # keep masked region only

        delta = delta_data.requires_grad_(True)
        optim = torch.optim.Adam([delta], lr=cfg.lr)

        # ----- Phase 3B: MI-Adam momentum gradient accumulator -----
        grad_buf: torch.Tensor | None = None
        if cfg.use_momentum:
            grad_buf = torch.zeros_like(z)

        # ----- optimization loop -----
        history: dict[str, list[float]] = {
            "L": [], "L_det": [], "L_obj": [],
            "L_perc": [], "L_reg": [], "p_max": [],
        }
        steps_taken = 0

        for t in range(cfg.num_steps):
            z_adv = z + Mz * delta
            x_dec = self.vae.decode(z_adv)
            x_adv = M * x_dec + (1 - M) * x

            raw = self.det.forward_raw(x_adv)              # (1, A, 4 + nc)
            class_conf = self.det.class_confidence(raw)    # (1, A, nc)

            # ----- detection loss -----
            L_det = vanishing_loss(class_conf, C_clean, gamma=cfg.gamma)

            # ----- Phase 3A: objectness term -----
            L_obj = raw.new_zeros(())
            if cfg.use_objectness:
                L_obj = objectness_loss(raw, gamma=cfg.gamma)

            # ----- perceptual loss -----
            if self._lpips is not None:
                if cfg.use_ssim:
                    # Phase 3D: combined LPIPS + SSIM
                    L_perc = perceptual_combined(
                        x_adv, x, M,
                        lpips_fn=self._lpips,
                        ssim_weight=cfg.ssim_weight,
                    )
                else:
                    L_perc = self._lpips(x_adv, x, M)
            else:
                if cfg.use_ssim:
                    L_perc = ssim_loss(x_adv, x, M)
                else:
                    L_perc = masked_l2(x_adv, x, M)

            L_reg = latent_l2(delta)

            L = (L_det
                 + cfg.obj_weight * L_obj
                 + cfg.lambda_p * L_perc
                 + cfg.lambda_r * L_reg)

            optim.zero_grad(set_to_none=True)
            L.backward()

            # ----- Phase 3B: momentum gradient accumulation -----
            if cfg.use_momentum and grad_buf is not None and delta.grad is not None:
                with torch.no_grad():
                    g = delta.grad
                    # Normalize by L1 norm to make the scale of mu predictable
                    l1_norm = g.abs().mean().clamp(min=1e-8)
                    g_normalized = g / l1_norm
                    # Running average: g_buf <- mu * g_buf + g_normalized
                    grad_buf.mul_(cfg.momentum_decay).add_(g_normalized)
                    # Replace the raw gradient with the accumulated version
                    delta.grad.copy_(grad_buf)

            optim.step()

            with torch.no_grad():
                delta.data.clamp_(-cfg.eps_z, cfg.eps_z)
                delta.data.mul_(Mz)

                # logging
                p_per_class = class_conf[0, :, C_clean].amax(dim=0)  # (|C|,)
                p_max = float(p_per_class.max().item())
                history["L"].append(float(L.item()))
                history["L_det"].append(float(L_det.item()))
                history["L_obj"].append(float(L_obj.item()))
                history["L_perc"].append(float(L_perc.item()))
                history["L_reg"].append(float(L_reg.item()))
                history["p_max"].append(p_max)

            steps_taken = t + 1
            if cfg.early_stop and p_max < cfg.gamma - cfg.early_stop_margin:
                break

        # ----- final adversarial image -----
        with torch.no_grad():
            z_adv = z + Mz * delta
            x_adv_final = (M * self.vae.decode(z_adv) + (1 - M) * x).clamp(0, 1)

        return AttackResult(
            x_adv=x_adv_final.detach(),
            delta=delta.detach(),
            M=M.detach(),
            detections_clean=D_clean,
            classes_clean=C_clean,
            history=history,
            steps_taken=steps_taken,
        )
