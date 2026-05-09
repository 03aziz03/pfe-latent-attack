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
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch

from .detector import YOLOv8Wrapper
from .vae import SDVAE
from .masks import boxes_to_pixel_mask, pixel_mask_to_latent_mask
from .losses import vanishing_loss, masked_l2, latent_l2, MaskedLPIPS
from .utils import Detection


@dataclass
class AttackConfig:
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

        Args:
            x: (1, 3, H, W) tensor in [0, 1]; H, W must be multiples of 8.

        Returns:
            AttackResult.
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
            # nothing to attack
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

        # ----- 4. init perturbation -----
        delta = torch.zeros_like(z, requires_grad=True)
        optim = torch.optim.Adam([delta], lr=cfg.lr)

        # ----- 5. optimization loop -----
        history = {"L": [], "L_det": [], "L_perc": [], "L_reg": [], "p_max": []}
        steps_taken = 0
        for t in range(cfg.num_steps):
            z_adv = z + Mz * delta
            x_dec = self.vae.decode(z_adv)
            x_adv = M * x_dec + (1 - M) * x

            raw = self.det.forward_raw(x_adv)             # (1, A, 4 + nc)
            class_conf = self.det.class_confidence(raw)   # (1, A, nc)

            L_det = vanishing_loss(class_conf, C_clean, gamma=cfg.gamma)
            if self._lpips is not None:
                L_perc = self._lpips(x_adv, x, M)
            else:
                L_perc = masked_l2(x_adv, x, M)
            L_reg = latent_l2(delta)
            L = L_det + cfg.lambda_p * L_perc + cfg.lambda_r * L_reg

            optim.zero_grad(set_to_none=True)
            L.backward()
            optim.step()

            with torch.no_grad():
                delta.data.clamp_(-cfg.eps_z, cfg.eps_z)
                delta.data.mul_(Mz)

                # logging
                p_per_class = class_conf[0, :, C_clean].amax(dim=0)  # (|C|,)
                p_max = float(p_per_class.max().item())
                history["L"].append(float(L.item()))
                history["L_det"].append(float(L_det.item()))
                history["L_perc"].append(float(L_perc.item()))
                history["L_reg"].append(float(L_reg.item()))
                history["p_max"].append(p_max)

            steps_taken = t + 1
            if cfg.early_stop and p_max < cfg.gamma - cfg.early_stop_margin:
                break

        # ----- 6. final adversarial image -----
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
