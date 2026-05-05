"""Utility helpers: config loading, IO, visualization."""
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import yaml
from PIL import Image


# ----------------------------- config -----------------------------


def load_config(path: str | Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------- IO ---------------------------------


def load_image(path: str | Path, imgsz: int = 640) -> torch.Tensor:
    """Load an image as a torch tensor in [0, 1], shape (1, 3, H, W).

    The image is resized so that the longest side equals ``imgsz`` and then
    letterboxed to a square. Returns a float32 tensor on CPU.
    """
    img = Image.open(path).convert("RGB")
    img = letterbox_pil(img, imgsz)
    arr = np.asarray(img).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    return t


def save_image(t: torch.Tensor, path: str | Path) -> None:
    """Save a (1, 3, H, W) or (3, H, W) tensor with values in [0, 1]."""
    if t.dim() == 4:
        t = t[0]
    t = t.detach().clamp(0, 1).cpu().permute(1, 2, 0).numpy()
    arr = (t * 255.0).round().astype(np.uint8)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def letterbox_pil(img: Image.Image, size: int) -> Image.Image:
    """Resize keeping aspect ratio, then pad to a square of side ``size``."""
    w, h = img.size
    scale = size / max(w, h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    img = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    canvas.paste(img, ((size - nw) // 2, (size - nh) // 2))
    return canvas


# ----------------------------- detection helpers ------------------


@dataclass
class Detection:
    box: tuple[float, float, float, float]   # xyxy in pixels
    cls: int
    score: float


def detections_to_classes(dets: Sequence[Detection]) -> list[int]:
    return sorted({d.cls for d in dets})


def iou_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU between two sets of boxes in xyxy format.

    a: (N, 4)   b: (M, 4)   -> (N, M)
    """
    if a.numel() == 0 or b.numel() == 0:
        return torch.zeros((a.shape[0], b.shape[0]), device=a.device)
    a_x1, a_y1, a_x2, a_y2 = a.unbind(-1)
    b_x1, b_y1, b_x2, b_y2 = b.unbind(-1)
    inter_x1 = torch.maximum(a_x1[:, None], b_x1[None, :])
    inter_y1 = torch.maximum(a_y1[:, None], b_y1[None, :])
    inter_x2 = torch.minimum(a_x2[:, None], b_x2[None, :])
    inter_y2 = torch.minimum(a_y2[:, None], b_y2[None, :])
    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    area_a = ((a_x2 - a_x1) * (a_y2 - a_y1))[:, None]
    area_b = ((b_x2 - b_x1) * (b_y2 - b_y1))[None, :]
    union = area_a + area_b - inter
    return inter / union.clamp(min=1e-8)


# ----------------------------- visualization ----------------------


def overlay_detections(img: torch.Tensor,
                        dets: Sequence[Detection],
                        names: dict[int, str] | None = None) -> Image.Image:
    """Return a PIL image with detection boxes overlaid (no matplotlib needed)."""
    from PIL import ImageDraw, ImageFont

    if img.dim() == 4:
        img = img[0]
    arr = (img.detach().clamp(0, 1).cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    pil = Image.fromarray(arr)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for d in dets:
        x1, y1, x2, y2 = d.box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        label = f"{names.get(d.cls, d.cls) if names else d.cls}:{d.score:.2f}"
        draw.text((x1 + 2, y1 + 2), label, fill="red", font=font)
    return pil
