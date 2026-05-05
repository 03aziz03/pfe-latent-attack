"""Folder-based image loader.

Walks an input directory for ``.jpg``, ``.jpeg`` and ``.png`` files (case
insensitive) and yields ``(stem, tensor)`` pairs in lexicographic order.

Designed to be tiny and dependency-free; swap in a COCO loader later by
implementing the same interface.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import torch

from .utils import load_image

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class ImageFolder:
    def __init__(self, root: str | Path, imgsz: int = 640):
        self.root = Path(root)
        self.imgsz = imgsz
        if not self.root.exists():
            raise FileNotFoundError(f"Image folder not found: {self.root}")
        self.paths = sorted(
            p for p in self.root.iterdir()
            if p.is_file() and p.suffix.lower() in VALID_EXTS
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __iter__(self) -> Iterator[tuple[str, torch.Tensor]]:
        for p in self.paths:
            yield p.stem, load_image(p, self.imgsz)
