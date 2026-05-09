"""Load and save per-frame detection results in a standard JSON schema.

Schema (both load and save)::

    {
        "<frame_id>": {
            "boxes":   [[x1, y1, x2, y2], ...],   // xyxy pixel coords
            "scores":  [0.9, 0.85, ...],
            "classes": [0, 1, ...]
        },
        ...
    }
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from src.eval.metrics import FrameDetections


def load_detections(path: Path) -> dict[str, FrameDetections]:
    """Load per-frame detections from a JSON file.

    Args:
        path: Path to a JSON file matching the schema above.

    Returns:
        Dict mapping frame_id → FrameDetections. Empty detections are
        represented by tensors with shape (0, 4), (0,), (0,).

    Raises:
        FileNotFoundError: if path does not exist.
        KeyError: if a frame entry is missing 'boxes', 'scores', or 'classes'.
    """
    with open(path, "r", encoding="utf-8") as fh:
        raw: dict = json.load(fh)

    result: dict[str, FrameDetections] = {}
    for frame_id, data in raw.items():
        boxes_list: list = data["boxes"]
        scores_list: list = data["scores"]
        classes_list: list = data["classes"]

        boxes = (
            torch.tensor(boxes_list, dtype=torch.float32)
            if boxes_list
            else torch.zeros((0, 4), dtype=torch.float32)
        )
        scores = torch.tensor(scores_list, dtype=torch.float32)
        classes = torch.tensor(classes_list, dtype=torch.long)

        result[frame_id] = FrameDetections(boxes=boxes, scores=scores, classes=classes)

    return result


def save_detections(dets: dict[str, FrameDetections], path: Path) -> None:
    """Save per-frame detections to a JSON file.

    Args:
        dets: Dict mapping frame_id → FrameDetections.
        path: Output path. Parent directories are created automatically.
    """
    raw: dict[str, dict] = {}
    for frame_id, fd in dets.items():
        raw[frame_id] = {
            "boxes": fd.boxes.tolist(),
            "scores": fd.scores.tolist(),
            "classes": fd.classes.tolist(),
        }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(raw, fh, indent=2)
