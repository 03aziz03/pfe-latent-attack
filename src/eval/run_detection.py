"""Run YOLOv8 inference on an image directory and save FrameDetections to JSON."""
from __future__ import annotations

from pathlib import Path

import torch

from src.eval.io import save_detections
from src.eval.metrics import FrameDetections


def run_detection(
    image_dir: Path,
    model_path: Path,
    output_path: Path,
    stems: list[str] | None = None,
    conf_thr: float = 0.25,
    iou_nms: float = 0.45,
    img_size: int = 640,
    device: str = "cpu",
) -> dict[str, FrameDetections]:
    """Run YOLO inference on images in *image_dir* and persist detections.

    Args:
        image_dir:   Directory containing images (JPG / PNG / BMP).
        model_path:  Path to YOLOv8 weights (.pt file).
        output_path: Destination JSON file (schema defined in src.eval.io).
        stems:       If given, only process files whose stem is in this list.
                     Stems are matched case-sensitively; extension is ignored.
        conf_thr:    Confidence threshold; detections below are discarded.
        iou_nms:     IoU threshold used by YOLO's built-in NMS.
        img_size:    Inference input size (pixels, square).
        device:      Torch device string — "cpu" or "cuda".

    Returns:
        Dict mapping each processed stem → FrameDetections (post-NMS,
        filtered at conf_thr). The same data is written to *output_path*.
    """
    from ultralytics import YOLO  # noqa: PLC0415 — optional heavy import

    image_dir = Path(image_dir)
    model_path = Path(model_path)
    output_path = Path(output_path)

    model = YOLO(str(model_path))

    _EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    all_paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in _EXTS)

    if stems is not None:
        stems_set = set(stems)
        all_paths = [p for p in all_paths if p.stem in stems_set]

    if not all_paths:
        raise FileNotFoundError(
            f"No supported images found in {image_dir}"
            + (f" matching stems {stems}" if stems else "")
        )

    results: dict[str, FrameDetections] = {}
    n = len(all_paths)
    for idx, img_path in enumerate(all_paths, 1):
        preds = model.predict(
            source=str(img_path),
            conf=conf_thr,
            iou=iou_nms,
            imgsz=img_size,
            device=device,
            verbose=False,
        )
        r = preds[0]
        boxes: torch.Tensor = r.boxes.xyxy.cpu().float()
        scores: torch.Tensor = r.boxes.conf.cpu().float()
        classes: torch.Tensor = r.boxes.cls.cpu().long()
        results[img_path.stem] = FrameDetections(boxes=boxes, scores=scores, classes=classes)

        if idx % 10 == 0 or idx == n:
            print(f"  [{idx:3d}/{n}] {img_path.stem}  → {len(boxes)} dets")

    save_detections(results, output_path)
    print(f"  Saved {len(results)} frames → {output_path}")
    return results
