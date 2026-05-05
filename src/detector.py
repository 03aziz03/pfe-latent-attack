"""YOLOv8 wrapper: frozen, with a differentiable pre-NMS forward.

The wrapper exposes two paths:

* ``forward_raw(x)``  -> raw pre-NMS predictions (gradient-friendly).
* ``detect_nms(x)``   -> post-NMS list of ``Detection`` (for D_clean / eval).

We deliberately avoid Ultralytics' high-level ``model.predict`` API during the
attack because it runs NMS, which is not differentiable. Instead we call the
underlying ``nn.Module`` directly.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torchvision.ops as tv_ops
from ultralytics import YOLO

from .utils import Detection


class YOLOv8Wrapper(nn.Module):
    """Frozen YOLOv8 detector with differentiable pre-NMS forward.

    Args:
        weights: path or alias for a YOLOv8 checkpoint (e.g. "yolov8n.pt").
        device:  "cuda" or "cpu".
    """

    def __init__(self, weights: str = "yolov8n.pt", device: str = "cuda"):
        super().__init__()
        self.device = torch.device(device)

        wrapper = YOLO(weights)              # ultralytics high-level wrapper
        self.model = wrapper.model.to(self.device).eval()
        self.names: dict[int, str] = wrapper.names

        # Freeze all parameters; we still need autograd through activations
        # so we do NOT wrap forward in torch.no_grad().
        for p in self.model.parameters():
            p.requires_grad_(False)

        # Number of classes (useful when slicing pre-NMS outputs).
        self.num_classes = len(self.names)

    # ------------------------------------------------------------------
    # forward paths
    # ------------------------------------------------------------------

    def forward_raw(self, x: torch.Tensor) -> torch.Tensor:
        """Pre-NMS forward producing predictions of shape ``(B, A, 4 + nc)``.

        For YOLOv8 the output layout per anchor is::

            [cx, cy, w, h, cls_logit_0, ..., cls_logit_{nc-1}]

        where the cls logits are already passed through sigmoid by the
        model's ``Detect`` head. We expose the tensor untouched so that
        ``losses.py`` can grab the class-confidence slice directly.

        Note: YOLOv8 (unlike v5) does NOT have a separate objectness term.
        Class confidences play the role of obj * cls in our loss.
        """
        x = x.to(self.device)
        out = self.model(x)
        # During eval, ultralytics returns (preds, _) or just preds.
        if isinstance(out, (list, tuple)):
            out = out[0]
        # Expected shape from the Detect head: (B, 4 + nc, A). Transpose to
        # (B, A, 4 + nc) for convenience.
        if out.dim() == 3 and out.shape[1] == 4 + self.num_classes:
            out = out.transpose(1, 2).contiguous()
        return out

    @torch.no_grad()
    def detect_nms(self,
                   x: torch.Tensor,
                   conf_thr: float = 0.25,
                   iou_thr: float = 0.45) -> list[Detection]:
        """Run YOLOv8 with NMS and return a list of ``Detection`` (B=1 only)."""
        if x.shape[0] != 1:
            raise ValueError("detect_nms expects a single-image batch.")
        raw = self.forward_raw(x)                      # (1, A, 4 + nc)
        anchors = raw[0]                               # (A, 4 + nc)

        scores, cls_ids = anchors[:, 4:].max(dim=1)   # (A,), (A,)
        keep = scores > conf_thr
        if not keep.any():
            return []

        anchors, scores, cls_ids = anchors[keep], scores[keep], cls_ids[keep]

        # cx,cy,w,h -> x1,y1,x2,y2
        cx, cy, w, h = anchors[:, 0], anchors[:, 1], anchors[:, 2], anchors[:, 3]
        boxes = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=1)

        nms_idx = tv_ops.batched_nms(boxes, scores, cls_ids, iou_threshold=iou_thr)
        nms_idx = nms_idx[:300]

        out: list[Detection] = []
        for i in nms_idx.cpu().tolist():
            x1, y1, x2, y2 = boxes[i].cpu().tolist()
            out.append(Detection(box=(x1, y1, x2, y2),
                                 cls=int(cls_ids[i].item()),
                                 score=float(scores[i].item())))
        return out

    # ------------------------------------------------------------------
    # convenience
    # ------------------------------------------------------------------

    def class_confidence(self, raw: torch.Tensor) -> torch.Tensor:
        """Slice class confidences from a raw prediction tensor.

        raw: (B, A, 4 + nc) -> returns (B, A, nc).
        """
        return raw[..., 4:4 + self.num_classes]
