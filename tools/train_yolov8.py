"""Train YOLOv8 on a YOLO-format dataset (e.g. DETRAC after conversion).

Local-GPU version of the Colab notebook. Use this if you have a workstation
with a CUDA GPU; otherwise prefer notebooks/train_yolov8_detrac.ipynb.

Usage:
    python tools/train_yolov8.py \
        --data dataset/data.yaml \
        --weights yolov8n.pt \
        --epochs 80 \
        --imgsz 640 \
        --batch 16 \
        --project runs/detrac \
        --name yolov8n_detrac
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to data.yaml")
    ap.add_argument("--weights", default="yolov8n.pt",
                    help="starting weights (yolov8n.pt for transfer learning)")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--project", default="runs/detrac")
    ap.add_argument("--name", default="yolov8n_detrac")
    ap.add_argument("--device", default="0", help="GPU id or 'cpu'")
    ap.add_argument("--patience", type=int, default=15,
                    help="early-stopping patience on val mAP")
    ap.add_argument("--lr0", type=float, default=0.01)
    ap.add_argument("--optimizer", default="SGD", choices=["SGD", "AdamW"])
    args = ap.parse_args()

    print(f"Starting from weights: {args.weights}")
    model = YOLO(args.weights)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        device=args.device,
        patience=args.patience,
        lr0=args.lr0,
        optimizer=args.optimizer,
        # standard YOLOv8 augmentations -- good defaults for surveillance video
        mosaic=1.0,
        close_mosaic=10,    # disable mosaic for the last 10 epochs
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=0.0,        # surveillance cameras don't rotate
        translate=0.1,
        scale=0.5,
        flipud=0.0,
        fliplr=0.5,
        # logging
        plots=True,
        save=True,
        save_period=10,
        verbose=True,
    )

    # final eval on the val split
    print("\nFinal validation:")
    metrics = model.val(data=args.data, imgsz=args.imgsz, batch=args.batch)
    print(metrics)

    best = Path(args.project) / args.name / "weights" / "best.pt"
    print(f"\nBest weights: {best.resolve()}")


if __name__ == "__main__":
    main()
