"""Convert UA-DETRAC annotations to YOLO format.

UA-DETRAC layout (input):
    detrac_root/
        Insight-MVT_Annotation_Train/   # or _Test
            MVI_20011/                  # sequence folder, contains frames
                img00001.jpg
                img00002.jpg
                ...
        DETRAC-Train-Annotations-XML/   # or DETRAC-Test-Annotations-XML
            MVI_20011.xml
            MVI_20012.xml
            ...

DETRAC XML format (one file per sequence):
    <sequence name="MVI_20011">
        <ignored_region> ... </ignored_region>
        <frame density="..." num="1">
            <target_list>
                <target id="1">
                    <box left="592.75" top="378.86" width="160.07" height="162.49"/>
                    <attribute orientation="..." speed="..." trajectory_length="..."
                               truncation_ratio="..." vehicle_type="car"/>
                </target>
                ...
            </target_list>
        </frame>
        ...
    </sequence>

YOLO format (output):
    yolo_root/
        images/
            train/MVI_20011_img00001.jpg
            val/MVI_40701_img00001.jpg
        labels/
            train/MVI_20011_img00001.txt    (one line per box: cls cx cy w h, all in [0,1])
            val/MVI_40701_img00001.txt
        data.yaml

Class mapping (DETRAC -> YOLO id):
    car        -> 0
    bus        -> 1
    van        -> 2
    others     -> 3

Frames inside <ignored_region> are NOT skipped at frame level (only boxes are
defined per target), but boxes can be filtered by truncation_ratio threshold.

Usage (typical):
    python tools/detrac_to_yolo.py \
        --detrac-root /path/to/DETRAC \
        --out         dataset \
        --split       train \
        --val-frac    0.1 \
        --frame-stride 5
"""
from __future__ import annotations

import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

CLASS_MAP = {
    "car": 0,
    "bus": 1,
    "van": 2,
    "others": 3,
    "other": 3,    # tolerate spelling variants
}
CLASS_NAMES = ["car", "bus", "van", "others"]


def parse_sequence(xml_path: Path,
                    frames_dir: Path,
                    truncation_max: float = 0.5) -> dict[str, list[tuple[int, float, float, float, float]]]:
    """Parse a DETRAC sequence XML.

    Returns a dict { frame_id (zero-padded 5-digit string) -> list of (cls, cx, cy, w, h) }
    where cx, cy, w, h are normalized to [0, 1] using the actual frame image size.
    """
    tree = ET.parse(xml_path)
    seq = tree.getroot()

    # determine image size from the first frame
    first_frame = next(frames_dir.glob("img*.jpg"), None)
    if first_frame is None:
        raise FileNotFoundError(f"No frames found under {frames_dir}")
    from PIL import Image
    with Image.open(first_frame) as im:
        W, H = im.size

    boxes_per_frame: dict[str, list[tuple[int, float, float, float, float]]] = {}
    for frame in seq.findall("frame"):
        num = int(frame.get("num"))
        frame_id = f"{num:05d}"
        boxes: list[tuple[int, float, float, float, float]] = []
        for target in frame.findall("target_list/target"):
            box = target.find("box")
            attr = target.find("attribute")
            if box is None or attr is None:
                continue
            try:
                trunc = float(attr.get("truncation_ratio", "0"))
            except ValueError:
                trunc = 0.0
            if trunc > truncation_max:
                continue
            vt = (attr.get("vehicle_type") or "").strip().lower()
            if vt not in CLASS_MAP:
                continue
            cls = CLASS_MAP[vt]
            try:
                left = float(box.get("left"))
                top = float(box.get("top"))
                w = float(box.get("width"))
                h = float(box.get("height"))
            except (TypeError, ValueError):
                continue
            cx = (left + w / 2.0) / W
            cy = (top + h / 2.0) / H
            nw = w / W
            nh = h / H
            # clamp
            cx = min(max(cx, 0.0), 1.0)
            cy = min(max(cy, 0.0), 1.0)
            nw = min(max(nw, 0.0), 1.0)
            nh = min(max(nh, 0.0), 1.0)
            if nw <= 0 or nh <= 0:
                continue
            boxes.append((cls, cx, cy, nw, nh))
        if boxes:
            boxes_per_frame[frame_id] = boxes
    return boxes_per_frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detrac-root", required=True, type=Path,
                    help="root containing Insight-MVT_Annotation_* and DETRAC-*-Annotations-XML")
    ap.add_argument("--out", required=True, type=Path,
                    help="output dataset root in YOLO format")
    ap.add_argument("--split", choices=["train", "test"], default="train",
                    help="which DETRAC split to convert (Train or Test)")
    ap.add_argument("--val-frac", type=float, default=0.1,
                    help="fraction of sequences held out for validation when --split=train")
    ap.add_argument("--frame-stride", type=int, default=5,
                    help="keep every Nth frame to reduce dataset size (DETRAC has ~140k frames)")
    ap.add_argument("--truncation-max", type=float, default=0.5,
                    help="drop boxes with truncation_ratio above this")
    ap.add_argument("--copy-images", action="store_true",
                    help="copy images instead of symlinking (use on Windows / Colab)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)

    if args.split == "train":
        frames_root = args.detrac_root / "Insight-MVT_Annotation_Train"
        xml_root = args.detrac_root / "DETRAC-Train-Annotations-XML"
    else:
        frames_root = args.detrac_root / "Insight-MVT_Annotation_Test"
        xml_root = args.detrac_root / "DETRAC-Test-Annotations-XML"

    if not frames_root.exists():
        raise FileNotFoundError(f"Frames root not found: {frames_root}")
    if not xml_root.exists():
        raise FileNotFoundError(f"XML root not found: {xml_root}")

    sequences = sorted(p.name for p in frames_root.iterdir() if p.is_dir() and p.name.startswith("MVI_"))
    if not sequences:
        raise FileNotFoundError(f"No MVI_* sequences found under {frames_root}")
    print(f"Found {len(sequences)} sequences in {frames_root}")

    # train/val split by sequence (avoids data leakage between splits)
    if args.split == "train":
        random.shuffle(sequences)
        n_val = max(1, int(round(len(sequences) * args.val_frac)))
        val_seqs = set(sequences[:n_val])
        train_seqs = set(sequences[n_val:])
    else:
        train_seqs = set()
        val_seqs = set(sequences)   # if converting test split, all goes to val

    out = args.out
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        (out / sub).mkdir(parents=True, exist_ok=True)

    n_kept = 0
    n_seqs = 0
    for seq_name in sequences:
        n_seqs += 1
        xml_path = xml_root / f"{seq_name}.xml"
        frames_dir = frames_root / seq_name
        if not xml_path.exists():
            print(f"  SKIP {seq_name}: no XML")
            continue
        try:
            boxes_per_frame = parse_sequence(xml_path, frames_dir,
                                              truncation_max=args.truncation_max)
        except Exception as e:
            print(f"  SKIP {seq_name}: {e}")
            continue

        target_split = "val" if seq_name in val_seqs else "train"

        for i, (frame_id, boxes) in enumerate(sorted(boxes_per_frame.items())):
            if i % args.frame_stride != 0:
                continue
            src_img = frames_dir / f"img{frame_id}.jpg"
            if not src_img.exists():
                continue
            dst_img = out / "images" / target_split / f"{seq_name}_img{frame_id}.jpg"
            dst_lbl = out / "labels" / target_split / f"{seq_name}_img{frame_id}.txt"
            if args.copy_images:
                shutil.copy2(src_img, dst_img)
            else:
                # symlink works on linux/colab; falls back to copy on windows
                try:
                    if dst_img.exists():
                        dst_img.unlink()
                    dst_img.symlink_to(src_img.resolve())
                except (OSError, NotImplementedError):
                    shutil.copy2(src_img, dst_img)
            with open(dst_lbl, "w") as f:
                for cls, cx, cy, w, h in boxes:
                    f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
            n_kept += 1
        if n_seqs % 10 == 0:
            print(f"  processed {n_seqs}/{len(sequences)} sequences ({n_kept} frames so far)")

    # data.yaml
    yaml_text = (
        f"# generated by tools/detrac_to_yolo.py\n"
        f"path: {out.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"\n"
        f"names:\n"
    )
    for i, name in enumerate(CLASS_NAMES):
        yaml_text += f"  {i}: {name}\n"
    (out / "data.yaml").write_text(yaml_text)

    print(f"\nDone. {n_kept} frames written under {out}/")
    print(f"  train sequences: {len(train_seqs)}   val sequences: {len(val_seqs)}")
    print(f"  data.yaml: {out / 'data.yaml'}")


if __name__ == "__main__":
    main()
