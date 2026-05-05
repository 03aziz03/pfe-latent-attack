# Dataset folder

This folder will contain your YOLO-format dataset after running the
DETRAC -> YOLO converter.

## Step 1: keep the raw DETRAC dataset OUTSIDE this repo

DETRAC is large (~10 GB). Keep the raw dataset somewhere else, e.g.:

```
D:\datasets\DETRAC\
    Insight-MVT_Annotation_Train\
        MVI_20011\
            img00001.jpg
            ...
        MVI_20012\
            ...
    Insight-MVT_Annotation_Test\
        ...
    DETRAC-Train-Annotations-XML\
        MVI_20011.xml
        ...
    DETRAC-Test-Annotations-XML\
        ...
```

Do NOT copy this whole tree into the repo; only the YOLO-format conversion lands here.

## Step 2: run the converter

From the repository root:

```bash
python tools/detrac_to_yolo.py \
    --detrac-root D:\datasets\DETRAC \
    --out         dataset \
    --split       train \
    --val-frac    0.1 \
    --frame-stride 5
```

`--frame-stride 5` keeps every 5th frame so you train on ~28k frames instead
of 140k (DETRAC has 25 fps; consecutive frames are highly redundant).

After conversion, this folder will look like:

```
dataset/
├── data.yaml
├── images/
│   ├── train/MVI_20011_img00001.jpg ...
│   └── val/MVI_40701_img00001.jpg ...
└── labels/
    ├── train/MVI_20011_img00001.txt ...
    └── val/MVI_40701_img00001.txt ...
```

Each label file has one line per bounding box:

```
<class_id> <cx> <cy> <w> <h>          (all coordinates normalized to [0, 1])
```

Class IDs:

| ID | Name |
|---|---|
| 0 | car |
| 1 | bus |
| 2 | van |
| 3 | others |

## Step 3: train

See `notebooks/train_yolov8_detrac.ipynb` (Colab Pro recommended) or
`tools/train_yolov8.py` (local GPU).

The trained weights end up at `runs/detect/train/weights/best.pt`. You will
later load these weights in `configs/default.yaml` (set
`detector.weights: path/to/best.pt`) so the adversarial attack runs against
your fine-tuned detector.
