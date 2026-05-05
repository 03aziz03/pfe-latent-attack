# Colab Training — Debug Log & Status

**Project**: Final-year engineering internship (PFE) on adversarial attacks against YOLOv8.
**Goal of this file**: Self-contained log of all Colab issues encountered during YOLOv8 training on UA-DETRAC, with fixes applied and current status. Intended to be uploaded into a separate Cowork session for later analysis.

---

## 0. Context (one paragraph)

Training YOLOv8n on UA-DETRAC vehicle-detection dataset (4 classes: car, bus, van, others) on Google Colab Pro, with a custom Drive-sync callback that copies `last.pt` and `best.pt` to Google Drive after every epoch (so an interrupted run loses at most one epoch). Dataset was converted from DETRAC's per-sequence XML annotations to YOLO format using a custom converter script (`tools/detrac_to_yolo.py` in the project repo).

Trained weights `best.pt` will eventually replace the COCO-pretrained `yolov8n.pt` in `configs/default.yaml` for the downstream adversarial-attack pipeline.

---

## 1. Current Status

### What's Done

| Stage | Status | Notes |
|---|---|---|
| Project repo created | DONE | `D:\minimal_research\` with full attack pipeline, baselines, scripts |
| DETRAC raw uploaded to Drive | DONE | `MyDrive/PFE/DETRAC.zip` (Windows-zipped ~10 GB) |
| Drive mounted on Colab | DONE | `/content/drive/MyDrive/PFE/` accessible |
| Zip extracted to `/content/DETRAC/` | DONE | Contains `DETRAC-Images/` + `DETRAC-Train-Annotations-XML/` |
| Folder name fix (symlink) | DONE | `DETRAC-Images` -> `Insight-MVT_Annotation_Train` |
| DETRAC -> YOLO conversion | DONE | 14 170 train / 2 224 val frames; 51 train / 9 val sequences |
| `data.yaml` written | DONE | 4 classes correctly mapped |
| Model loads (yolov8n.pt) | DONE | Architecture initialized, transfer learning OK |

### What's Pending

| Stage | Blocker |
|---|---|
| **Training start** | Wandb integration error (Issue #5 below) — being fixed |
| Drive-sync callback verified | Will verify after first successful epoch |
| 60 epochs completed | Not started |
| Best.pt downloaded to local machine | Not started |
| Update `configs/default.yaml` with new weights | Not started |
| Run sanity check on attack with trained detector | Not started |

### Hardware

- Currently on **NVIDIA L4 (22.5 GB VRAM)** — Colab Pro
- Earlier on T4 (free tier) — switched to L4 for ~3x speedup
- Estimated training time: 2–3 hours for 60 epochs on L4

### Dataset Stats

- **Train**: 14 170 images / 51 sequences
- **Val**: 2 224 images / 9 sequences
- **Split ratio**: 86% / 14% (split BY sequence to avoid leakage)
- **Frame stride**: 5 (kept every 5th frame; original dataset has 25 fps with high redundancy)
- **Classes**: car (0), bus (1), van (2), others (3)
- **~37 sequences skipped** during conversion because no XML annotation (the test sequences mixed into `DETRAC-Images/` without their corresponding test XMLs — expected behavior)

---

## 2. Issues Encountered & Fixes (chronological)

### Issue #1 — Pip dependency-conflict warnings on first install

**Symptom (truncated)**:
```
ERROR: pip's dependency resolver does not currently take into account...
cupy-cuda12x 14.0.1 requires numpy<2.6,>=2.0, but you have numpy 1.26.4
rasterio 1.5.0 requires numpy>=2, but you have numpy 1.26.4
jaxlib 0.7.2 requires numpy>=2.0, but you have numpy 1.26.4
opencv-contrib-python 4.13.0.92 requires numpy>=2; ...
shap 0.51.0 requires numpy>=2, but you have numpy 1.26.4
```

**Cause**: Colab's preinstalled environment has migrated to numpy 2.x. Ultralytics' dependency chain (specifically OpenCV) wanted numpy 1.x in some sub-paths, so pip downgraded to 1.26.4. This created warnings about *other* Colab packages (rasterio, jax, cupy, shap, tifffile) that wanted numpy ≥ 2.

**Fix attempted (BAD ADVICE — caused Issue #4)**: Pinned `numpy<2` in install command.

**Lesson**: These warnings are **cosmetic**. Pip still installed everything correctly. The conflicting packages (rasterio, jax, etc.) aren't used by YOLOv8 training. **DO NOT pin numpy** on Colab — let it stay at 2.x.

**Status**: Resolved by reverting to no-numpy-pin (after Issue #4 forced a fix).

---

### Issue #2 — Bash unzip failed on Windows-created zip

**Symptom**:
```
warning: /content/drive/MyDrive/PFE/DETRAC.zip appears to use backslashes as path separators
CalledProcessError: Command '...' returned non-zero exit status 1.
```

**Cause**: PowerShell's `Compress-Archive` produces zips with `\` instead of `/` as path separators (non-standard but common on Windows). Linux `unzip` extracts correctly but emits a warning and exit code 1. The notebook used `%%bash -e` which treats any non-zero exit as fatal.

**Fix applied**: Replaced `%%bash` cell with Python `zipfile`:
```python
import zipfile, os
with zipfile.ZipFile('/content/drive/MyDrive/PFE/DETRAC.zip', 'r') as z:
    z.extractall('/content/DETRAC')
```

Python's `zipfile` handles backslashes correctly (converts to `/` during extraction).

**Alternative**: keep bash but use `unzip ... || true` to ignore exit code, and remove `-e` from `%%bash`.

**Status**: Resolved.

---

### Issue #3 — DETRAC folder name mismatch

**Symptom**: After unzip, `/content/DETRAC/` contained `DETRAC-Images/` and `DETRAC-Train-Annotations-XML/` instead of the expected `Insight-MVT_Annotation_Train/` and `DETRAC-Train-Annotations-XML/`.

**Cause**: DETRAC is distributed under different folder naming conventions depending on the source. The converter script (`tools/detrac_to_yolo.py`) hard-codes the older `Insight-MVT_Annotation_Train` name.

**Fix applied**: Created a symlink so the converter can find the data:
```python
os.symlink('/content/DETRAC/DETRAC-Images',
           '/content/DETRAC/Insight-MVT_Annotation_Train')
```

**Better long-term fix (TODO)**: Update `tools/detrac_to_yolo.py` to accept either folder name (auto-detect or via flag).

**Status**: Resolved by symlink. Repo-level fix pending.

---

### Issue #4 — Numpy ABI incompatibility blocking training

**Symptom**:
```
File ".../numpy/random/_pickle.py", line 1, in <module>
    from .mtrand import RandomState
ValueError: numpy.dtype size changed, may indicate binary incompatibility.
Expected 96 from C header, got 88 from PyObject
```

**Cause**: Colab's torch 2.10 + CUDA 12.8 + Python 3.12 stack has C extensions compiled against numpy 2.x (dtype struct size = 96 bytes). My earlier `numpy<2` pin downgraded numpy to 1.26.4 (dtype struct = 88 bytes). Mismatch -> ValueError when `numpy.random` initializes during `model.train()`.

**Fix applied**:
1. `Runtime -> Restart session` (mandatory — broken numpy is in memory)
2. Re-installed without numpy pin: `!pip install -q -U "numpy>=2" "ultralytics>=8.2,<9" pyyaml pillow tqdm`
3. Re-ran import test cell -> `numpy 2.x.x` confirmed
4. (`/content/dataset/` survived the restart; only Drive mount needed re-mounting)

**Lesson**: On Colab with recent torch builds, **do not constrain numpy to <2**. The "pip dependency conflict" warnings about numpy are cosmetic; the ABI break from forcing 1.x is real.

**Status**: Resolved.

---

### Issue #5 — Wandb integration error (CURRENT BLOCKER)

**Symptom**:
```
UsageError: Invalid project name '/content/runs/detrac':
cannot contain characters '/,\\,#,?,%,:', found '/'
```

**Cause**: Weights & Biases is preinstalled on Colab, and ultralytics auto-detects it and tries to create a wandb project named after the `project=` parameter passed to `model.train()`. We pass `project='/content/runs/detrac'` (a local filesystem path), which contains `/` — forbidden in wandb project names.

**Fix attempted #1**: `!pip uninstall -y wandb` + relaunch training.
- Failed because wandb was already loaded into `sys.modules` by ultralytics' Trainer. Uninstalling the package on disk doesn't remove imported modules from memory.

**Fix attempted #2 (current)**: Disable wandb via env vars + ultralytics SETTINGS + clean `sys.modules`:
```python
import os
os.environ['WANDB_DISABLED'] = 'true'
os.environ['WANDB_MODE'] = 'disabled'
from ultralytics.utils import SETTINGS
SETTINGS['wandb'] = False
import sys
for mod in list(sys.modules):
    if mod == 'wandb' or mod.startswith('wandb.'):
        del sys.modules[mod]
```

**Status**: Pending verification — last status: user encountered an `IndentationError` because of accidental leading whitespace when copying the fix cell. Once they paste cleanly, the training should start.

**Better long-term fix (TODO)**:
- Add wandb-disable code to the notebook's import-test cell (run automatically every fresh session).
- Alternative: change `project=` to a non-path-style name like `'detrac'` and let ultralytics decide where to save (default = `/content/runs/detect/`). But this changes our Drive-sync paths, so prefer the env-var approach.

---

### Issue #6 — IndentationError when pasting code

**Symptom**:
```
File "<ipython-input-...>", line 2
    os.environ['WANDB_DISABLED'] = 'true'
    ^
IndentationError: unexpected indent
```

**Cause**: When copying the fix code from the chat into a Colab cell, accidental leading spaces ended up before `import os` on line 1, making line 2 inconsistent.

**Fix**: Clear cell completely (`Ctrl+A`, `Delete`), re-paste with cursor at column 1.

**Lesson**: Markdown rendering sometimes adds spaces that aren't visible. Always start cells with code touching the left margin.

**Status**: User instructed; awaiting confirmation.

---

## 3. Cheatsheet — One-cell sequence to bring up training from scratch

If you ever need to restart from a fresh runtime, run these cells in order:

### Cell A — Install (no numpy pin!)
```python
%pip install -q "ultralytics>=8.2,<9" pyyaml pillow tqdm
```

### Cell B — Disable wandb
```python
import os
os.environ['WANDB_DISABLED'] = 'true'
os.environ['WANDB_MODE'] = 'disabled'
from ultralytics.utils import SETTINGS
SETTINGS['wandb'] = False
import sys
for mod in list(sys.modules):
    if mod == 'wandb' or mod.startswith('wandb.'):
        del sys.modules[mod]
print('wandb disabled OK')
```

### Cell C — Import sanity check
```python
import numpy as np, torch, ultralytics
print('numpy', np.__version__)
print('torch', torch.__version__, 'cuda:', torch.cuda.is_available())
print('ultralytics', ultralytics.__version__)
```
Expect: numpy 2.x, torch 2.10+, cuda True, ultralytics 8.2.x.

### Cell D — Drive mount
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Cell E — Paths
```python
import os
DRIVE_RUNS    = '/content/drive/MyDrive/PFE/runs'
RUN_NAME      = 'yolov8n_detrac'
LOCAL_PROJECT = '/content/runs/detrac'
DRIVE_RUN_DIR = os.path.join(DRIVE_RUNS, RUN_NAME)
os.makedirs(DRIVE_RUN_DIR, exist_ok=True)
```

### Cell F — Skip conversion if already done
```python
import os
print('train images:', len(os.listdir('/content/dataset/images/train')))
print('val images  :', len(os.listdir('/content/dataset/images/val')))
```
If non-zero, dataset is intact — skip the conversion cell.

### Cell G — Drive-sync callback
```python
import os, shutil, time

def make_drive_sync_callback(drive_run_dir):
    os.makedirs(os.path.join(drive_run_dir, 'weights'), exist_ok=True)
    def _copy_safe(src, dst):
        if os.path.exists(src):
            try: shutil.copy(src, dst)
            except Exception as e: print('[drive-sync] failed:', e)
    def on_fit_epoch_end(trainer):
        save_dir = str(trainer.save_dir)
        epoch = int(trainer.epoch) + 1
        t0 = time.time()
        for fn in ['last.pt', 'best.pt']:
            _copy_safe(os.path.join(save_dir, 'weights', fn),
                       os.path.join(drive_run_dir, 'weights', fn))
        _copy_safe(os.path.join(save_dir, 'results.csv'),
                   os.path.join(drive_run_dir, 'results.csv'))
        print(f'[drive-sync] epoch {epoch} -> {drive_run_dir} ({time.time()-t0:.1f}s)')
    return on_fit_epoch_end
```

### Cell H — Train
```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.add_callback('on_fit_epoch_end', make_drive_sync_callback(DRIVE_RUN_DIR))
results = model.train(
    data='/content/dataset/data.yaml',
    epochs=60, imgsz=640, batch=32,
    optimizer='SGD', lr0=0.01, cos_lr=True, patience=15,
    mosaic=1.0, close_mosaic=10,
    degrees=0.0, flipud=0.0, fliplr=0.5,
    project=LOCAL_PROJECT, name=RUN_NAME, exist_ok=True,
    device=0, plots=True, save=True, save_period=10, verbose=True,
)
```

### Cell I — Resume from Drive (only if interrupted)
```python
import os, shutil
local_run = os.path.join(LOCAL_PROJECT, RUN_NAME)
os.makedirs(os.path.join(local_run, 'weights'), exist_ok=True)
for fn in ['last.pt', 'best.pt']:
    src = os.path.join(DRIVE_RUN_DIR, 'weights', fn)
    if os.path.exists(src):
        shutil.copy(src, os.path.join(local_run, 'weights', fn))
        print('Restored', fn)

from ultralytics import YOLO
model = YOLO(os.path.join(local_run, 'weights', 'last.pt'))
model.add_callback('on_fit_epoch_end', make_drive_sync_callback(DRIVE_RUN_DIR))
results = model.train(resume=True)
```

---

## 4. Open Questions / TODO for the next session

1. **Did the wandb fix work?** Need confirmation that epoch 1 actually started. Check for the message `Epoch GPU_mem box_loss cls_loss dfl_loss ...` in training output.
2. **Did the Drive-sync callback fire correctly?** After epoch 1, look in `MyDrive/PFE/runs/yolov8n_detrac/weights/` — should see `last.pt`. If yes, the safety net works.
3. **What mAP did training reach?** Target for DETRAC + yolov8n: mAP@50 ≥ 0.75 overall. If significantly lower, consider yolov8s.pt as starting weights.
4. **Patch the converter** (`tools/detrac_to_yolo.py`) to auto-detect both folder names (`Insight-MVT_Annotation_Train` AND `DETRAC-Images`).
5. **Patch the notebook** to:
   - Embed the wandb-disable cell as the official cell 3 (right after install).
   - Replace the `%%bash -e` unzip with the Python `zipfile` version.
   - Add a numpy-version sanity check.

---

## 5. Lessons Learned (one-liners)

1. **Don't pin numpy on Colab.** Recent torch needs numpy 2.x; pinning <2 breaks ABI.
2. **Disable wandb explicitly** when using path-style `project=` paths — env vars + ultralytics SETTINGS + sys.modules cleanup.
3. **Windows-zipped DETRAC** breaks `unzip -e`; use Python `zipfile` instead.
4. **DETRAC folder names vary** between distributions; symlink as a quick fix, or generalize the converter.
5. **Pip dependency-conflict warnings ≠ errors.** Read the actual import test output before "fixing" things.
6. **Always restart runtime** after a numpy or core-package reinstall — modules cached in `sys.modules` won't reload.
7. **Drive-sync after every epoch** is non-negotiable on Colab — sessions die unpredictably.
8. **`/content/`** survives `Restart session` but not `Disconnect and delete runtime`. Drive mount is lost on either.

---

*Generated during a Cowork session on 2026-05-04. Upload this file to a fresh Cowork session to resume analysis.*
