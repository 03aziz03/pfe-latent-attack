"""Generate all publication figures (f01–f13) as PNG + PDF.

v2: 13 figures (10 reworked + 3 new), output to results/figures/png/ and
results/figures/pdf/, with a single HTML index at results/figures/index.html.

Usage
-----
    python scripts/generate_figures.py [--out-dir results/figures]

Output
------
    results/figures/
        png/  f01_dfr_distribution.png ... f13_iou_distribution.png
        pdf/  f01_dfr_distribution.pdf ... f13_iou_distribution.pdf
        index.html
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.viz.style import ATTACK_ORDER, setup_publication_style  # noqa: E402
from src.viz import (  # noqa: E402
    metrics_plots,
    detection_overlay,
    perturbation,
    grids,
    analysis_plots,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

RESULTS_DIR = ROOT / "results"
CLEAN_DIR   = ROOT / "data" / "images"
ADV_DIRS: dict[str, Path] = {
    "latent": RESULTS_DIR / "adv_latent",
    "pgd":    RESULTS_DIR / "adv_pgd",
    "fgsm":   RESULTS_DIR / "adv_fgsm",
}
METRICS_FILES: dict[str, Path] = {
    "latent": RESULTS_DIR / "metrics_latent.json",
    "pgd":    RESULTS_DIR / "metrics_pgd.json",
    "fgsm":   RESULTS_DIR / "metrics_fgsm.json",
}
METRICS_FULL_PATH = RESULTS_DIR / "metrics_full.json"
DETS_FILES: dict[str, Path] = {
    "clean":  RESULTS_DIR / "dets_clean.json",
    "latent": RESULTS_DIR / "dets_latent.json",
    "pgd":    RESULTS_DIR / "dets_pgd.json",
    "fgsm":   RESULTS_DIR / "dets_fgsm.json",
}

DETRAC_CLASSES = {0: "car", 1: "bus", 2: "van", 3: "others"}

# Frames for visual figures
HEATMAP_STEM  = "img00001"
OVERLAY_STEM  = "img00001"
GRID_STEMS    = ["img00001", "img00020", "img00037", "img00048"]

# Binary-DFR success frames for timeseries highlights
BINARY_STEMS  = ["img00001", "img00002", "img00003", "img00004", "img00008"]


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def load_per_image_data() -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for atk, path in METRICS_FILES.items():
        if not path.exists():
            print(f"[WARN] Missing {path.name}")
            continue
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
        out[atk] = raw.get("per_image", [])
    return out


def load_summary_data() -> dict[str, dict]:
    summary: dict[str, dict] = {}
    for atk, path in METRICS_FILES.items():
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
        s = dict(raw.get("summary", {}))
        s.setdefault("DFR_loose", s.get("DFR"))
        s.setdefault("ASR_loose", s.get("ASR"))
        s.setdefault("PSNR_mask", s.get("mean_PSNR_mask_dB"))
        per_image = raw.get("per_image", [])
        valid = [f for f in per_image if f.get("n_clean", 0) > 0]
        if valid:
            dfr_props = [1.0 - f["n_adv"] / max(f["n_clean"], 1) for f in valid]
            s["dfr_prop"] = float(sum(dfr_props) / len(dfr_props))
            s["dfr_bin"]  = float(sum(1.0 for f in valid if f["n_adv"] == 0) / len(valid))
        summary[atk] = s

    if METRICS_FULL_PATH.exists():
        with open(METRICS_FULL_PATH, encoding="utf-8") as fh:
            full = json.load(fh)
        for atk in ATTACK_ORDER:
            if atk in full and atk in summary:
                summary[atk].update(full[atk].get("summary", {}))

    return summary


def load_dets() -> dict[str, dict]:
    from src.eval.io import load_detections  # noqa: PLC0415
    out: dict[str, dict] = {}
    for key, path in DETS_FILES.items():
        if path.exists():
            try:
                out[key] = load_detections(path)
            except Exception as exc:
                print(f"[WARN] Could not load {path.name}: {exc}")
    return out


def load_per_frame_metrics_full() -> dict[str, dict[str, dict]]:
    """Load per_frame list from metrics_full.json keyed as {attack: {stem: row}}."""
    if not METRICS_FULL_PATH.exists():
        return {}
    with open(METRICS_FULL_PATH, encoding="utf-8") as fh:
        full = json.load(fh)
    result: dict[str, dict[str, dict]] = {}
    for atk in ATTACK_ORDER:
        if atk not in full:
            continue
        result[atk] = {f["stem"]: f for f in full[atk].get("per_frame", [])}
    return result


def _find_img(directory: Path, stem: str) -> Path | None:
    for ext in (".jpg", ".jpeg", ".png", ".bmp"):
        p = directory / f"{stem}{ext}"
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------


def _save(fig: plt.Figure, name: str, png_dir: Path, pdf_dir: Path, dpi: int = 300) -> None:
    png_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_dir / f"{name}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure makers — each returns a Figure or raises
# ---------------------------------------------------------------------------


def make_f01(per_image, **_):
    return metrics_plots.per_frame_dfr_distribution(per_image)


def make_f02(per_image, **_):
    return metrics_plots.n_clean_vs_n_adv_scatter(per_image)


def make_f03(summary, **_):
    return metrics_plots.metric_bar_chart(summary)


def make_f04(summary, **_):
    return metrics_plots.stealth_vs_effectiveness_preview(summary)


def make_f05(**_):
    clean_path = _find_img(CLEAN_DIR, HEATMAP_STEM)
    if clean_path is None:
        raise FileNotFoundError(f"Clean image {HEATMAP_STEM} not found")
    adv_paths = {a: p for a in ATTACK_ORDER
                 if (p := _find_img(ADV_DIRS[a], HEATMAP_STEM)) is not None}
    return perturbation.perturbation_heatmap(clean_path, adv_paths)


def make_f06(**_):
    clean_path = _find_img(CLEAN_DIR, HEATMAP_STEM)
    if clean_path is None:
        raise FileNotFoundError(f"Clean image {HEATMAP_STEM} not found")
    adv_paths = {a: p for a in ATTACK_ORDER
                 if (p := _find_img(ADV_DIRS[a], HEATMAP_STEM)) is not None}
    return perturbation.difference_grid(clean_path, adv_paths)


def _make_overlay(attack: str, dets_all: dict, **_) -> plt.Figure:
    clean_path = _find_img(CLEAN_DIR, OVERLAY_STEM)
    adv_path   = _find_img(ADV_DIRS[attack], OVERLAY_STEM)
    if clean_path is None or adv_path is None:
        raise FileNotFoundError(f"Images not found for {OVERLAY_STEM}/{attack}")
    clean_fd = dets_all.get("clean", {}).get(OVERLAY_STEM)
    adv_fd   = dets_all.get(attack, {}).get(OVERLAY_STEM)
    return detection_overlay.overlay_clean_vs_adv(
        clean_path, adv_path,
        clean_dets=clean_fd, adv_dets=adv_fd,
        attack_name=attack,
        class_names=DETRAC_CLASSES,
    )


def make_f07(dets_all, **_):
    return _make_overlay("latent", dets_all)


def make_f08(dets_all, **_):
    return _make_overlay("pgd", dets_all)


def make_f09(dets_all, **_):
    return _make_overlay("fgsm", dets_all)


def make_f10(dets_all, per_frame_full, **_):
    attacks_present = [a for a in ATTACK_ORDER if ADV_DIRS[a].exists()]
    clean_fd = dets_all.get("clean")
    adv_fds  = {a: dets_all[a] for a in attacks_present if a in dets_all}
    return grids.attack_comparison_grid(
        clean_dir=CLEAN_DIR,
        adv_dirs={a: ADV_DIRS[a] for a in attacks_present},
        stems=GRID_STEMS,
        clean_dets=clean_fd,
        adv_dets_dict=adv_fds,
        per_frame_metrics=per_frame_full,
        class_names=DETRAC_CLASSES,
    )


def make_f11(dets_all, **_):
    clean_fd = dets_all.get("clean")
    if not clean_fd:
        raise FileNotFoundError("dets_clean.json not available")
    adv_fds = {a: dets_all[a] for a in ATTACK_ORDER if a in dets_all}
    if not adv_fds:
        raise FileNotFoundError("No adversarial dets loaded")
    return analysis_plots.class_breakdown_chart(
        clean_dets=clean_fd,
        adv_dets_dict=adv_fds,
        class_names=DETRAC_CLASSES,
    )


def make_f12(per_frame_full, per_image, **_):
    # Use per_frame from metrics_full.json if available, else fall back to per_image
    if per_frame_full:
        # Build a per_image-compatible dict from per_frame_full
        data = {
            atk: list(per_frame_full[atk].values())
            for atk in ATTACK_ORDER if atk in per_frame_full
        }
    else:
        data = per_image
    if not data:
        raise ValueError("No per-frame data available for timeseries")
    return analysis_plots.detection_timeseries(data, highlight_stems=BINARY_STEMS)


def make_f13(dets_all, **_):
    clean_fd = dets_all.get("clean")
    if not clean_fd:
        raise FileNotFoundError("dets_clean.json not available")
    adv_fds = {a: dets_all[a] for a in ATTACK_ORDER if a in dets_all}
    return analysis_plots.iou_distribution(
        clean_dets=clean_fd,
        adv_dets_dict=adv_fds,
        iou_thr=0.3,
    )


# ---------------------------------------------------------------------------
# HTML index
# ---------------------------------------------------------------------------


def write_index(figure_info: list[tuple[str, str, bool]], fig_dir: Path) -> None:
    rows = ""
    for tag, title, ok in figure_info:
        status = "OK" if ok else "SKIP"
        png_rel = f"png/{tag}.png"
        pdf_rel = f"pdf/{tag}.pdf"
        if ok:
            rows += (
                f"<tr><td>{status}</td><td>{title}</td>"
                f'<td><a href="{png_rel}"><img src="{png_rel}" width="260"></a></td>'
                f'<td><a href="{pdf_rel}">PDF</a></td></tr>\n'
            )
        else:
            rows += f"<tr><td>{status}</td><td>{title}</td><td>—</td><td>—</td></tr>\n"

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Adversarial Evaluation Figures v2</title>
<style>
  body {{ font-family: sans-serif; max-width: 1100px; margin: 2em auto; }}
  table {{ border-collapse: collapse; width: 100%; }}
  td, th {{ border: 1px solid #ddd; padding: 6px 10px; vertical-align: top; }}
  th {{ background: #f4f4f4; }}
  img {{ border: 1px solid #ccc; }}
</style>
</head>
<body>
<h1>Adversarial Evaluation Figures (v2)</h1>
<p>Auto-generated by <code>scripts/generate_figures.py</code>.</p>
<table>
<tr><th>Status</th><th>Figure</th><th>Preview (PNG)</th><th>Vector</th></tr>
{rows}
</table>
</body>
</html>
"""
    with open(fig_dir / "index.html", "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"  index.html -> {fig_dir / 'index.html'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

FIGURE_DEFS: list[tuple[str, str]] = [
    ("f01_dfr_distribution",       "Per-frame DFR distribution (violin + mean + CI)"),
    ("f02_nclean_vs_nadv",         "n_clean vs n_adv scatter (jittered, region counts)"),
    ("f03_metric_barchart",        "DFR / ASR bar charts with 95% CI error bars"),
    ("f04_stealth_vs_eff",         "Stealth vs effectiveness (PSNR vs DFR, CI bars)"),
    ("f05_perturbation_heatmap",   "Perturbation heatmap 2x2 grid (shared vmax)"),
    ("f06_difference_grid",        "Difference grid (per-attack calibrated amplification)"),
    ("f07_overlay_latent",         "Detection overlay: Clean vs LATENT with boxes"),
    ("f08_overlay_pgd",            "Detection overlay: Clean vs PGD with boxes"),
    ("f09_overlay_fgsm",           "Detection overlay: Clean vs FGSM with boxes"),
    ("f10_qualitative_grid",       "Qualitative grid: 4 frames x 4 cols, boxes drawn"),
    ("f11_class_breakdown",        "Per-class detection survival rates"),
    ("f12_timeseries",             "Per-frame detection count timeseries"),
    ("f13_iou_distribution",       "IoU distribution of matched detection pairs"),
]

MAKER_MAP = {
    "f01_dfr_distribution":       make_f01,
    "f02_nclean_vs_nadv":         make_f02,
    "f03_metric_barchart":        make_f03,
    "f04_stealth_vs_eff":         make_f04,
    "f05_perturbation_heatmap":   make_f05,
    "f06_difference_grid":        make_f06,
    "f07_overlay_latent":         make_f07,
    "f08_overlay_pgd":            make_f08,
    "f09_overlay_fgsm":           make_f09,
    "f10_qualitative_grid":       make_f10,
    "f11_class_breakdown":        make_f11,
    "f12_timeseries":             make_f12,
    "f13_iou_distribution":       make_f13,
}


def main(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    png_dir = out_dir / "png"
    pdf_dir = out_dir / "pdf"
    setup_publication_style()

    print("[INFO] Loading data ...")
    per_image    = load_per_image_data()
    summary      = load_summary_data()
    dets_all     = load_dets()
    per_frame_full = load_per_frame_metrics_full()

    ctx = dict(
        per_image=per_image,
        summary=summary,
        dets_all=dets_all,
        per_frame_full=per_frame_full,
    )

    print(f"[INFO] Writing figures to {out_dir}")
    figures: list[tuple[str, str, bool]] = []

    for tag, title in FIGURE_DEFS:
        fn = MAKER_MAP[tag]
        try:
            fig = fn(**ctx)
            _save(fig, tag, png_dir, pdf_dir)
            ok = True
        except Exception as exc:
            print(f"  [ERROR] {tag}: {exc}")
            if "--verbose" in sys.argv:
                traceback.print_exc()
            ok = False
        status = "OK" if ok else "SKIP"
        print(f"  [{status}] {tag}")
        figures.append((tag, title, ok))

    write_index(figures, out_dir)

    n_ok = sum(1 for _, _, ok in figures if ok)
    print(f"\n[DONE] {n_ok}/{len(figures)} figures generated -> {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR / "figures")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    main(args.out_dir)
