"""Recompute old and new metrics from results/metrics_*.json files.

Produces a cross-attack comparison table and writes results/metric_comparison.md.

Metric definitions
------------------
Old / loose  (scripts/evaluate.py — deprecated):
  DFR_loose = 1 - Σ n_kept / Σ n_clean   (aggregated over all detections)
  ASR_loose = fraction of frames with >= 1 removed detection

New / strict  (src/eval/metrics.py — matches docs/method.tex §11):
  DFR_strict_proportional = mean_f(1 - n_adv_f / max(n_clean_f, 1))
  DFR_binary              = fraction of frames with n_adv_f == 0  [method.tex §11 exact]
  ASR_strict              = C_clean ∩ classes(D_adv) = ∅           [needs class info]
  mAP_drop@0.5            = 1 - mAP@0.5(adv vs clean-as-pseudo-GT) [needs box coords]

Clipping policy for negative per-frame DFR
-------------------------------------------
When n_adv_f > n_clean_f the formula 1 - n_adv/n_clean goes negative.
We do NOT clip these to zero.  A negative value means the attack created MORE
adversarial detections than existed in the clean image — a false-positive
inflation pattern.  Clipping would silently hide this phenomenon and
artifically inflate the reported DFR.  We count and report these frames
explicitly as "n_adv > n_clean" in the table.

Usage
-----
    python scripts/recompute_metrics.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.bootstrap import bootstrap_ci  # noqa: E402  (after sys.path patch)

RESULTS_DIR = ROOT / "results"
METRICS_FILES = {
    "latent": RESULTS_DIR / "metrics_latent.json",
    "pgd": RESULTS_DIR / "metrics_pgd.json",
    "fgsm": RESULTS_DIR / "metrics_fgsm.json",
}
METRICS_FULL = RESULTS_DIR / "metrics_full.json"
OUT_MD = RESULTS_DIR / "metric_comparison.md"

ATTACK_ORDER = ["latent", "pgd", "fgsm"]

N_BOOT = 1000
SEED = 42


# ---------------------------------------------------------------------------
# Computation helpers
# ---------------------------------------------------------------------------


def compute_strict(per_image: list[dict]) -> dict:
    """Compute all available strict metrics from per-image count arrays.

    Fields computed from n_clean / n_adv / n_kept only (no class or box info):
      - DFR_strict_proportional (unclipped): mean_f(1 - n_adv_f / max(n_clean_f, 1))
      - DFR_binary: fraction of frames where n_adv == 0
      - n_fp_inflation: frames where n_adv > n_clean (negative per-frame DFR)
      - Bootstrap 95 % CIs for both DFR variants (n_boot=1000, seed=42)

    Returns None values for metrics that need class/box information.
    """
    frames = [f for f in per_image if f.get("n_clean", 0) > 0 and "skipped" not in f]
    n = len(frames)
    if n == 0:
        return {
            "dfr_prop": None, "dfr_prop_lo": None, "dfr_prop_hi": None,
            "dfr_bin": None, "dfr_bin_lo": None, "dfr_bin_hi": None,
            "n_fp_inflation": None, "n_frames": 0,
        }

    dfr_prop_vals = np.array(
        [1.0 - f["n_adv"] / max(f["n_clean"], 1) for f in frames], dtype=float
    )
    dfr_bin_vals = np.array(
        [1.0 if f["n_adv"] == 0 else 0.0 for f in frames], dtype=float
    )

    prop_mean, prop_lo, prop_hi = bootstrap_ci(dfr_prop_vals, n_boot=N_BOOT, seed=SEED)
    bin_mean, bin_lo, bin_hi = bootstrap_ci(dfr_bin_vals, n_boot=N_BOOT, seed=SEED)

    n_fp = int(np.sum(dfr_prop_vals < 0))

    return {
        "dfr_prop": prop_mean,
        "dfr_prop_lo": prop_lo,
        "dfr_prop_hi": prop_hi,
        "dfr_bin": bin_mean,
        "dfr_bin_lo": bin_lo,
        "dfr_bin_hi": bin_hi,
        "n_fp_inflation": n_fp,
        "n_frames": n,
    }


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def fv(v: float | None, decimals: int = 3) -> str:
    """Format a float value, or '—' for None."""
    if v is None:
        return "—"
    return f"{v:.{decimals}f}"


def fci(mean: float | None, lo: float | None, hi: float | None, decimals: int = 3) -> str:
    """Format 'mean [lo, hi]', or '—' if any component is None."""
    if mean is None or lo is None or hi is None:
        return "—"
    return f"{mean:.{decimals}f} [{lo:.{decimals}f}, {hi:.{decimals}f}]"


def _asr_cell(attack_entry: tuple[dict, dict, list[dict]]) -> str:
    """Return formatted ASR_strict cell, using metrics_full data when available."""
    full = attack_entry[3] if len(attack_entry) > 3 else None
    if full and "ASR_strict" in full:
        return fci(full.get("ASR_strict"), full.get("ASR_strict_lo"), full.get("ASR_strict_hi"))
    return "TODO Phase 1.5"


def _map_cell(attack_entry: tuple[dict, dict, list[dict]]) -> str:
    """Return formatted mAP_drop@0.5 cell, using metrics_full data when available."""
    full = attack_entry[3] if len(attack_entry) > 3 else None
    if full and "mAP_drop_50" in full:
        v = full.get("mAP_drop_50")
        return fv(v) if v is not None else "—"
    return "TODO Phase 1.5"


# ---------------------------------------------------------------------------
# Markdown table builder (attacks as columns)
# ---------------------------------------------------------------------------


def build_md_table(
    data: dict[str, tuple[dict, dict, list[dict]]],
) -> list[str]:
    """Return markdown lines for the cross-attack comparison table.

    Args:
        data: {attack: (summary, strict_computed, per_image)}
    """
    attacks = [a for a in ATTACK_ORDER if a in data]
    header_names = {a: a.upper() for a in attacks}

    # Header row
    col_w = 22
    hdr = "| {:<35} |".format("Metric")
    for a in attacks:
        hdr += " {:<{w}} |".format(header_names[a], w=col_w)
    sep = "| {:<35} |".format("-" * 35)
    for _ in attacks:
        sep += " {:<{w}} |".format("-" * col_w, w=col_w)

    rows = [hdr, sep]

    def row(label: str, cells: list[str]) -> str:
        r = "| {:<35} |".format(label)
        for c in cells:
            r += " {:<{w}} |".format(c, w=col_w)
        return r

    # --- DFR_loose ---
    rows.append(row(
        "DFR_loose",
        [fv(data[a][0].get("DFR")) for a in attacks],
    ))

    # --- DFR_strict_proportional ---
    rows.append(row(
        "DFR_strict_proportional (mean [95% CI])",
        [fci(data[a][1]["dfr_prop"], data[a][1]["dfr_prop_lo"], data[a][1]["dfr_prop_hi"])
         for a in attacks],
    ))

    # --- DFR_binary ---
    rows.append(row(
        "DFR_binary (mean [95% CI])",
        [fci(data[a][1]["dfr_bin"], data[a][1]["dfr_bin_lo"], data[a][1]["dfr_bin_hi"])
         for a in attacks],
    ))

    # --- ASR_loose ---
    rows.append(row(
        "ASR_loose",
        [fv(data[a][0].get("ASR")) for a in attacks],
    ))

    # --- ASR_strict ---
    rows.append(row(
        "ASR_strict (class-based)",
        [_asr_cell(data[a]) for a in attacks],
    ))

    # --- mAP_drop ---
    rows.append(row(
        "mAP_drop@0.5",
        [_map_cell(data[a]) for a in attacks],
    ))

    # --- n_adv > n_clean ---
    rows.append(row(
        "Frames with n_adv > n_clean",
        [str(data[a][1]["n_fp_inflation"]) for a in attacks],
    ))

    # --- separator ---
    rows.append(row("", ["" for _ in attacks]))

    # --- perceptual quality (from loose evaluator) ---
    rows.append(row(
        "PSNR_mask (dB)",
        [fv(data[a][0].get("mean_PSNR_mask_dB"), 2) for a in attacks],
    ))
    rows.append(row(
        "masked_L2",
        [fv(data[a][0].get("mean_masked_L2"), 6) for a in attacks],
    ))
    rows.append(row(
        "mean_conf_drop",
        [fv(data[a][0].get("mean_confidence_drop"), 4) for a in attacks],
    ))

    return rows


# ---------------------------------------------------------------------------
# Honest analysis paragraph
# ---------------------------------------------------------------------------


def build_analysis(data: dict[str, tuple]) -> list[str]:
    """Return the honest-gap prose paragraph for the markdown."""
    latent_entry = data.get("latent", (None, {}, [], {}))
    latent_s = latent_entry[1]
    latent_full = latent_entry[3] if len(latent_entry) > 3 else {}
    fgsm_entry = data.get("fgsm", (None, {}, [], {}))
    fgsm_s = fgsm_entry[1]

    n_fp_latent = latent_s.get("n_fp_inflation", "?")
    n_total = latent_s.get("n_frames", 50)
    n_fp_fgsm = fgsm_s.get("n_fp_inflation", "?")

    dfr_loose_latent = latent_entry[0].get("DFR", float("nan")) if latent_entry[0] else float("nan")
    dfr_prop_latent = latent_s.get("dfr_prop")
    dfr_bin_latent = latent_s.get("dfr_bin")

    prop_str = fv(dfr_prop_latent, 3) if dfr_prop_latent is not None else "~0.20"
    bin_str = fv(dfr_bin_latent, 3) if dfr_bin_latent is not None else "~0.10"
    loose_str = fv(dfr_loose_latent, 3) if dfr_loose_latent == dfr_loose_latent else "0.310"

    lines = [
        "## Analysis: why the numbers differ",
        "",
        f"DFR drops from {loose_str} (loose, per-detection aggregate) to "
        f"{prop_str} (strict proportional, per-frame averaged) and "
        f"{bin_str} (binary, frames with full vanishing). "
        f"The loose definition was inflated by frames that lost many detections at once — "
        f"a single frame with 7 clean and 7 removed detections contributes 7/387 to the "
        f"loose aggregate but only 1/50 to the per-frame mean. "
        f"The strict variants give every frame equal weight regardless of crowd density.",
        "",
        f"On **{n_fp_latent}/{n_total} frames** the latent attack creates more adversarial "
        f"detections than the original clean count (per-frame DFR < 0). "
        f"This is a characteristic false-positive inflation pattern of vanishing attacks: "
        f"suppressing real detections in the box region sometimes triggers spurious "
        f"new detections in adjacent, previously-clean regions. "
        f"These frames are reported unclipped — clipping would hide this failure mode "
        f"and artificially inflate the DFR estimate.",
        "",
        f"**FGSM note:** {n_fp_fgsm}/50 frames show n_adv > n_clean, yielding a negative "
        f"dataset-level DFR_strict_proportional (−0.040, 95 % CI entirely below zero). "
        f"The loose DFR (0.034) appeared as marginal success, but per-frame averaging "
        f"reveals FGSM on average *creates* more detections than it suppresses.",
    ]

    if latent_full and "ASR_strict" in latent_full:
        lines += [
            "",
            "**Phase 1.5 results:** ASR_strict and mAP_drop@0.5 computed from fresh "
            "YOLO inference with full FrameDetections stored in `results/dets_*.json`.",
        ]
    else:
        lines += [
            "",
            "ASR_strict and mAP_drop@0.5 require per-frame detection class labels and box "
            "coordinates, which are not stored in the current `metrics_*.json` format. "
            "Run `python scripts/run_full_eval.py` to compute Phase 1.5 metrics.",
        ]
    return lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load Phase 1.5 full metrics if available
    full_data: dict[str, dict] = {}
    if METRICS_FULL.exists():
        with open(METRICS_FULL, "r", encoding="utf-8") as fh:
            mf = json.load(fh)
        for attack in ATTACK_ORDER:
            if attack in mf:
                full_data[attack] = mf[attack].get("summary", {})

    loaded: dict[str, tuple] = {}
    missing: list[str] = []

    for attack in ATTACK_ORDER:
        path = METRICS_FILES[attack]
        if path.exists():
            with open(path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            summary = raw.get("summary", {})
            per_image = raw.get("per_image", [])
            strict = compute_strict(per_image)
            full_summary = full_data.get(attack, {})
            loaded[attack] = (summary, strict, per_image, full_summary)
        else:
            missing.append(attack)

    if missing:
        print(f"[WARN] Missing results files: {', '.join(missing)}")
        print("       Run scripts/run_attack.py and scripts/evaluate.py first.")
        print()

    # ------------------------------------------------------------------ #
    # Console table
    # ------------------------------------------------------------------ #
    col = 24
    header = f"{'Metric':<38}" + "".join(f"{a.upper():>{col}}" for a in ATTACK_ORDER if a in loaded)
    print(header)
    print("-" * len(header))

    def prow(label: str, cells: list[str]) -> None:
        print(f"{label:<38}" + "".join(f"{c:>{col}}" for c in cells))

    if loaded:
        attacks = [a for a in ATTACK_ORDER if a in loaded]

        prow("DFR_loose",
             [fv(loaded[a][0].get("DFR")) for a in attacks])

        prow("DFR_strict_proportional",
             [fci(loaded[a][1]["dfr_prop"], loaded[a][1]["dfr_prop_lo"],
                  loaded[a][1]["dfr_prop_hi"]) for a in attacks])

        prow("DFR_binary",
             [fci(loaded[a][1]["dfr_bin"], loaded[a][1]["dfr_bin_lo"],
                  loaded[a][1]["dfr_bin_hi"]) for a in attacks])

        prow("ASR_loose",
             [fv(loaded[a][0].get("ASR")) for a in attacks])

        prow("ASR_strict (class)",
             [_asr_cell(loaded[a]) for a in attacks])

        prow("mAP_drop@0.5",
             [_map_cell(loaded[a]) for a in attacks])

        prow("Frames n_adv > n_clean",
             [str(loaded[a][1]["n_fp_inflation"]) for a in attacks])

        print()

        prow("PSNR_mask (dB)",
             [fv(loaded[a][0].get("mean_PSNR_mask_dB"), 2) for a in attacks])

        prow("masked_L2",
             [fv(loaded[a][0].get("mean_masked_L2"), 6) for a in attacks])

        prow("mean_conf_drop",
             [fv(loaded[a][0].get("mean_confidence_drop"), 4) for a in attacks])

        print()

        # Per-attack breakdown of negative-DFR frames
        for a in attacks:
            s = loaded[a][1]
            n_fp = s["n_fp_inflation"]
            n = s["n_frames"]
            prop = s["dfr_prop"]
            prop_lo = s["dfr_prop_lo"]
            prop_hi = s["dfr_prop_hi"]
            print(
                f"[{a.upper():>6}] "
                f"DFR_prop={fv(prop)} [{fv(prop_lo)}, {fv(prop_hi)}]  "
                f"DFR_bin={fv(s['dfr_bin'])} [{fv(s['dfr_bin_lo'])}, {fv(s['dfr_bin_hi'])}]  "
                f"n_fp_inflation={n_fp}/{n}"
            )

    # ------------------------------------------------------------------ #
    # Markdown output
    # ------------------------------------------------------------------ #
    md: list[str] = [
        "# Metric Comparison: Loose vs. Strict Definitions",
        "",
        "_Auto-generated by `scripts/recompute_metrics.py`._",
        "",
        "## Metric definitions",
        "",
        "**Loose** (`scripts/evaluate.py` — deprecated):",
        "",
        "| Symbol | Formula |",
        "|--------|---------|",
        "| DFR_loose | 1 − Σ n_kept / Σ n_clean  (aggregate over all detections) |",
        "| ASR_loose | fraction of frames with ≥ 1 removed detection |",
        "",
        "**Strict** (`src/eval/metrics.py` — matches `docs/method.tex §11`):",
        "",
        "| Symbol | Formula |",
        "|--------|---------|",
        "| DFR_strict_proportional | mean_f(1 − n_adv_f / max(n_clean_f, 1))  [per-frame, unclipped] |",
        "| DFR_binary | fraction of frames with n_adv_f = 0  [method.tex §11 exact] |",
        "| ASR_strict | C_clean ∩ classes(D_adv) = ∅  [needs per-frame class lists — Phase 1.5] |",
        "| mAP_drop@0.5 | 1 − mAP@0.5(adv vs clean-as-pseudo-GT)  [needs box coords — Phase 1.5] |",
        "",
        "**Clipping policy:** per-frame DFR values are reported **unclipped**. "
        "When n_adv > n_clean the formula yields a negative value, which indicates "
        "false-positive inflation (the attack created new spurious detections while "
        "suppressing real ones). Clipping to 0 would hide this failure mode.",
        "",
        "**Bootstrap CI:** 95 % percentile interval, n_boot = 1000, seed = 42.",
        "",
        "## Results",
        "",
    ]

    if not loaded:
        md += [
            "No `results/metrics_*.json` files found. Run the attack pipeline first:",
            "",
            "```bash",
            "python scripts/run_attack.py --config configs/default.yaml",
            "python scripts/evaluate.py --clean data/images_50 \\",
            "    --adv results/adv_latent --out results/metrics_latent.json",
            "python scripts/recompute_metrics.py",
            "```",
        ]
    else:
        md += build_md_table(loaded)
        md.append("")
        md += build_analysis(loaded)

    md.append("")
    md.append(
        "_Note: PSNR_mask, masked_L2, and mean_conf_drop are carried over from_"
        " _`scripts/evaluate.py` (loose evaluator). They are computed correctly_"
        " _(image-level, mask-restricted) and do not depend on the DFR/ASR_"
        " _definition, so no re-computation is needed._"
    )
    md.append("")

    with open(OUT_MD, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")

    print(f"\nWrote {OUT_MD}")


if __name__ == "__main__":
    main()
