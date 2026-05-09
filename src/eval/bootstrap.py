"""Bootstrap confidence-interval utilities.

All functions use np.random.default_rng(seed) for reproducibility.
"""
from __future__ import annotations

import numpy as np


def bootstrap_ci(
    values: np.ndarray,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap CI for the mean of *values*.

    Draws *n_boot* bootstrap samples (with replacement) from *values*,
    computes the mean of each, and returns the (1−ci)/2 and (1+ci)/2
    percentiles of that distribution as the confidence interval.

    Args:
        values: 1-D array of observed values.
        n_boot: Number of bootstrap replicates.
        ci:     Nominal coverage (e.g. 0.95 for a 95 % CI).
        seed:   RNG seed for reproducibility (np.random.default_rng).

    Returns:
        (mean, lo, hi) where mean is the sample mean and [lo, hi] is the
        bootstrap percentile interval at the requested coverage.
    """
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))

    boot_means = np.empty(n_boot, dtype=float)
    for k in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        boot_means[k] = float(np.mean(sample))

    alpha = (1.0 - ci) / 2.0
    lo = float(np.percentile(boot_means, alpha * 100.0))
    hi = float(np.percentile(boot_means, (1.0 - alpha) * 100.0))
    return mean, lo, hi


def bootstrap_metric_dict(
    per_frame: list[dict],
    keys: list[str],
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, tuple[float, float, float]]:
    """Bootstrap CI for each requested metric key across frames.

    Filters out None and non-finite values before bootstrapping.

    Args:
        per_frame: List of per-frame metric dicts (same format as aggregate()).
        keys:      Metric names to process.
        n_boot:    Bootstrap replicates per metric.
        seed:      RNG seed (passed to bootstrap_ci).

    Returns:
        Dict mapping key → (mean, lo, hi). Keys with no valid values are
        omitted.
    """
    result: dict[str, tuple[float, float, float]] = {}
    for key in keys:
        raw = [f[key] for f in per_frame if key in f and f[key] is not None]
        vals = np.array(
            [float(v) for v in raw if np.isfinite(float(v))], dtype=float
        )
        if len(vals) == 0:
            continue
        result[key] = bootstrap_ci(vals, n_boot=n_boot, seed=seed)
    return result
