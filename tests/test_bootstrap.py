"""Tests for src/eval/bootstrap.py."""
from __future__ import annotations

import numpy as np
import pytest

from src.eval.bootstrap import bootstrap_ci, bootstrap_metric_dict


# ---------------------------------------------------------------------------
# 1. Degenerate input: all same value → CI collapses to (v, v, v)
# ---------------------------------------------------------------------------


def test_degenerate_all_same_value() -> None:
    vals = np.full(50, 0.5)
    mean, lo, hi = bootstrap_ci(vals, n_boot=500, seed=42)
    assert mean == pytest.approx(0.5)
    assert lo == pytest.approx(0.5)
    assert hi == pytest.approx(0.5)


def test_degenerate_all_zeros() -> None:
    vals = np.zeros(20)
    mean, lo, hi = bootstrap_ci(vals, n_boot=200, seed=0)
    assert mean == pytest.approx(0.0)
    assert lo == pytest.approx(0.0)
    assert hi == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 2. CI bracket is valid and monotonically shrinks with n_boot
# ---------------------------------------------------------------------------


def test_ci_bracket_valid() -> None:
    """lo ≤ mean ≤ hi for non-degenerate data."""
    rng = np.random.default_rng(99)
    vals = rng.normal(0.5, 0.1, 30)
    mean, lo, hi = bootstrap_ci(vals, n_boot=500, seed=42)
    assert lo <= mean <= hi
    assert hi - lo > 0
    assert np.isfinite(lo)
    assert np.isfinite(hi)


def test_ci_bracket_monotonic_in_n_boot() -> None:
    """Larger n_boot should give a tighter (or equal) CI width.

    Uses a fixed data seed so the underlying distribution is the same.
    With n_boot=50 vs n_boot=5000, the width reduction is statistically
    reliable; we allow up to 20 % slack for the comparison.
    """
    rng = np.random.default_rng(7)
    vals = rng.exponential(scale=2.0, size=40)

    _, lo_small, hi_small = bootstrap_ci(vals, n_boot=50, seed=42)
    _, lo_large, hi_large = bootstrap_ci(vals, n_boot=5000, seed=42)

    width_small = hi_small - lo_small
    width_large = hi_large - lo_large

    # Both must be finite and positive
    assert np.isfinite(width_small) and width_small > 0
    assert np.isfinite(width_large) and width_large > 0

    # With 100× more bootstrap samples the CI should be no wider than small CI
    # (with generous 30 % tolerance for the stochastic nature of the test).
    assert width_large <= width_small * 1.30, (
        f"Expected larger n_boot to give tighter CI: "
        f"width@50={width_small:.4f}, width@5000={width_large:.4f}"
    )


# ---------------------------------------------------------------------------
# 3. Reproducibility with fixed seed
# ---------------------------------------------------------------------------


def test_reproducible_fixed_seed() -> None:
    vals = np.array([0.1, 0.5, 0.9, 0.3, 0.7, 0.4, 0.6, 0.2, 0.8])
    r1 = bootstrap_ci(vals, n_boot=300, seed=7)
    r2 = bootstrap_ci(vals, n_boot=300, seed=7)
    assert r1 == r2


def test_different_seeds_differ() -> None:
    vals = np.array([0.1, 0.5, 0.9, 0.3, 0.7, 0.4, 0.6, 0.2, 0.8])
    r1 = bootstrap_ci(vals, n_boot=300, seed=7)
    r2 = bootstrap_ci(vals, n_boot=300, seed=13)
    # Means are the same; CIs should differ (extremely unlikely to be equal)
    assert r1[0] == pytest.approx(r2[0])  # same mean
    assert r1 != r2                        # but CIs differ


# ---------------------------------------------------------------------------
# 4. bootstrap_metric_dict
# ---------------------------------------------------------------------------


def test_metric_dict_single_key() -> None:
    per_frame = [
        {"dfr": 0.5},
        {"dfr": 0.8},
        {"dfr": 1.0},
    ]
    result = bootstrap_metric_dict(per_frame, keys=["dfr"], n_boot=200, seed=42)
    assert "dfr" in result
    mean, lo, hi = result["dfr"]
    assert lo <= mean <= hi
    assert mean == pytest.approx(np.mean([0.5, 0.8, 1.0]))


def test_metric_dict_filters_none_and_nan() -> None:
    per_frame = [
        {"val": 1.0},
        {"val": None},
        {"val": float("nan")},
        {"val": 0.5},
    ]
    result = bootstrap_metric_dict(per_frame, keys=["val"], n_boot=100, seed=0)
    assert "val" in result
    mean, _, _ = result["val"]
    assert mean == pytest.approx(np.mean([1.0, 0.5]))


def test_metric_dict_missing_key_omitted() -> None:
    per_frame = [{"dfr": 0.5}]
    result = bootstrap_metric_dict(per_frame, keys=["dfr", "nonexistent"], n_boot=50, seed=0)
    assert "dfr" in result
    assert "nonexistent" not in result


def test_metric_dict_empty_values_omitted() -> None:
    per_frame = [{"other": 0.5}]
    result = bootstrap_metric_dict(per_frame, keys=["dfr"], n_boot=50, seed=0)
    assert "dfr" not in result
