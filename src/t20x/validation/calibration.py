"""Calibration metrics for binary probability predictions.

Used to evaluate the Win Probability classifier (and any downstream binary
model). Standard metrics:
    - Brier score
    - Log loss
    - Expected Calibration Error (ECE)
    - Per-decile calibration table
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-12


def brier_score(p: np.ndarray, y: np.ndarray) -> float:
    """Mean squared error of predicted probability vs binary outcome."""
    return float(np.mean((p - y) ** 2))


def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    """Average negative log-likelihood. Lower is better. 0 = perfect."""
    p_clipped = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p_clipped) + (1 - y) * np.log(1 - p_clipped)))


def expected_calibration_error(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """Average absolute gap between mean predicted and mean actual within bins,
    weighted by bin population. 0 = perfectly calibrated.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(p, bins) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    total = len(p)
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        n = mask.sum()
        if n == 0:
            continue
        gap = abs(p[mask].mean() - y[mask].mean())
        ece += (n / total) * gap
    return float(ece)


def calibration_table(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Per-bin calibration breakdown for plotting / printing."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(p, bins) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = idx == b
        n = int(mask.sum())
        if n == 0:
            rows.append((b, bins[b], bins[b + 1], 0, np.nan, np.nan, np.nan))
            continue
        pred_mean = float(p[mask].mean())
        actual_mean = float(y[mask].mean())
        rows.append((b, bins[b], bins[b + 1], n, pred_mean, actual_mean, pred_mean - actual_mean))
    return pd.DataFrame(
        rows,
        columns=["bin", "lo", "hi", "count", "pred_mean", "actual_mean", "gap"],
    )


def baseline_constant(p_const: float, y: np.ndarray) -> dict[str, float]:
    """Metrics for a constant-probability baseline."""
    p = np.full_like(y, p_const, dtype="float64")
    return {
        "brier": brier_score(p, y),
        "log_loss": log_loss(p, y),
    }


def summarize(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> dict[str, float]:
    """One-call summary of all calibration metrics."""
    return {
        "brier": brier_score(p, y),
        "log_loss": log_loss(p, y),
        "ece": expected_calibration_error(p, y, n_bins=n_bins),
        "n": int(len(p)),
        "base_rate": float(y.mean()),
        "pred_mean": float(p.mean()),
    }
