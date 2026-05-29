"""Held-out evaluation of the Win Probability classifier.

Train WP on deliveries before a cutoff date, score WP on later deliveries,
report calibration metrics against actual match outcomes.

This is the foundational validation: every WAR number we ship depends on the
WP model being well-calibrated. If WP(state) = 0.7 but batting team actually
wins from such states only 50% of the time, every Δ_WP downstream is wrong.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import duckdb
import numpy as np
import pandas as pd

from t20x.ratings.win_probability import (
    WP_FEATURES,
    WinProbabilityModel,
    _state_frame,
    load_wpa_frame,
)
from t20x.validation.calibration import (
    baseline_constant,
    calibration_table,
    summarize,
)


@dataclass
class WPHoldoutResult:
    cutoff: dt.date
    n_train: int
    n_test: int
    test_metrics: dict[str, float]
    baseline_metrics: dict[str, float]
    calibration: pd.DataFrame
    model: WinProbabilityModel


def evaluate_wp_holdout(
    conn: duckdb.DuckDBPyConnection,
    cutoff: dt.date,
    league: str | None = None,
    n_bins: int = 10,
) -> WPHoldoutResult:
    """Train WP on pre-cutoff deliveries, evaluate on post-cutoff deliveries.

    Returns calibration metrics on the *test set*. The test-set labels are the
    actual match winners (each delivery gets the binary "did the batting team
    in this innings win this match" label).
    """
    df = load_wpa_frame(conn, league=league)
    if df.empty:
        raise RuntimeError("No data returned from load_wpa_frame.")

    # Need a per-delivery date. load_wpa_frame doesn't include match date today,
    # so we join it from `matches`.
    dates = conn.execute(
        "SELECT match_id, date FROM matches WHERE winner IS NOT NULL AND winner != ''"
    ).fetchdf()
    df = df.merge(dates, on="match_id", how="left")
    df["date"] = pd.to_datetime(df["date"])
    cutoff_ts = pd.Timestamp(cutoff)
    train_mask = df["date"] < cutoff_ts
    test_mask = ~train_mask
    if not train_mask.any() or not test_mask.any():
        raise RuntimeError(f"Empty train or test set at cutoff {cutoff}.")

    df_train = df[train_mask]
    df_test = df[test_mask]

    X_train = _state_frame(df_train, "post")
    y_train = df_train["batting_team_won"].to_numpy()
    model = WinProbabilityModel().fit(X_train, y_train)

    X_test = _state_frame(df_test, "post")
    y_test = df_test["batting_team_won"].to_numpy().astype(int)
    p_test = model.predict_wp(X_test)

    test_metrics = summarize(p_test, y_test, n_bins=n_bins)
    baseline_metrics = baseline_constant(float(y_train.mean()), y_test)
    cal = calibration_table(p_test, y_test, n_bins=n_bins)

    return WPHoldoutResult(
        cutoff=cutoff,
        n_train=int(train_mask.sum()),
        n_test=int(test_mask.sum()),
        test_metrics=test_metrics,
        baseline_metrics=baseline_metrics,
        calibration=cal,
        model=model,
    )
