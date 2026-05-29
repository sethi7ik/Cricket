"""Cross-league transfer test for the Win Probability model.

Does a WP model trained on one league's deliveries calibrate well on another?
If yes, the model captures universal T20 win-dynamics. If no, league-specific
calibration is needed (different pitches, scoring environments, etc.).

We train WP on a source league (e.g., IPL) and evaluate calibration on a target
league (e.g., BBL). Compare to a within-league baseline trained on the target.
"""

from __future__ import annotations

from dataclasses import dataclass

import duckdb

from t20x.ratings.win_probability import (
    WinProbabilityModel,
    _state_frame,
    load_wpa_frame,
)
from t20x.validation.calibration import baseline_constant, summarize


@dataclass
class CrossLeagueResult:
    source_league: str
    target_league: str
    n_source: int
    n_target: int
    transfer_metrics: dict[str, float]
    within_metrics: dict[str, float]
    baseline_metrics: dict[str, float]


def evaluate_cross_league(
    conn: duckdb.DuckDBPyConnection,
    source_league: str,
    target_league: str,
    n_bins: int = 10,
) -> CrossLeagueResult:
    df_source = load_wpa_frame(conn, league=source_league)
    df_target = load_wpa_frame(conn, league=target_league)
    if df_source.empty or df_target.empty:
        raise RuntimeError("Empty source or target league frame.")

    X_src = _state_frame(df_source, "post")
    y_src = df_source["batting_team_won"].to_numpy().astype(int)
    X_tgt = _state_frame(df_target, "post")
    y_tgt = df_target["batting_team_won"].to_numpy().astype(int)

    # Train on source, predict target → transfer test
    transfer_model = WinProbabilityModel().fit(X_src, y_src)
    p_transfer = transfer_model.predict_wp(X_tgt)

    # Train on target itself for within-league reference
    within_model = WinProbabilityModel().fit(X_tgt, y_tgt)
    p_within = within_model.predict_wp(X_tgt)

    return CrossLeagueResult(
        source_league=source_league,
        target_league=target_league,
        n_source=len(df_source),
        n_target=len(df_target),
        transfer_metrics=summarize(p_transfer, y_tgt, n_bins=n_bins),
        within_metrics=summarize(p_within, y_tgt, n_bins=n_bins),
        baseline_metrics=baseline_constant(float(y_tgt.mean()), y_tgt),
    )
