"""Predict match outcomes from career WPA — the real downstream test.

If WPA correctly measures player value, then summing the playing XI's career
WPAs (computed from data strictly before match M) should predict match M's
outcome better than baselines.

This is the test that promotes WPA from "internally calibrated" to "actually
predictive of real-world matches."
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import duckdb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from t20x.ratings.win_probability import compute_wpa
from t20x.validation.calibration import brier_score, log_loss


@dataclass
class MatchPredictionResult:
    cutoff: dt.date
    n_test_matches: int
    metrics: dict[str, dict[str, float]]
    n_train_fit: int


def _bowling_team(df: pd.DataFrame) -> pd.Series:
    return np.where(df["team_1"] == df["batting_team"], df["team_2"], df["team_1"])


def evaluate_match_prediction(
    conn: duckdb.DuckDBPyConnection,
    cutoff: dt.date,
    fit_split: float = 0.5,
    random_state: int = 42,
    lookback_years: float | None = None,
) -> MatchPredictionResult:
    """Career WPA prior to `cutoff` predicts match outcomes after `cutoff`.

    Methodology:
        1. Compute Δ_WP for every delivery (single WP model fit on full data —
           a small leak for the WP itself, but the test is whether *aggregated
           career WPA* predicts matches, which has no leak).
        2. For each player, sum batting Δ_WP and bowling −Δ_WP over deliveries
           strictly before `cutoff`. These are the "prior" career WPAs.
        3. For each post-cutoff match, build playing XIs from who-batted-for-whom
           and who-bowled-for-whom. Team strength = sum of (batting + bowling)
           career WPAs across the XI.
        4. Split post-cutoff matches into a fitting half and a holdout half.
           Fit a logistic regression of P(team-A wins) on team-A-minus-team-B
           strength differential. Evaluate on the holdout half.
        5. Compare to baselines: 50/50, toss-winner-wins, bat-first-wins.
    """
    df = compute_wpa(conn, league=None)
    # Attach match date
    dates = conn.execute(
        "SELECT match_id, date, toss_winner, toss_decision, winner, team_1, team_2 FROM matches "
        "WHERE winner IS NOT NULL AND winner != ''"
    ).fetchdf()
    dates["date"] = pd.to_datetime(dates["date"])
    df = df.merge(dates[["match_id", "date"]], on="match_id", how="left")
    df["bowling_team"] = _bowling_team(df)

    cutoff_ts = pd.Timestamp(cutoff)
    pre = df[df["date"] < cutoff_ts]
    if lookback_years is not None:
        window_start = cutoff_ts - pd.DateOffset(years=int(round(lookback_years * 100)) // 100)
        # pd.DateOffset doesn't accept float years; round to whole years for simplicity.
        window_start = cutoff_ts - pd.DateOffset(years=int(lookback_years))
        pre = pre[pre["date"] >= window_start]
    post = df[df["date"] >= cutoff_ts]

    if pre.empty or post.empty:
        raise RuntimeError(f"Empty pre or post split at cutoff {cutoff}.")

    # Career batting WPA (positive delta = positive for batter)
    bat_wpa = pre.groupby("batter_id")["delta_wp"].sum().rename("bat_wpa")
    # Career bowling WPA (negative delta = positive for bowler)
    bowl_wpa = pre.groupby("bowler_id")["delta_wp"].apply(lambda s: -s.sum()).rename("bowl_wpa")

    # Per-player career value = bat WPA + bowl WPA (default 0 if missing one role)
    player_value = bat_wpa.to_frame().join(bowl_wpa, how="outer").fillna(0.0)
    player_value["total"] = player_value["bat_wpa"] + player_value["bowl_wpa"]

    # Derive playing XI per (match, team) from post-cutoff deliveries
    bat_aff = (
        post[["match_id", "batter_id", "batting_team"]]
        .drop_duplicates()
        .rename(columns={"batter_id": "player_id", "batting_team": "team"})
    )
    bowl_aff = (
        post[["match_id", "bowler_id", "bowling_team"]]
        .drop_duplicates()
        .rename(columns={"bowler_id": "player_id", "bowling_team": "team"})
    )
    affiliation = pd.concat([bat_aff, bowl_aff], ignore_index=True).drop_duplicates()

    affiliation = affiliation.merge(
        player_value.reset_index().rename(columns={"index": "player_id"})[["player_id", "total"]],
        on="player_id",
        how="left",
    )
    # Players with no pre-cutoff record get value 0 (debutants/replacement-equivalent)
    affiliation["total"] = affiliation["total"].fillna(0.0)

    team_strength = (
        affiliation.groupby(["match_id", "team"])["total"].sum().reset_index().rename(
            columns={"total": "team_value"}
        )
    )

    # Build match-level frame: team_1, team_2 strengths, outcome
    matches = post[["match_id", "date"]].drop_duplicates()
    matches = matches.merge(dates[["match_id", "team_1", "team_2", "winner", "toss_winner", "toss_decision"]], on="match_id")
    matches = matches.merge(
        team_strength.rename(columns={"team": "team_1", "team_value": "team_1_value"}),
        on=["match_id", "team_1"],
        how="left",
    )
    matches = matches.merge(
        team_strength.rename(columns={"team": "team_2", "team_value": "team_2_value"}),
        on=["match_id", "team_2"],
        how="left",
    )
    matches = matches.dropna(subset=["team_1_value", "team_2_value"])
    matches["team_1_won"] = (matches["winner"] == matches["team_1"]).astype(int)
    matches["diff"] = matches["team_1_value"] - matches["team_2_value"]

    rng = np.random.RandomState(random_state)
    order = rng.permutation(len(matches))
    n_fit = int(len(matches) * fit_split)
    fit_idx = order[:n_fit]
    test_idx = order[n_fit:]
    fit = matches.iloc[fit_idx]
    test = matches.iloc[test_idx]

    # Fit logistic on team-strength differential
    clf = LogisticRegression()
    clf.fit(fit[["diff"]].values, fit["team_1_won"].values)
    p_test = clf.predict_proba(test[["diff"]].values)[:, 1]
    y_test = test["team_1_won"].to_numpy()
    pred_test = (p_test >= 0.5).astype(int)

    metrics: dict[str, dict[str, float]] = {}
    metrics["WPA_model"] = {
        "accuracy": float((pred_test == y_test).mean()),
        "log_loss": log_loss(p_test, y_test),
        "brier": brier_score(p_test, y_test),
        "coef": float(clf.coef_[0][0]),
        "intercept": float(clf.intercept_[0]),
    }
    metrics["coin_flip"] = {
        "accuracy": 0.5,
        "log_loss": log_loss(np.full_like(y_test, 0.5, dtype="float64"), y_test),
        "brier": brier_score(np.full_like(y_test, 0.5, dtype="float64"), y_test),
    }
    # Toss-winner-wins baseline
    toss_pred = (test["toss_winner"] == test["team_1"]).astype(int).to_numpy()
    metrics["toss_winner_wins"] = {
        "accuracy": float((toss_pred == y_test).mean()),
        "log_loss": log_loss(toss_pred.astype("float64") * 0.9 + 0.05, y_test),
        "brier": brier_score(toss_pred.astype("float64"), y_test),
    }
    # Bat-first-wins baseline (innings 1 team)
    # toss_decision == 'bat' means toss winner bats first; 'field' means other team bats first
    bat_first = np.where(
        test["toss_decision"] == "bat",
        test["toss_winner"],
        np.where(test["team_1"] == test["toss_winner"], test["team_2"], test["team_1"]),
    )
    bat_first_is_t1 = (bat_first == test["team_1"]).astype(int).to_numpy()
    metrics["bat_first_wins"] = {
        "accuracy": float((bat_first_is_t1 == y_test).mean()),
        "log_loss": log_loss(bat_first_is_t1.astype("float64") * 0.9 + 0.05, y_test),
        "brier": brier_score(bat_first_is_t1.astype("float64"), y_test),
    }

    return MatchPredictionResult(
        cutoff=cutoff,
        n_test_matches=int(len(test)),
        metrics=metrics,
        n_train_fit=int(len(fit)),
    )
