"""Win Probability (WP) model and Win Probability Added (WPA) attribution.

The cricket analog of baseball WPA. For every delivery we compute:

    Δ_WP = WP(post_state) − WP(pre_state)

This delta is the change in the batting team's win probability caused by that
single ball. We then attribute:

    +Δ_WP  to the batter who faced the ball
    −Δ_WP  to the bowler who bowled it

Summing across a career yields a player's total Win Probability Added — a
clutch-aware, counterfactual measure of value that natively rewards:
  - scoring in tight chases more than scoring in routine ones,
  - wicket avoidance (each ball survived in pressure → small positive Δ_WP),
  - wickets that swing matches (dismissing a top batter when chasing low).

Features used by the underlying classifier are situation-only:
    innings, balls_remaining, wickets_in_hand, runs_so_far,
    target_remaining (innings 2 only), required_rate (innings 2 only)

The WP model intentionally does NOT condition on player identity — it produces
a baseline against which players are then measured.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import duckdb
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

WP_FEATURES: tuple[str, ...] = (
    "innings",
    "balls_remaining",
    "wickets_in_hand",
    "runs_so_far",
    "target_remaining",
    "required_rate",
)

T20_INNINGS_BALLS = 120
WICKETS_PER_INNINGS = 10


@dataclass
class WinProbabilityModel:
    """Predicts P(batting team wins) from match state."""

    max_iter: int = 300
    learning_rate: float = 0.05
    max_depth: int | None = 6
    min_samples_leaf: int = 500
    random_state: int = 42

    classifier_: HistGradientBoostingClassifier | None = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "WinProbabilityModel":
        clf = HistGradientBoostingClassifier(
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
        )
        clf.fit(X[list(WP_FEATURES)].astype("float64"), y.astype(int))
        self.classifier_ = clf
        return self

    def predict_wp(self, X: pd.DataFrame) -> np.ndarray:
        if self.classifier_ is None:
            raise RuntimeError("WP model must be fit first.")
        proba = self.classifier_.predict_proba(X[list(WP_FEATURES)].astype("float64"))
        # Class 1 = batting team wins
        class_idx = int(np.where(self.classifier_.classes_ == 1)[0][0])
        return proba[:, class_idx]


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------


def load_wpa_frame(
    conn: duckdb.DuckDBPyConnection,
    league: str | None = None,
) -> pd.DataFrame:
    """Pull deliveries + match metadata and derive per-delivery WP features.

    Skips matches with no result (winner is empty).

    Returns a DataFrame with one row per delivery and these key columns:
        match_id, innings, batter_id, bowler_id,
        pre_balls_remaining, pre_wickets_in_hand, pre_runs_so_far,
        pre_target_remaining, pre_required_rate,
        post_balls_remaining, post_wickets_in_hand, post_runs_so_far,
        post_target_remaining, post_required_rate,
        batting_team_won (1/0)
    """
    where = "WHERE m.winner IS NOT NULL AND m.winner != ''"
    if league:
        where += f" AND m.league = '{league}'"

    query = f"""
        WITH innings_totals AS (
            SELECT match_id, innings, MAX(innings_runs) AS final_runs
            FROM deliveries
            GROUP BY match_id, innings
        ),
        targets AS (
            SELECT match_id,
                   MAX(CASE WHEN innings = 1 THEN final_runs END) AS innings1_total
            FROM innings_totals
            GROUP BY match_id
        )
        SELECT
            d.match_id,
            d.innings,
            d.delivery_seq,
            d.over_number,
            d.ball_number,
            d.phase,
            d.batter_id,
            d.bowler_id,
            d.is_wicket,
            d.batter_runs,
            d.total_runs,
            d.innings_runs        AS post_runs,
            d.innings_wickets     AS post_wickets,
            d.balls_remaining     AS post_balls_remaining,
            d.required_rate       AS post_required_rate,
            t.innings1_total,
            m.toss_winner,
            m.toss_decision,
            m.team_1,
            m.team_2,
            m.winner
        FROM deliveries d
        JOIN matches m ON d.match_id = m.match_id
        LEFT JOIN targets t ON d.match_id = t.match_id
        {where}
        ORDER BY d.match_id, d.innings, d.delivery_seq
    """
    df = conn.execute(query).fetchdf()
    if df.empty:
        return df

    # Identify batting team per innings via toss
    is_innings_1 = df["innings"] == 1
    toss_bat = df["toss_decision"] == "bat"
    bat_first_team = np.where(toss_bat, df["toss_winner"], np.where(df["team_1"] == df["toss_winner"], df["team_2"], df["team_1"]))
    bat_second_team = np.where(df["team_1"] == bat_first_team, df["team_2"], df["team_1"])
    df["batting_team"] = np.where(is_innings_1, bat_first_team, bat_second_team)
    df["batting_team_won"] = (df["batting_team"] == df["winner"]).astype(int)

    # Post-state derived
    df["post_wickets_in_hand"] = WICKETS_PER_INNINGS - df["post_wickets"]
    df["post_target_remaining"] = np.where(
        df["innings"] == 2,
        (df["innings1_total"] + 1) - df["post_runs"],
        np.nan,
    )
    # Use stored required_rate (computed at ingest) but recompute when missing
    rr = df["post_required_rate"].astype("float64")
    fallback = np.where(
        (df["innings"] == 2) & (df["post_balls_remaining"] > 0),
        df["post_target_remaining"] * 6.0 / df["post_balls_remaining"].clip(lower=1),
        np.nan,
    )
    df["post_required_rate"] = np.where(rr.notna() & (rr > 0), rr, fallback)

    # Pre-state via shift within (match_id, innings) groups
    grp = df.groupby(["match_id", "innings"], sort=False)
    df["pre_runs_so_far"] = grp["post_runs"].shift(1).fillna(0)
    df["pre_wickets_in_hand"] = grp["post_wickets_in_hand"].shift(1).fillna(WICKETS_PER_INNINGS)
    df["pre_balls_remaining"] = grp["post_balls_remaining"].shift(1).fillna(T20_INNINGS_BALLS)
    df["pre_target_remaining"] = np.where(
        df["innings"] == 2,
        (df["innings1_total"] + 1) - df["pre_runs_so_far"],
        np.nan,
    )
    df["pre_required_rate"] = np.where(
        (df["innings"] == 2) & (df["pre_balls_remaining"] > 0),
        df["pre_target_remaining"] * 6.0 / df["pre_balls_remaining"].clip(lower=1),
        np.nan,
    )

    return df


def _state_frame(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Build a feature matrix for either pre- or post-state."""
    return pd.DataFrame(
        {
            "innings": df["innings"].astype("float64"),
            "balls_remaining": df[f"{prefix}_balls_remaining"].astype("float64"),
            "wickets_in_hand": df[f"{prefix}_wickets_in_hand"].astype("float64"),
            "runs_so_far": df[f"{prefix}_runs_so_far"].astype("float64") if prefix == "pre" else df["post_runs"].astype("float64"),
            "target_remaining": df[f"{prefix}_target_remaining"].astype("float64"),
            "required_rate": df[f"{prefix}_required_rate"].astype("float64"),
        }
    )


def compute_wpa(
    conn: duckdb.DuckDBPyConnection,
    league: str | None = None,
) -> pd.DataFrame:
    """End-to-end: load data, train WP model, score all deliveries, return Δ_WP per ball.

    Returns the same DataFrame as `load_wpa_frame` plus columns:
        wp_pre, wp_post, delta_wp (= wp_post − wp_pre)
    """
    df = load_wpa_frame(conn, league=league)
    if df.empty:
        return df

    X_post = _state_frame(df, "post")
    y = df["batting_team_won"].values
    model = WinProbabilityModel().fit(X_post, y)

    X_pre = _state_frame(df, "pre")
    df["wp_post"] = model.predict_wp(X_post)
    df["wp_pre"] = model.predict_wp(X_pre)
    df["delta_wp"] = df["wp_post"] - df["wp_pre"]
    return df


def aggregate_player_wpa(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate per-delivery Δ_WP into per-player totals.

    Returns:
        (batter_wpa_df, bowler_wpa_df) — each indexed by player_id with columns:
            wpa, balls, wpa_per_ball
    """
    bat = (
        df.groupby("batter_id")
        .agg(wpa=("delta_wp", "sum"), balls=("delta_wp", "size"))
        .reset_index()
        .rename(columns={"batter_id": "player_id"})
    )
    bat["wpa_per_ball"] = bat["wpa"] / bat["balls"]

    bowl = (
        df.groupby("bowler_id")
        .agg(wpa=("delta_wp", lambda s: -s.sum()), balls=("delta_wp", "size"))
        .reset_index()
        .rename(columns={"bowler_id": "player_id"})
    )
    bowl["wpa_per_ball"] = bowl["wpa"] / bowl["balls"]
    return bat, bowl


def compute_war(
    df: pd.DataFrame,
    replacement_pct: float = 25.0,
    min_balls: int = 500,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert WPA to WAR using a replacement-level baseline.

    Replacement level = the `replacement_pct`-th percentile of per-ball WPA
    among players meeting `min_balls`. WAR = WPA − replacement_rate × balls.
    """
    bat, bowl = aggregate_player_wpa(df)

    bat_repl = np.percentile(
        bat.loc[bat["balls"] >= min_balls, "wpa_per_ball"], replacement_pct
    )
    bowl_repl = np.percentile(
        bowl.loc[bowl["balls"] >= min_balls, "wpa_per_ball"], replacement_pct
    )

    bat["war"] = bat["wpa"] - bat_repl * bat["balls"]
    bowl["war"] = bowl["wpa"] - bowl_repl * bowl["balls"]
    bat.attrs["replacement_per_ball"] = float(bat_repl)
    bowl.attrs["replacement_per_ball"] = float(bowl_repl)
    return bat, bowl


def aggregate_player_wpa_by_phase(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate Δ_WP per (player, phase)."""
    bat = (
        df.groupby(["batter_id", "phase"])
        .agg(wpa=("delta_wp", "sum"), balls=("delta_wp", "size"))
        .reset_index()
        .rename(columns={"batter_id": "player_id"})
    )
    bat["wpa_per_ball"] = bat["wpa"] / bat["balls"]

    bowl = (
        df.groupby(["bowler_id", "phase"])
        .agg(wpa=("delta_wp", lambda s: -s.sum()), balls=("delta_wp", "size"))
        .reset_index()
        .rename(columns={"bowler_id": "player_id"})
    )
    bowl["wpa_per_ball"] = bowl["wpa"] / bowl["balls"]
    return bat, bowl


def compute_war_by_phase(
    df: pd.DataFrame,
    replacement_pct: float = 25.0,
    min_balls_phase: int = 150,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-phase WAR with **phase-specific** replacement-level baselines.

    Replacement-level is computed independently within each phase because Δ_WP
    magnitudes differ by phase (death-overs balls swing WP much more than
    powerplay balls).
    """
    bat, bowl = aggregate_player_wpa_by_phase(df)

    bat_repl_by_phase: dict[str, float] = {}
    for phase, sub in bat.groupby("phase"):
        eligible = sub[sub["balls"] >= min_balls_phase]["wpa_per_ball"]
        bat_repl_by_phase[str(phase)] = float(np.percentile(eligible, replacement_pct)) if len(eligible) else 0.0

    bowl_repl_by_phase: dict[str, float] = {}
    for phase, sub in bowl.groupby("phase"):
        eligible = sub[sub["balls"] >= min_balls_phase]["wpa_per_ball"]
        bowl_repl_by_phase[str(phase)] = float(np.percentile(eligible, replacement_pct)) if len(eligible) else 0.0

    bat["war"] = bat.apply(
        lambda r: r["wpa"] - bat_repl_by_phase.get(str(r["phase"]), 0.0) * r["balls"], axis=1
    )
    bowl["war"] = bowl.apply(
        lambda r: r["wpa"] - bowl_repl_by_phase.get(str(r["phase"]), 0.0) * r["balls"], axis=1
    )
    bat.attrs["replacement_per_ball_by_phase"] = bat_repl_by_phase
    bowl.attrs["replacement_per_ball_by_phase"] = bowl_repl_by_phase
    return bat, bowl
