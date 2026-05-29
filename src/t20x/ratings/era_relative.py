"""Era-relative WAR — per-season WAR with per-season replacement baselines.

A 2008 IPL season should be compared against 2008's replacement bar, not the
all-time bar. This makes "Gayle's 2011 WAR" directly comparable to
"Suryavanshi's 2025 WAR" because each is denominated in wins-above-the-bar
*for that season*.

Two related but distinct concepts:

1. **Per-season WAR** — for each (player, season), compute WAR using only that
   season's deliveries and a season-specific replacement-level baseline.
   This is the canonical "career arc" view.

2. **Era-comparable WAR** — sum a player's per-season WAR across all seasons
   they played. Because each season is calibrated to its own era, the sum is
   already era-fair (it's not the same as summing raw Δ_WP).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ALL_PHASES = ("powerplay", "middle", "death")


def add_year_column(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Add a `year` column (calendar year extracted from match date)."""
    out = df.copy()
    out["year"] = pd.to_datetime(out[date_col]).dt.year.astype(int)
    return out


def per_season_player_wpa(df: pd.DataFrame, season_col: str = "year") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-(player, season) WPA totals for batters and bowlers.

    df must have: `delta_wp`, `batter_id`, `bowler_id`, and the `season_col`.
    """
    bat = (
        df.groupby([season_col, "batter_id"])
        .agg(wpa=("delta_wp", "sum"), balls=("delta_wp", "size"))
        .reset_index()
        .rename(columns={"batter_id": "player_id"})
    )
    bat["wpa_per_ball"] = bat["wpa"] / bat["balls"]

    bowl = (
        df.groupby([season_col, "bowler_id"])
        .agg(wpa=("delta_wp", lambda s: -s.sum()), balls=("delta_wp", "size"))
        .reset_index()
        .rename(columns={"bowler_id": "player_id"})
    )
    bowl["wpa_per_ball"] = bowl["wpa"] / bowl["balls"]
    return bat, bowl


def compute_per_season_war(
    bat: pd.DataFrame,
    bowl: pd.DataFrame,
    season_col: str = "year",
    replacement_pct: float = 25.0,
    min_balls_season: int = 120,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add per-season WAR using per-season replacement-level baselines.

    Replacement-level for a season = the `replacement_pct`-th percentile of
    `wpa_per_ball` among players meeting `min_balls_season` that season.
    """
    def _war(df: pd.DataFrame, sign_for_baseline: int = +1) -> pd.DataFrame:
        out = df.copy()
        out["war"] = np.nan
        repl_by_season: dict[int, float] = {}
        for s, sub in df.groupby(season_col):
            eligible = sub.loc[sub["balls"] >= min_balls_season, "wpa_per_ball"]
            repl = float(np.percentile(eligible, replacement_pct)) if len(eligible) else 0.0
            repl_by_season[int(s)] = repl
            mask = out[season_col] == s
            out.loc[mask, "war"] = out.loc[mask, "wpa"] - repl * out.loc[mask, "balls"]
        out.attrs["replacement_by_season"] = repl_by_season
        return out

    return _war(bat), _war(bowl)


def career_arc(
    per_season: pd.DataFrame,
    player_id: str,
    season_col: str = "year",
) -> pd.DataFrame:
    """Return a single player's per-season trajectory, sorted by season."""
    return per_season[per_season["player_id"] == player_id].sort_values(season_col).reset_index(drop=True)


def career_total_era_adjusted(
    per_season: pd.DataFrame,
    player_id: str,
    season_col: str = "year",
) -> dict[str, float]:
    """Sum a player's per-season WAR across their career (already era-fair)."""
    arc = career_arc(per_season, player_id, season_col)
    if arc.empty:
        return {"war_total": 0.0, "wpa_total": 0.0, "balls_total": 0, "seasons": 0}
    return {
        "war_total": float(arc["war"].sum()),
        "wpa_total": float(arc["wpa"].sum()),
        "balls_total": int(arc["balls"].sum()),
        "seasons": int(arc[season_col].nunique()),
        "first_season": int(arc[season_col].min()),
        "last_season": int(arc[season_col].max()),
        "peak_season": int(arc.loc[arc["war"].idxmax(), season_col]),
        "peak_war": float(arc["war"].max()),
    }
