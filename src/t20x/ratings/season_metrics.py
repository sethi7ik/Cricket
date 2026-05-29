"""Season-level cricket metrics — the macro evidence for evolution analyses.

For each (league, season) cell we expose:
    matches, deliveries, runs_per_over, dot_pct, four_pct, six_pct,
    boundary_pct, wicket_pct (wickets per ball),
    powerplay_run_rate, middle_run_rate, death_run_rate

Together these answer macro questions like "did the IPL get more 6-heavy after
2015?" and "how did death-overs scoring evolve from Malinga's prime to Bumrah's?"
"""

from __future__ import annotations

import duckdb
import pandas as pd


def season_metrics(
    conn: duckdb.DuckDBPyConnection,
    league: str | None = None,
) -> pd.DataFrame:
    """Per-season aggregates over deliveries.

    Returns DataFrame keyed by `season` (calendar year) with one row per
    season. If `league` is given, filter to that league; otherwise aggregate
    across all leagues.
    """
    where = ""
    params: list[str] = []
    if league:
        where = "WHERE m.league = ?"
        params.append(league)

    query = f"""
        WITH d AS (
            SELECT
                EXTRACT(year FROM m.date)::INTEGER AS season,
                d.match_id,
                d.phase,
                d.batter_runs,
                d.total_runs,
                d.is_wide,
                d.is_noball,
                d.is_dot,
                d.is_boundary_four,
                d.is_boundary_six,
                d.is_wicket,
                CASE WHEN d.is_wide OR d.is_noball THEN 0 ELSE 1 END AS legal_ball
            FROM deliveries d
            JOIN matches m ON d.match_id = m.match_id
            {where}
        )
        SELECT
            season,
            COUNT(DISTINCT match_id)                  AS matches,
            COUNT(*)                                  AS deliveries,
            SUM(legal_ball)                           AS legal_balls,
            SUM(total_runs)                           AS total_runs,
            SUM(batter_runs)                          AS batter_runs,
            SUM(CASE WHEN legal_ball=1 AND is_dot THEN 1 ELSE 0 END)::DOUBLE
                / NULLIF(SUM(legal_ball), 0)           AS dot_pct,
            SUM(CASE WHEN is_boundary_four THEN 1 ELSE 0 END)::DOUBLE
                / NULLIF(SUM(legal_ball), 0)           AS four_pct,
            SUM(CASE WHEN is_boundary_six THEN 1 ELSE 0 END)::DOUBLE
                / NULLIF(SUM(legal_ball), 0)           AS six_pct,
            SUM(CASE WHEN is_boundary_four OR is_boundary_six THEN 1 ELSE 0 END)::DOUBLE
                / NULLIF(SUM(legal_ball), 0)           AS boundary_pct,
            SUM(CASE WHEN is_wicket THEN 1 ELSE 0 END)::DOUBLE
                / NULLIF(SUM(legal_ball), 0)           AS wicket_pct,
            6.0 * SUM(total_runs)::DOUBLE
                / NULLIF(SUM(legal_ball), 0)           AS runs_per_over
        FROM d
        GROUP BY season
        ORDER BY season
    """
    return conn.execute(query, params).fetchdf()


def season_phase_metrics(
    conn: duckdb.DuckDBPyConnection,
    league: str | None = None,
) -> pd.DataFrame:
    """Per-(season, phase) run rate, dot%, six% — for tracking phase evolution."""
    where = ""
    params: list[str] = []
    if league:
        where = "WHERE m.league = ?"
        params.append(league)

    query = f"""
        WITH d AS (
            SELECT
                EXTRACT(year FROM m.date)::INTEGER AS season,
                d.phase,
                d.batter_runs,
                d.total_runs,
                d.is_wide,
                d.is_noball,
                d.is_dot,
                d.is_boundary_six,
                d.is_wicket,
                CASE WHEN d.is_wide OR d.is_noball THEN 0 ELSE 1 END AS legal_ball
            FROM deliveries d
            JOIN matches m ON d.match_id = m.match_id
            {where}
        )
        SELECT
            season, phase,
            COUNT(*) AS deliveries,
            6.0 * SUM(total_runs)::DOUBLE / NULLIF(SUM(legal_ball), 0) AS runs_per_over,
            SUM(CASE WHEN legal_ball=1 AND is_dot THEN 1 ELSE 0 END)::DOUBLE
                / NULLIF(SUM(legal_ball), 0) AS dot_pct,
            SUM(CASE WHEN is_boundary_six THEN 1 ELSE 0 END)::DOUBLE
                / NULLIF(SUM(legal_ball), 0) AS six_pct,
            SUM(CASE WHEN is_wicket THEN 1 ELSE 0 END)::DOUBLE
                / NULLIF(SUM(legal_ball), 0) AS wicket_pct
        FROM d
        GROUP BY season, phase
        ORDER BY season, phase
    """
    return conn.execute(query, params).fetchdf()


def season_bowler_type_metrics(
    conn: duckdb.DuckDBPyConnection,
    league: str | None = None,
) -> pd.DataFrame:
    """Per-(season, bowler-style) economy + dot/six rates.

    Requires the players table to have `bowling_pace` and `bowling_style`
    populated (run `t20x enrich bowler-styles` first).
    """
    where = "WHERE p.bowling_pace IS NOT NULL"
    params: list[str] = []
    if league:
        where += " AND m.league = ?"
        params.append(league)

    query = f"""
        WITH d AS (
            SELECT
                EXTRACT(year FROM m.date)::INTEGER AS season,
                d.phase,
                d.batter_runs,
                d.total_runs,
                d.is_wide,
                d.is_noball,
                d.is_dot,
                d.is_boundary_six,
                d.is_wicket,
                p.bowling_pace,
                p.bowling_style,
                CASE WHEN p.bowling_pace = 'spin' THEN p.bowling_style ELSE p.bowling_pace END AS bowler_bucket,
                CASE WHEN d.is_wide OR d.is_noball THEN 0 ELSE 1 END AS legal_ball
            FROM deliveries d
            JOIN matches m ON d.match_id = m.match_id
            JOIN players p ON d.bowler_id = p.player_id
            {where}
        )
        SELECT
            season, bowler_bucket,
            COUNT(*) AS deliveries,
            6.0 * SUM(total_runs)::DOUBLE / NULLIF(SUM(legal_ball), 0) AS runs_per_over,
            SUM(CASE WHEN legal_ball=1 AND is_dot THEN 1 ELSE 0 END)::DOUBLE
                / NULLIF(SUM(legal_ball), 0) AS dot_pct,
            SUM(CASE WHEN is_wicket THEN 1 ELSE 0 END)::DOUBLE
                / NULLIF(SUM(legal_ball), 0) AS wicket_pct
        FROM d
        GROUP BY season, bowler_bucket
        ORDER BY season, bowler_bucket
    """
    return conn.execute(query, params).fetchdf()
