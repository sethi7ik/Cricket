"""Load parsed match data into DuckDB using fast pandas DataFrame bulk inserts."""

from __future__ import annotations

from typing import Iterator

import duckdb
import pandas as pd

from t20x.models.domain import ParsedMatch


def load_matches(
    conn: duckdb.DuckDBPyConnection,
    matches: Iterator[ParsedMatch],
    batch_size: int = 500,
) -> dict[str, int]:
    """Load parsed matches into DuckDB tables using DataFrame bulk inserts.

    Args:
        conn: DuckDB connection with schema initialized.
        matches: Iterator of ParsedMatch objects.
        batch_size: Number of matches to batch before inserting.

    Returns:
        Dict with counts: {"matches": N, "deliveries": N, "players": N}
    """
    match_rows: list[dict] = []
    delivery_rows: list[dict] = []
    player_rows: list[dict] = []
    seen_players: set[str] = set()

    # Load existing player IDs to avoid duplicates
    try:
        existing = conn.execute("SELECT player_id FROM players").fetchall()
        seen_players = {row[0] for row in existing}
    except Exception:
        pass

    # Load existing match IDs to skip duplicates
    existing_matches: set[str] = set()
    try:
        rows = conn.execute("SELECT match_id FROM matches").fetchall()
        existing_matches = {row[0] for row in rows}
    except Exception:
        pass

    counts = {"matches": 0, "deliveries": 0, "players": 0}

    def flush() -> None:
        if match_rows:
            df = pd.DataFrame(match_rows)
            conn.execute("INSERT OR IGNORE INTO matches SELECT * FROM df")
            match_rows.clear()

        if player_rows:
            df = pd.DataFrame(player_rows)
            conn.execute("INSERT OR IGNORE INTO players (player_id, name, full_name, country) SELECT * FROM df")
            player_rows.clear()

        if delivery_rows:
            df = pd.DataFrame(delivery_rows)
            conn.execute("INSERT OR IGNORE INTO deliveries SELECT * FROM df")
            delivery_rows.clear()

    for match in matches:
        info = match.info

        # Skip already-loaded matches
        if info.match_id in existing_matches:
            continue

        match_rows.append({
            "match_id": info.match_id,
            "league": info.league,
            "season": info.season,
            "date": str(info.date),
            "venue": info.venue,
            "city": info.city,
            "team_1": info.team_1,
            "team_2": info.team_2,
            "toss_winner": info.toss_winner,
            "toss_decision": info.toss_decision,
            "winner": info.winner,
            "win_margin": info.win_margin,
            "win_type": info.win_type,
            "player_of_match": info.player_of_match,
            "gender": info.gender,
        })
        existing_matches.add(info.match_id)

        for pid, player in match.players.items():
            if pid not in seen_players:
                player_rows.append({
                    "player_id": pid,
                    "name": player.name,
                    "full_name": player.full_name,
                    "country": player.country,
                })
                seen_players.add(pid)
                counts["players"] += 1

        for d in match.deliveries:
            delivery_rows.append({
                "match_id": d.match_id,
                "innings": d.innings,
                "over_number": d.over_number,
                "ball_number": d.ball_number,
                "delivery_seq": d.delivery_seq,
                "batter_id": d.batter_id,
                "bowler_id": d.bowler_id,
                "non_striker_id": d.non_striker_id,
                "batter_runs": d.batter_runs,
                "extra_runs": d.extra_runs,
                "total_runs": d.total_runs,
                "is_wide": d.is_wide,
                "is_noball": d.is_noball,
                "is_bye": d.is_bye,
                "is_legbye": d.is_legbye,
                "is_boundary_four": d.is_boundary_four,
                "is_boundary_six": d.is_boundary_six,
                "is_dot": d.is_dot,
                "is_wicket": d.is_wicket,
                "dismissal_kind": d.dismissal_kind,
                "player_out_id": d.player_out_id,
                "phase": d.phase.value,
                "innings_runs": d.innings_runs,
                "innings_wickets": d.innings_wickets,
                "run_rate": d.run_rate,
                "required_rate": d.required_rate,
                "balls_remaining": d.balls_remaining,
            })
            counts["deliveries"] += 1

        counts["matches"] += 1

        if counts["matches"] % batch_size == 0:
            flush()

    # Final flush
    flush()
    return counts
