"""Enrich the players table with bowler-style metadata.

Cricsheet match JSONs ship name → UUID only; no bowling style. We load a
hand-curated CSV (`data/bowler_styles.csv`) and update the `players` table's
`bowling_arm`, `bowling_pace`, `bowling_style` columns by matching on display
name. When multiple `player_id`s share a name, we pick the one with the most
deliveries bowled (correct for the famous one).

Use `enrich_from_csv(conn, csv_path)` to apply.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd


def _disambiguate_player_id(
    conn: duckdb.DuckDBPyConnection, name: str
) -> str | None:
    """For a display name, return the player_id with the most balls bowled."""
    rows = conn.execute(
        """
        SELECT p.player_id, COALESCE(b.balls, 0) AS balls
        FROM players p
        LEFT JOIN (
            SELECT bowler_id, COUNT(*) AS balls
            FROM deliveries
            GROUP BY bowler_id
        ) b ON p.player_id = b.bowler_id
        WHERE p.name = ?
        ORDER BY balls DESC
        """,
        [name],
    ).fetchall()
    return rows[0][0] if rows else None


def enrich_from_csv(
    conn: duckdb.DuckDBPyConnection, csv_path: str | Path
) -> dict[str, int]:
    """Apply bowler-style metadata from CSV to the players table.

    Returns counts: matched, missing, total_rows.
    """
    df = pd.read_csv(csv_path)
    required = {"cricsheet_name", "arm", "pace", "style"}
    if missing := required - set(df.columns):
        raise ValueError(f"CSV missing columns: {missing}")

    matched = 0
    missing_names: list[str] = []
    for row in df.itertuples():
        pid = _disambiguate_player_id(conn, row.cricsheet_name)
        if pid is None:
            missing_names.append(row.cricsheet_name)
            continue
        conn.execute(
            """
            UPDATE players
            SET bowling_arm = ?, bowling_pace = ?, bowling_style = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE player_id = ?
            """,
            [row.arm, row.pace, row.style, pid],
        )
        matched += 1

    if missing_names:
        print(f"  Could not find {len(missing_names)} names in players table:")
        for nm in missing_names[:20]:
            print(f"    - {nm}")
        if len(missing_names) > 20:
            print(f"    ... and {len(missing_names) - 20} more")

    return {
        "matched": matched,
        "missing": len(missing_names),
        "total_rows": len(df),
    }


def coverage_report(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """How many bowled-ball player_ids are now classified vs unclassified."""
    return conn.execute(
        """
        WITH bowled AS (
            SELECT bowler_id AS player_id, COUNT(*) AS balls
            FROM deliveries GROUP BY bowler_id
        )
        SELECT
            CASE WHEN p.bowling_pace IS NULL THEN 'unclassified' ELSE 'classified' END AS state,
            CASE WHEN p.bowling_pace = 'spin' THEN p.bowling_style
                 WHEN p.bowling_pace IS NULL THEN '(none)'
                 ELSE p.bowling_pace
            END AS bucket,
            COUNT(DISTINCT p.player_id) AS bowlers,
            SUM(bowled.balls) AS deliveries
        FROM bowled
        JOIN players p ON bowled.player_id = p.player_id
        GROUP BY 1, 2
        ORDER BY 1, 4 DESC
        """
    ).fetchdf()
