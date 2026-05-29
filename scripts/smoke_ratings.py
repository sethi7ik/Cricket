"""Smoke-test the rating orchestrator on IPL-only data.

Goal: verify the pipeline runs end-to-end and the top-10 lists pass face validity.
"""

from __future__ import annotations

import time
from pathlib import Path

import duckdb

from t20x.ratings.convergence import RatingOrchestrator


DB_PATH = Path.home() / ".t20x" / "data" / "t20x.duckdb"
LEAGUE = "Indian Premier League"
TOP_N = 15
MIN_DELIVERIES = 500


def player_lookup(conn: duckdb.DuckDBPyConnection) -> dict[str, str]:
    rows = conn.execute("SELECT player_id, name FROM players").fetchall()
    return {pid: name for pid, name in rows}


def delivery_counts(conn: duckdb.DuckDBPyConnection, league: str) -> tuple[dict[str, int], dict[str, int]]:
    bat = dict(conn.execute(
        """
        SELECT d.batter_id, COUNT(*) FROM deliveries d
        JOIN matches m ON d.match_id = m.match_id
        WHERE m.league = ?
        GROUP BY d.batter_id
        """,
        [league],
    ).fetchall())
    bowl = dict(conn.execute(
        """
        SELECT d.bowler_id, COUNT(*) FROM deliveries d
        JOIN matches m ON d.match_id = m.match_id
        WHERE m.league = ?
        GROUP BY d.bowler_id
        """,
        [league],
    ).fetchall())
    return bat, bowl


def print_top(label: str, ratings: dict[str, float], counts: dict[str, int], names: dict[str, str], n: int, min_deliveries: int) -> None:
    eligible = [(pid, r) for pid, r in ratings.items() if counts.get(pid, 0) >= min_deliveries]
    eligible.sort(key=lambda x: x[1], reverse=True)
    print(f"\n=== Top {n} {label} (min {min_deliveries} deliveries) ===")
    for i, (pid, r) in enumerate(eligible[:n], 1):
        name = names.get(pid, pid)
        d = counts.get(pid, 0)
        print(f"  {i:2d}. {name:30s}  rating={r:7.1f}  deliveries={d:,}")


def main() -> None:
    conn = duckdb.connect(str(DB_PATH), read_only=True)
    names = player_lookup(conn)
    bat_counts, bowl_counts = delivery_counts(conn, LEAGUE)

    t0 = time.perf_counter()
    # v1 xR test: Elo-on-residuals only, BT disabled, to isolate the xR effect.
    # Context-only xR has no rating feedback, so one epoch is canonical.
    # Multi-epoch convergence will be re-enabled when xR consumes ratings as features.
    orch = RatingOrchestrator(
        max_epochs=1,
        epsilon=1.0,
        elo_k=0.5,
        bt_blend_weight=0.0,
        use_xr=True,
    )
    history = orch.run(conn, league=LEAGUE, verbose=True)
    elapsed = time.perf_counter() - t0
    print(f"\nTotal time: {elapsed:.1f}s   epochs run: {len(history)}")

    if not history:
        print("No history — pipeline produced nothing.")
        return

    final = history[-1]
    print_top("BATTERS", final.bat_ratings, bat_counts, names, TOP_N, MIN_DELIVERIES)
    print_top("BOWLERS", final.bowl_ratings, bowl_counts, names, TOP_N, MIN_DELIVERIES)


if __name__ == "__main__":
    main()
