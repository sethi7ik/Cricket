"""Check where well-known IPL batters land in the rating distribution."""

from __future__ import annotations

from pathlib import Path

import duckdb

from t20x.ratings.convergence import RatingOrchestrator

DB_PATH = Path.home() / ".t20x" / "data" / "t20x.duckdb"
LEAGUE = "Indian Premier League"

KNOWN_BATTERS = [
    "V Kohli", "RG Sharma", "JC Buttler", "DA Warner", "CH Gayle",
    "MS Dhoni", "KL Rahul", "SK Raina", "SR Tendulkar", "SE Marsh",
    "AB de Villiers", "H Klaasen", "PD Salt", "SA Yadav",
]


def main() -> None:
    conn = duckdb.connect(str(DB_PATH), read_only=True)
    names = dict(conn.execute("SELECT player_id, name FROM players").fetchall())
    name_to_id = {n: pid for pid, n in names.items()}

    bat_counts = dict(conn.execute(
        """
        SELECT d.batter_id, COUNT(*) FROM deliveries d
        JOIN matches m ON d.match_id = m.match_id
        WHERE m.league = ?
        GROUP BY d.batter_id
        """, [LEAGUE]).fetchall())

    orch = RatingOrchestrator(
        max_epochs=1, epsilon=1.0, elo_k=0.5, bt_blend_weight=0.0, use_xr=True,
    )
    orch.run(conn, league=LEAGUE, verbose=False)
    final = orch.history[-1]

    # Rank all batters with >= 500 deliveries
    eligible = [(pid, r) for pid, r in final.bat_ratings.items() if bat_counts.get(pid, 0) >= 500]
    eligible.sort(key=lambda x: x[1], reverse=True)
    rank_map = {pid: i + 1 for i, (pid, _) in enumerate(eligible)}
    print(f"Total eligible batters (>=500 deliveries): {len(eligible)}\n")

    print(f"{'Name':30s}  {'rank':>5s}  {'rating':>8s}  {'deliveries':>10s}")
    for name in KNOWN_BATTERS:
        pid = name_to_id.get(name)
        if not pid:
            print(f"{name:30s}  {'N/A':>5s}  (not in players table)")
            continue
        r = final.bat_ratings.get(pid)
        d = bat_counts.get(pid, 0)
        rank = rank_map.get(pid, "—")
        if r is None:
            print(f"{name:30s}  {'—':>5s}  (not rated; d={d})")
        else:
            print(f"{name:30s}  {rank!s:>5s}  {r:8.1f}  {d:>10,}")


if __name__ == "__main__":
    main()
