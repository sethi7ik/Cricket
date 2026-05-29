"""Smoke-test era-relative WAR on IPL — Gayle and Suryavanshi as anchor cases."""

from __future__ import annotations

import time
from pathlib import Path

import duckdb
import pandas as pd

from t20x.ratings.era_relative import (
    add_year_column,
    career_arc,
    career_total_era_adjusted,
    compute_per_season_war,
    per_season_player_wpa,
)
from t20x.ratings.win_probability import compute_wpa

DB_PATH = Path.home() / ".t20x" / "data" / "t20x.duckdb"
LEAGUE = "Indian Premier League"


def main() -> None:
    conn = duckdb.connect(str(DB_PATH), read_only=True)

    t0 = time.perf_counter()
    df = compute_wpa(conn, league=LEAGUE)
    # Attach match date for season extraction
    dates = conn.execute(
        "SELECT match_id, date FROM matches WHERE winner IS NOT NULL AND winner != ''"
    ).fetchdf()
    df = df.merge(dates, on="match_id", how="left")
    df = add_year_column(df)
    print(f"Computed WPA + year for {len(df):,} deliveries  ({time.perf_counter()-t0:.1f}s)")

    bat, bowl = per_season_player_wpa(df)
    bat, bowl = compute_per_season_war(bat, bowl, min_balls_season=120)
    print(f"Per-season-WPA tables: {len(bat):,} batter-seasons, {len(bowl):,} bowler-seasons")
    print(f"\nReplacement WPA/ball by season (batters):")
    for s, r in sorted(bat.attrs["replacement_by_season"].items()):
        print(f"  {s}  bat_repl/ball={r:+.5f}")

    # Look up player IDs
    name_to_id = dict(conn.execute("SELECT name, player_id FROM players").fetchall())
    # name collisions: pick the most-balls one within IPL
    activity = pd.concat([df["batter_id"], df["bowler_id"]]).value_counts()
    best_by_name: dict[str, str] = {}
    for pid, nm in conn.execute("SELECT player_id, name FROM players").fetchall():
        balls = int(activity.get(pid, 0))
        if best_by_name.get(nm) is None or balls > activity.get(best_by_name[nm], 0):
            best_by_name[nm] = pid

    def show_arc(name: str) -> None:
        pid = best_by_name.get(name)
        if pid is None:
            print(f"\n[{name}] not in players table")
            return
        arc = career_arc(bat, pid)
        if arc.empty:
            print(f"\n[{name}] no batter-season rows")
            return
        print(f"\n=== {name} — career arc (IPL batter, per-season WAR) ===")
        print(f"{'season':>7s}  {'WAR':>7s}  {'WPA':>7s}  {'WPA/ball':>10s}  {'balls':>6s}")
        for _, r in arc.iterrows():
            print(f"  {int(r['year']):>5d}  {r['war']:>+7.2f}  {r['wpa']:>+7.2f}  {r['wpa_per_ball']:>+10.5f}  {int(r['balls']):>6,}")
        tot = career_total_era_adjusted(bat, pid)
        print(f"  TOTAL  era-adj WAR {tot['war_total']:+.2f}   peak {tot['peak_season']} ({tot['peak_war']:+.2f})")

    for name in ("CH Gayle", "V Kohli", "AB de Villiers", "BB McCullum", "Vaibhav Suryavanshi", "V Suryavanshi"):
        show_arc(name)


if __name__ == "__main__":
    main()
