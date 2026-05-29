"""Smoke-test WPA / WAR on IPL data."""

from __future__ import annotations

import time
from pathlib import Path

import duckdb

from t20x.ratings.win_probability import compute_war, compute_wpa


DB_PATH = Path.home() / ".t20x" / "data" / "t20x.duckdb"
LEAGUE = "Indian Premier League"
TOP_N = 15
MIN_BALLS = 500


KNOWN_BATTERS = [
    "V Kohli", "RG Sharma", "JC Buttler", "DA Warner", "CH Gayle",
    "MS Dhoni", "KL Rahul", "SK Raina", "AB de Villiers", "SA Yadav",
    "H Klaasen", "PD Salt", "RR Pant", "SR Watson", "F du Plessis",
]
KNOWN_BOWLERS = [
    "JJ Bumrah", "SP Narine", "Rashid Khan", "YS Chahal", "R Ashwin",
    "SL Malinga", "DW Steyn", "JC Archer", "M Muralitharan", "B Kumar",
    "AR Patel", "Harbhajan Singh",
]


def build_name_to_id(conn: duckdb.DuckDBPyConnection, df) -> dict[str, str]:
    """Resolve display-name collisions (Cricsheet has multiple 'Rashid Khan's etc.)
    by picking the player_id with the most rows in the scored dataset.
    """
    import pandas as pd  # local import keeps top of script lean
    rows = conn.execute("SELECT player_id, name FROM players").fetchall()
    id_to_name = {pid: n for pid, n in rows}

    activity = (
        pd.concat([df["batter_id"].rename("pid"), df["bowler_id"].rename("pid")])
        .value_counts()
        .rename("balls")
    )

    by_name: dict[str, tuple[str, int]] = {}  # name -> (best_pid, best_balls)
    for pid, name in rows:
        balls = int(activity.get(pid, 0))
        cur = by_name.get(name)
        if cur is None or balls > cur[1]:
            by_name[name] = (pid, balls)
    return {n: pid for n, (pid, _) in by_name.items()}


def main() -> None:
    conn = duckdb.connect(str(DB_PATH), read_only=True)
    names = dict(conn.execute("SELECT player_id, name FROM players").fetchall())

    t0 = time.perf_counter()
    df = compute_wpa(conn, league=LEAGUE)
    print(f"Computed Δ_WP for {len(df):,} deliveries  ({time.perf_counter()-t0:.1f}s)")
    name_to_id = build_name_to_id(conn, df)
    print(f"Δ_WP mean = {df['delta_wp'].mean():+.5f}  std = {df['delta_wp'].std():.5f}")
    print(f"WP_pre mean = {df['wp_pre'].mean():.3f}   WP_post mean = {df['wp_post'].mean():.3f}")
    print(f"Frac batting-team-won across deliveries = {df['batting_team_won'].mean():.3f}")

    bat, bowl = compute_war(df, replacement_pct=25.0, min_balls=MIN_BALLS)
    bat = bat.merge(
        bat["player_id"].map(names).rename("name").to_frame().assign(player_id=bat["player_id"]),
        on="player_id",
    )
    bowl = bowl.merge(
        bowl["player_id"].map(names).rename("name").to_frame().assign(player_id=bowl["player_id"]),
        on="player_id",
    )

    print(f"\nReplacement-level: batter={bat.attrs['replacement_per_ball']:+.5f} WPA/ball, "
          f"bowler={bowl.attrs['replacement_per_ball']:+.5f} WPA/ball")

    bat_top = bat[bat["balls"] >= MIN_BALLS].sort_values("war", ascending=False).head(TOP_N)
    bowl_top = bowl[bowl["balls"] >= MIN_BALLS].sort_values("war", ascending=False).head(TOP_N)

    print(f"\n=== Top {TOP_N} BATTERS by WAR (min {MIN_BALLS} balls) ===")
    print(f"{'#':>3} {'Name':30s} {'WAR':>8} {'WPA':>8} {'WPA/ball':>10} {'balls':>8}")
    for i, row in enumerate(bat_top.itertuples(), 1):
        print(f"{i:>3} {row.name:30s} {row.war:>+8.2f} {row.wpa:>+8.2f} {row.wpa_per_ball:>+10.5f} {row.balls:>8,}")

    print(f"\n=== Top {TOP_N} BOWLERS by WAR (min {MIN_BALLS} balls) ===")
    print(f"{'#':>3} {'Name':30s} {'WAR':>8} {'WPA':>8} {'WPA/ball':>10} {'balls':>8}")
    for i, row in enumerate(bowl_top.itertuples(), 1):
        print(f"{i:>3} {row.name:30s} {row.war:>+8.2f} {row.wpa:>+8.2f} {row.wpa_per_ball:>+10.5f} {row.balls:>8,}")

    # Known-player check
    def show_known(label: str, known: list[str], df_p: "pd.DataFrame") -> None:  # type: ignore[name-defined]
        eligible = df_p[df_p["balls"] >= MIN_BALLS].copy()
        eligible = eligible.sort_values("war", ascending=False).reset_index(drop=True)
        eligible["rank"] = eligible.index + 1
        rank_map = dict(zip(eligible["player_id"], eligible["rank"]))
        total = len(eligible)
        print(f"\n=== Known {label} (out of {total} eligible) ===")
        print(f"{'Name':30s} {'rank':>5} {'WAR':>8} {'WPA':>8} {'balls':>8}")
        for nm in known:
            pid = name_to_id.get(nm)
            if pid is None:
                print(f"{nm:30s}  N/A (not in players)")
                continue
            row = df_p[df_p["player_id"] == pid]
            if row.empty:
                print(f"{nm:30s}  — (no deliveries)")
                continue
            r = row.iloc[0]
            rk = rank_map.get(pid, "—")
            print(f"{nm:30s} {rk!s:>5} {r['war']:>+8.2f} {r['wpa']:>+8.2f} {int(r['balls']):>8,}")

    show_known("BATTERS", KNOWN_BATTERS, bat)
    show_known("BOWLERS", KNOWN_BOWLERS, bowl)


if __name__ == "__main__":
    main()
