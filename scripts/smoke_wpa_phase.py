"""Phase-specific WPA/WAR smoke test on IPL data."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from t20x.ratings.win_probability import compute_war_by_phase, compute_wpa


DB_PATH = Path.home() / ".t20x" / "data" / "t20x.duckdb"
LEAGUE = "Indian Premier League"
TOP_N = 10
MIN_BALLS_PHASE = 150
PHASES = ("powerplay", "middle", "death")


def build_name_to_id(conn: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> dict[str, str]:
    rows = conn.execute("SELECT player_id, name FROM players").fetchall()
    activity = (
        pd.concat([df["batter_id"].rename("pid"), df["bowler_id"].rename("pid")])
        .value_counts()
        .rename("balls")
    )
    by_name: dict[str, tuple[str, int]] = {}
    for pid, name in rows:
        balls = int(activity.get(pid, 0))
        cur = by_name.get(name)
        if cur is None or balls > cur[1]:
            by_name[name] = (pid, balls)
    return {n: pid for n, (pid, _) in by_name.items()}


def show_phase(label: str, phase: str, df_p: pd.DataFrame, id_to_name: dict[str, str], n: int, min_balls: int) -> None:
    sub = df_p[(df_p["phase"] == phase) & (df_p["balls"] >= min_balls)].copy()
    sub = sub.sort_values("war", ascending=False).head(n)
    print(f"\n=== Top {n} {label} — {phase.upper()} (min {min_balls} balls in phase) ===")
    print(f"{'#':>3} {'Name':30s} {'WAR':>7} {'WPA':>7} {'WPA/ball':>10} {'balls':>7}")
    for i, row in enumerate(sub.itertuples(), 1):
        nm = id_to_name.get(row.player_id, row.player_id)
        print(f"{i:>3} {nm:30s} {row.war:>+7.2f} {row.wpa:>+7.2f} {row.wpa_per_ball:>+10.5f} {row.balls:>7,}")


def main() -> None:
    conn = duckdb.connect(str(DB_PATH), read_only=True)
    df = compute_wpa(conn, league=LEAGUE)
    print(f"Computed Δ_WP for {len(df):,} deliveries")

    # Phase mix
    phase_mix = df["phase"].value_counts()
    print(f"\nDeliveries by phase:\n{phase_mix.to_string()}")
    print("\nMean |Δ_WP| by phase (gives a sense of swing magnitude):")
    print(df.groupby("phase")["delta_wp"].apply(lambda s: s.abs().mean()).to_string())

    bat, bowl = compute_war_by_phase(df, replacement_pct=25.0, min_balls_phase=MIN_BALLS_PHASE)
    print(f"\nReplacement-level by phase (WPA/ball):")
    print("  Batters:", bat.attrs["replacement_per_ball_by_phase"])
    print("  Bowlers:", bowl.attrs["replacement_per_ball_by_phase"])

    name_to_id = build_name_to_id(conn, df)
    id_to_name = {pid: n for n, pid in name_to_id.items()}
    # Also map any other player_ids appearing
    all_names = dict(conn.execute("SELECT player_id, name FROM players").fetchall())
    for pid, n in all_names.items():
        id_to_name.setdefault(pid, n)

    for phase in PHASES:
        show_phase("BATTERS", phase, bat, id_to_name, TOP_N, MIN_BALLS_PHASE)
    for phase in PHASES:
        show_phase("BOWLERS", phase, bowl, id_to_name, TOP_N, MIN_BALLS_PHASE)


if __name__ == "__main__":
    main()
