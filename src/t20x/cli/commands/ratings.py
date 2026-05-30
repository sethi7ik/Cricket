"""CLI commands for rating computation and player lookup."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from t20x.constants import LEAGUES
from t20x.db.engine import database
from t20x.ratings.era_relative import (
    add_year_column,
    career_arc,
    career_total_era_adjusted,
    compute_per_season_war,
    per_season_player_wpa,
)
from t20x.ratings.win_probability import (
    aggregate_player_wpa_by_phase,
    compute_war,
    compute_war_by_phase,
    compute_wpa,
)

app = typer.Typer(help="Rate, rank, and look up players via Win Probability Added.")
console = Console()


def _resolve_league(league: Optional[str]) -> Optional[str]:
    """Accept either a short code ('IPL') or display name ('Indian Premier League')."""
    if league is None:
        return None
    return LEAGUES.get(league.upper(), league)


def _build_name_resolver(conn, df: pd.DataFrame) -> tuple[dict[str, str], dict[str, str]]:
    """Returns (name_to_id, id_to_name). Disambiguates name collisions by picking
    the player_id with the most balls in the scored dataset.
    """
    rows = conn.execute("SELECT player_id, name FROM players").fetchall()
    activity = (
        pd.concat([df["batter_id"].rename("pid"), df["bowler_id"].rename("pid")])
        .value_counts()
        .rename("balls")
    )
    by_name: dict[str, tuple[str, int]] = {}
    id_to_name: dict[str, str] = {}
    for pid, name in rows:
        id_to_name[pid] = name
        balls = int(activity.get(pid, 0))
        cur = by_name.get(name)
        if cur is None or balls > cur[1]:
            by_name[name] = (pid, balls)
    return {n: pid for n, (pid, _) in by_name.items()}, id_to_name


def _render_top(title: str, df: pd.DataFrame, id_to_name: dict[str, str], top: int) -> None:
    table = Table(title=title)
    table.add_column("#", justify="right", style="dim")
    table.add_column("Player", style="cyan")
    table.add_column("WAR", justify="right", style="bold green")
    table.add_column("WPA", justify="right")
    table.add_column("WPA/ball", justify="right")
    table.add_column("Balls", justify="right")

    for i, row in enumerate(df.head(top).itertuples(), 1):
        name = id_to_name.get(row.player_id, row.player_id)
        table.add_row(
            str(i),
            name,
            f"{row.war:+.2f}",
            f"{row.wpa:+.2f}",
            f"{row.wpa_per_ball:+.5f}",
            f"{int(row.balls):,}",
        )
    console.print(table)


@app.command()
def rank(
    role: str = typer.Argument(..., help="batter | bowler"),
    league: Optional[str] = typer.Option(
        None, "--league", "-l", help="Filter to a single league (short code or display name)."
    ),
    phase: Optional[str] = typer.Option(
        None, "--phase", "-p", help="powerplay | middle | death. Omit for overall."
    ),
    top: int = typer.Option(20, "--top", "-n", help="Number of players to show."),
    min_balls: int = typer.Option(
        500, "--min-balls", help="Minimum balls (overall or in phase) to qualify."
    ),
    db_path: Optional[str] = typer.Option(None, "--db", help="Path to DuckDB file."),
) -> None:
    """Rank players by WAR via Win Probability Added.

    Examples:
        t20x ratings rank batter --league IPL --top 15
        t20x ratings rank bowler --phase death --min-balls 300
    """
    role = role.lower()
    if role not in ("batter", "bowler"):
        raise typer.BadParameter("role must be 'batter' or 'bowler'")
    if phase and phase not in ("powerplay", "middle", "death"):
        raise typer.BadParameter("phase must be powerplay | middle | death")

    league_resolved = _resolve_league(league)
    db = Path(db_path) if db_path else None

    with database(path=db, read_only=True) as conn:
        with console.status("Computing Δ_WP over deliveries..."):
            df = compute_wpa(conn, league=league_resolved)
        console.print(f"[dim]Scored {len(df):,} deliveries.[/]")

        if phase:
            bat, bowl = compute_war_by_phase(df, min_balls_phase=min_balls)
            bat = bat[bat["phase"] == phase]
            bowl = bowl[bowl["phase"] == phase]
        else:
            bat, bowl = compute_war(df, min_balls=min_balls)

        chosen = bat if role == "batter" else bowl
        chosen = chosen[chosen["balls"] >= min_balls].sort_values("war", ascending=False)
        _, id_to_name = _build_name_resolver(conn, df)

        scope = league_resolved or "all T20"
        phase_label = phase or "overall"
        _render_top(
            f"Top {top} {role.upper()}S by WAR — {scope}, {phase_label}",
            chosen,
            id_to_name,
            top,
        )


@app.command()
def show(
    name: str = typer.Argument(..., help="Player display name (e.g. 'V Kohli')."),
    league: Optional[str] = typer.Option(None, "--league", "-l"),
    db_path: Optional[str] = typer.Option(None, "--db", help="Path to DuckDB file."),
) -> None:
    """Show a player's WPA / WAR with phase breakdown."""
    league_resolved = _resolve_league(league)
    db = Path(db_path) if db_path else None

    with database(path=db, read_only=True) as conn:
        with console.status("Computing Δ_WP over deliveries..."):
            df = compute_wpa(conn, league=league_resolved)
        name_to_id, id_to_name = _build_name_resolver(conn, df)
        pid = name_to_id.get(name)
        if pid is None:
            console.print(f"[red]No player named {name!r} found.[/]")
            raise typer.Exit(1)

        # Use the same replacement cohort as `rank` so WAR is comparable.
        bat_all, bowl_all = compute_war(df, min_balls=500)
        bat_phase, bowl_phase = compute_war_by_phase(df, min_balls_phase=150)

        scope = league_resolved or "all T20"
        console.print(f"\n[bold]{name}[/] ({scope})\n")

        def render(role_label: str, overall: pd.DataFrame, phased: pd.DataFrame) -> None:
            row = overall[overall["player_id"] == pid]
            if row.empty:
                return
            r = row.iloc[0]
            console.print(
                f"  [bold]{role_label}[/] overall: WAR {r['war']:+.2f}   "
                f"WPA {r['wpa']:+.2f}   per-ball {r['wpa_per_ball']:+.5f}   balls {int(r['balls']):,}"
            )
            sub = phased[phased["player_id"] == pid]
            if sub.empty:
                return
            t = Table(show_header=True, header_style="bold")
            t.add_column("Phase", style="cyan")
            t.add_column("WAR", justify="right", style="green")
            t.add_column("WPA", justify="right")
            t.add_column("WPA/ball", justify="right")
            t.add_column("Balls", justify="right")
            for phase in ("powerplay", "middle", "death"):
                psub = sub[sub["phase"] == phase]
                if psub.empty:
                    continue
                pr = psub.iloc[0]
                t.add_row(
                    phase,
                    f"{pr['war']:+.2f}",
                    f"{pr['wpa']:+.2f}",
                    f"{pr['wpa_per_ball']:+.5f}",
                    f"{int(pr['balls']):,}",
                )
            console.print(t)

        render("BATTING", bat_all, bat_phase)
        render("BOWLING", bowl_all, bowl_phase)


@app.command()
def arc(
    name: str = typer.Argument(..., help="Player display name (e.g. 'CH Gayle')."),
    league: Optional[str] = typer.Option(None, "--league", "-l"),
    role: str = typer.Option("auto", "--role", help="batter | bowler | auto (both)"),
    min_balls: int = typer.Option(120, "--min-balls", help="Min balls per season to compute season replacement."),
    db_path: Optional[str] = typer.Option(None, "--db", help="Path to DuckDB file."),
) -> None:
    """Show a player's per-season WAR trajectory (era-relative).

    Each season uses its own replacement-level baseline so 2009 numbers are
    comparable to 2025 numbers.
    """
    league_resolved = _resolve_league(league)
    db = Path(db_path) if db_path else None

    with database(path=db, read_only=True) as conn:
        with console.status("Computing per-season Δ_WP..."):
            df = compute_wpa(conn, league=league_resolved)
            dates = conn.execute(
                "SELECT match_id, date FROM matches WHERE winner IS NOT NULL AND winner != ''"
            ).fetchdf()
            df = df.merge(dates, on="match_id", how="left")
            df = add_year_column(df)
            bat, bowl = per_season_player_wpa(df)
            bat, bowl = compute_per_season_war(bat, bowl, min_balls_season=min_balls)

        name_to_id, id_to_name = _build_name_resolver(conn, df)
        pid = name_to_id.get(name)
        if pid is None:
            console.print(f"[red]No player named {name!r} found.[/]")
            raise typer.Exit(1)

        scope = league_resolved or "all T20"
        console.print(f"\n[bold]{name}[/] — per-season WAR trajectory ({scope})\n")

        def render(label: str, df_p: pd.DataFrame) -> None:
            arc_df = career_arc(df_p, pid)
            if arc_df.empty:
                return
            tot = career_total_era_adjusted(df_p, pid)
            t = Table(title=f"{label}", show_header=True, header_style="bold")
            t.add_column("Season", justify="right", style="cyan")
            t.add_column("WAR", justify="right", style="green")
            t.add_column("WPA", justify="right")
            t.add_column("WPA/ball", justify="right")
            t.add_column("Balls", justify="right")
            for _, r in arc_df.iterrows():
                t.add_row(
                    str(int(r["year"])),
                    f"{r['war']:+.2f}",
                    f"{r['wpa']:+.2f}",
                    f"{r['wpa_per_ball']:+.5f}",
                    f"{int(r['balls']):,}",
                )
            console.print(t)
            console.print(
                f"  [dim]era-adj career WAR {tot['war_total']:+.2f}   "
                f"peak {tot['peak_season']} ({tot['peak_war']:+.2f})   "
                f"{tot['seasons']} seasons, {tot['balls_total']:,} balls[/]\n"
            )

        if role in ("auto", "batter"):
            render("BATTING", bat)
        if role in ("auto", "bowler"):
            render("BOWLING", bowl)
