"""CLI commands for data ingestion."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from t20x.config import CRICSHEET_LEAGUE_URLS
from t20x.db.engine import database
from t20x.ingest.cricsheet import CricsheetSource
from t20x.ingest.loader import load_matches

app = typer.Typer()
console = Console()


@app.callback(invoke_without_command=True)
def cricsheet(
    league: str = typer.Option(
        "all_t20",
        "--league",
        "-l",
        help=f"League to ingest. Options: {', '.join(CRICSHEET_LEAGUE_URLS.keys())}",
    ),
    path: Optional[str] = typer.Option(
        None,
        "--path",
        "-p",
        help="Path to a local Cricsheet JSON ZIP file. Omit to auto-download.",
    ),
    db_path: Optional[str] = typer.Option(
        None,
        "--db",
        help="Path to DuckDB file. Omit for default (~/.t20x/data/t20x.duckdb).",
    ),
    force_download: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force re-download even if cached.",
    ),
) -> None:
    """Ingest ball-by-ball data from Cricsheet into the t20x database."""
    console.print(f"[bold blue]t20x[/] — Ingesting data from Cricsheet ({league})")

    source = CricsheetSource(league=league)

    if force_download and path is None:
        from t20x.ingest.cricsheet import download_cricsheet

        download_cricsheet(league=league, force=True)

    db = Path(db_path) if db_path else None

    with database(path=db) as conn:
        console.print("Parsing and loading matches...")
        matches_iter = source.parse(path=path)
        counts = load_matches(conn, matches_iter)

        # Show results
        table = Table(title="Ingestion Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Count", style="green", justify="right")
        table.add_row("Matches loaded", f"{counts['matches']:,}")
        table.add_row("Deliveries loaded", f"{counts['deliveries']:,}")
        table.add_row("Players registered", f"{counts['players']:,}")
        console.print(table)

        # Quick sanity check
        row = conn.execute("SELECT COUNT(*) FROM matches").fetchone()
        total_matches = row[0] if row else 0
        row = conn.execute("SELECT COUNT(*) FROM deliveries").fetchone()
        total_deliveries = row[0] if row else 0
        console.print(f"\n[bold]Database totals:[/] {total_matches:,} matches, {total_deliveries:,} deliveries")


@app.command()
def status(
    db_path: Optional[str] = typer.Option(None, "--db", help="Path to DuckDB file."),
) -> None:
    """Show current database status."""
    db = Path(db_path) if db_path else None
    with database(path=db, read_only=True) as conn:
        table = Table(title="t20x Database Status")
        table.add_column("Table", style="cyan")
        table.add_column("Rows", style="green", justify="right")

        for tbl in ["matches", "players", "deliveries", "player_ratings", "expected_runs"]:
            row = conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()
            count = row[0] if row else 0
            table.add_row(tbl, f"{count:,}")

        console.print(table)

        # Top leagues
        rows = conn.execute(
            "SELECT league, COUNT(*) as n FROM matches GROUP BY league ORDER BY n DESC LIMIT 10"
        ).fetchall()
        if rows:
            league_table = Table(title="Matches by League")
            league_table.add_column("League", style="cyan")
            league_table.add_column("Matches", style="green", justify="right")
            for row in rows:
                league_table.add_row(str(row[0]), f"{row[1]:,}")
            console.print(league_table)
