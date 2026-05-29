"""CLI commands for enriching player metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from t20x.db.engine import database
from t20x.ingest.bowler_styles import coverage_report, enrich_from_csv

app = typer.Typer(help="Enrich player metadata from external sources.")
console = Console()


@app.command(name="bowler-styles")
def bowler_styles(
    csv: str = typer.Option(
        "metadata/bowler_styles.csv",
        "--csv",
        help="Path to the bowler styles CSV.",
    ),
    db_path: Optional[str] = typer.Option(None, "--db", help="Path to DuckDB file."),
) -> None:
    """Apply curated bowling-style metadata to the players table."""
    csv_path = Path(csv)
    if not csv_path.exists():
        console.print(f"[red]CSV not found: {csv_path}[/]")
        raise typer.Exit(1)

    db = Path(db_path) if db_path else None
    with database(path=db) as conn:
        counts = enrich_from_csv(conn, csv_path)
        console.print(
            f"\nApplied {counts['matched']}/{counts['total_rows']} rows "
            f"(missing: {counts['missing']}).\n"
        )

        report = coverage_report(conn)
        table = Table(title="Bowler-style coverage")
        table.add_column("State", style="cyan")
        table.add_column("Bucket", style="magenta")
        table.add_column("Distinct bowlers", justify="right")
        table.add_column("Deliveries", justify="right", style="green")
        for _, r in report.iterrows():
            table.add_row(
                str(r["state"]),
                str(r["bucket"]),
                f"{int(r['bowlers']):,}",
                f"{int(r['deliveries']):,}",
            )
        console.print(table)
