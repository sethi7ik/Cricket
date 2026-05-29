"""t20x CLI application."""

import typer

from t20x.cli.commands import enrich as enrich_cmd
from t20x.cli.commands import ingest as ingest_cmd
from t20x.cli.commands import ratings as ratings_cmd

app = typer.Typer(
    name="t20x",
    help="T20 cricket analytics: opponent-quality-adjusted ratings, expected runs, and cricket WAR.",
    no_args_is_help=True,
)

app.add_typer(ingest_cmd.app, name="ingest", help="Ingest cricket data from various sources.")
app.add_typer(ratings_cmd.app, name="ratings", help="Rank, compute, and look up player ratings.")
app.add_typer(enrich_cmd.app, name="enrich", help="Enrich player metadata from external sources.")


def main() -> None:
    app()
