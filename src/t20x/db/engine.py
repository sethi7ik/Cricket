"""DuckDB database engine for t20x."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import duckdb

from t20x.config import DB_PATH, ensure_dirs
from t20x.db.schema import INDEX_SQL, SCHEMA_SQL


def get_db(path: Path | str | None = None, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Get a DuckDB connection, creating the database and schema if needed.

    Args:
        path: Database file path. None = default (~/.t20x/data/t20x.duckdb).
              Use ":memory:" for in-memory database (tests).
        read_only: Open in read-only mode.
    """
    if path is None:
        ensure_dirs()
        path = DB_PATH

    db_path = str(path)
    conn = duckdb.connect(db_path, read_only=read_only)
    return conn


def init_schema(conn: duckdb.DuckDBPyConnection) -> None:
    """Create all tables and indexes."""
    conn.execute(SCHEMA_SQL)
    conn.execute(INDEX_SQL)


@contextmanager
def database(
    path: Path | str | None = None, read_only: bool = False
) -> Generator[duckdb.DuckDBPyConnection, None, None]:
    """Context manager for database connections.

    Usage:
        with database() as conn:
            conn.execute("SELECT * FROM matches")
    """
    conn = get_db(path, read_only=read_only)
    try:
        if not read_only:
            init_schema(conn)
        yield conn
    finally:
        conn.close()


def reset_database(conn: duckdb.DuckDBPyConnection) -> None:
    """Drop all tables and recreate. Use with caution."""
    tables = ["expected_runs", "player_ratings", "deliveries", "players", "matches"]
    for table in tables:
        conn.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
    init_schema(conn)
