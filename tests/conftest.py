"""Shared test fixtures for t20x."""

import json
from pathlib import Path

import duckdb
import pytest

from t20x.db.engine import init_schema
from t20x.ingest.cricsheet import parse_match_json


@pytest.fixture
def sample_match_path() -> Path:
    return Path(__file__).parent / "data" / "sample_match.json"


@pytest.fixture
def sample_match_data(sample_match_path: Path) -> dict:
    with open(sample_match_path) as f:
        return json.load(f)


@pytest.fixture
def parsed_match(sample_match_data: dict):
    return parse_match_json(sample_match_data, match_id="test_001")


@pytest.fixture
def db() -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB for testing."""
    conn = duckdb.connect(":memory:")
    init_schema(conn)
    yield conn
    conn.close()
