"""Tests for Cricsheet ingestion."""

from t20x.ingest.loader import load_matches
from t20x.models.enums import Phase


def test_load_matches_into_db(db, parsed_match):
    """Test loading a parsed match into DuckDB."""
    counts = load_matches(db, iter([parsed_match]))

    assert counts["matches"] == 1
    assert counts["deliveries"] > 0
    assert counts["players"] > 0

    # Verify match in DB
    row = db.execute("SELECT COUNT(*) FROM matches").fetchone()
    assert row[0] == 1

    # Verify deliveries
    row = db.execute("SELECT COUNT(*) FROM deliveries").fetchone()
    assert row[0] == counts["deliveries"]

    # Verify players
    row = db.execute("SELECT COUNT(*) FROM players").fetchone()
    assert row[0] >= 1


def test_delivery_phases_in_db(db, parsed_match):
    """Test that phases are correctly stored."""
    load_matches(db, iter([parsed_match]))

    phases = db.execute("SELECT DISTINCT phase FROM deliveries").fetchall()
    phase_values = {row[0] for row in phases}
    assert Phase.POWERPLAY.value in phase_values  # Our sample is only overs 0-1


def test_boundary_flags_in_db(db, parsed_match):
    """Test boundary detection."""
    load_matches(db, iter([parsed_match]))

    fours = db.execute("SELECT COUNT(*) FROM deliveries WHERE is_boundary_four = true").fetchone()
    sixes = db.execute("SELECT COUNT(*) FROM deliveries WHERE is_boundary_six = true").fetchone()
    assert fours[0] >= 1  # At least one four in sample data
    assert sixes[0] >= 1  # At least one six in sample data


def test_wickets_in_db(db, parsed_match):
    """Test wicket detection."""
    load_matches(db, iter([parsed_match]))

    wickets = db.execute("SELECT COUNT(*) FROM deliveries WHERE is_wicket = true").fetchone()
    assert wickets[0] >= 1


def test_no_duplicate_on_reload(db, parsed_match):
    """Test that reloading the same match doesn't create duplicates."""
    load_matches(db, iter([parsed_match]))
    load_matches(db, iter([parsed_match]))

    row = db.execute("SELECT COUNT(*) FROM matches").fetchone()
    assert row[0] == 1  # Still only 1 match


def test_player_ids_resolved(db, parsed_match):
    """Test that player IDs are Cricsheet UUIDs, not raw names."""
    load_matches(db, iter([parsed_match]))

    batters = db.execute("SELECT DISTINCT batter_id FROM deliveries").fetchall()
    batter_ids = {row[0] for row in batters}
    # Should be UUIDs like "bat001", not names like "Batter One"
    assert "bat001" in batter_ids
    assert "Batter One" not in batter_ids
