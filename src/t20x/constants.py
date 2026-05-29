"""Constants for T20 cricket analytics."""

from __future__ import annotations

# T20 phase boundaries (over numbers, 0-indexed)
POWERPLAY_END = 6  # Overs 1-6 (0-5 in 0-indexed)
MIDDLE_END = 15  # Overs 7-15 (6-14 in 0-indexed)
DEATH_END = 20  # Overs 16-20 (15-19 in 0-indexed)

# Elo defaults
ELO_INITIAL = 1500.0
ELO_K_FACTOR = 0.5
ELO_SCALE = 400.0

# Outcome-to-score mapping for Elo (batsman perspective)
# Higher = better for batsman, lower = better for bowler
OUTCOME_SCORES: dict[str, float] = {
    "W": 0.0,  # Wicket (bowler dismissal)
    "0": 0.1,  # Dot ball
    "1": 0.3,  # Single
    "2": 0.5,  # Two runs
    "3": 0.6,  # Three runs
    "4": 0.8,  # Boundary four
    "6": 1.0,  # Six
}

# Balls per over
BALLS_PER_OVER = 6

# Total balls in a T20 innings
TOTAL_BALLS_T20 = 120

# Known T20 leagues and their Cricsheet identifiers
LEAGUES: dict[str, str] = {
    "IPL": "Indian Premier League",
    "BBL": "Big Bash League",
    "PSL": "Pakistan Super League",
    "CPL": "Caribbean Premier League",
    "T20I": "T20 International",
    "T20 Blast": "T20 Blast",
    "SA20": "SA20",
    "MLC": "Major League Cricket",
    "ILT20": "International League T20",
    "LPL": "Lanka Premier League",
    "BPL": "Bangladesh Premier League",
}
