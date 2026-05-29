"""Configuration and paths for t20x."""

from __future__ import annotations

from pathlib import Path

# Default data directory: ~/.t20x/
HOME_DIR = Path.home() / ".t20x"
DATA_DIR = HOME_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
DB_PATH = DATA_DIR / "t20x.duckdb"

# Cricsheet download URLs
CRICSHEET_BASE_URL = "https://cricsheet.org/downloads"

CRICSHEET_PEOPLE_URL = "https://cricsheet.org/register/people.csv"

# Map league keys to download URLs
# all_t20 includes ALL T20 matches (international + domestic) in one ZIP
CRICSHEET_LEAGUE_URLS: dict[str, str] = {
    "all_t20": f"{CRICSHEET_BASE_URL}/t20s_json.zip",
    "ipl": f"{CRICSHEET_BASE_URL}/ipl_json.zip",
    "bbl": f"{CRICSHEET_BASE_URL}/bbl_json.zip",
    "psl": f"{CRICSHEET_BASE_URL}/psl_json.zip",
    "cpl": f"{CRICSHEET_BASE_URL}/cpl_json.zip",
    "t20i": f"{CRICSHEET_BASE_URL}/t20s_json.zip",  # T20 internationals (same ZIP as all_t20)
    "the_hundred": f"{CRICSHEET_BASE_URL}/hnd_male_json.zip",
    "sa20": f"{CRICSHEET_BASE_URL}/sat_male_json.zip",
    "t20_blast": f"{CRICSHEET_BASE_URL}/ntb_male_json.zip",
}


def ensure_dirs() -> None:
    """Create data directories if they don't exist."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
