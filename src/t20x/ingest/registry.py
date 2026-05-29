"""Player identity resolution using Cricsheet registry."""

from __future__ import annotations

import csv
import io
from pathlib import Path

import httpx

from t20x.config import CRICSHEET_PEOPLE_URL, RAW_DIR, ensure_dirs


def download_people_csv(force: bool = False) -> Path:
    """Download the Cricsheet people.csv registry.

    Returns:
        Path to the downloaded CSV file.
    """
    ensure_dirs()
    dest = RAW_DIR / "people.csv"
    if dest.exists() and not force:
        return dest

    resp = httpx.get(CRICSHEET_PEOPLE_URL, follow_redirects=True, timeout=60)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return dest


def load_people_registry(path: Path | None = None) -> dict[str, dict]:
    """Load the Cricsheet people registry.

    Returns:
        Dict mapping identifier -> {name, unique_name, ...}
    """
    if path is None:
        path = download_people_csv()

    registry: dict[str, dict] = {}
    text = path.read_text(encoding="utf-8")
    reader = csv.DictReader(io.StringIO(text))
    for row in reader:
        identifier = row.get("identifier", "").strip()
        if identifier:
            registry[identifier] = {
                "name": row.get("name", "").strip(),
                "unique_name": row.get("unique_name", "").strip(),
            }
    return registry


def build_name_to_id_map(registry_in_match: dict[str, str]) -> dict[str, str]:
    """Build a name -> player_id map from a single match's registry.people section.

    The Cricsheet JSON has:
        "registry": {"people": {"Player Name": "unique_id", ...}}

    Args:
        registry_in_match: The "people" dict from a match JSON.

    Returns:
        Dict mapping player display name -> canonical ID.
    """
    return {name: pid for name, pid in registry_in_match.items()}
