"""Cricsheet JSON data ingestion — primary data source for t20x."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Iterator

import httpx
from tqdm import tqdm

from t20x.config import CRICSHEET_LEAGUE_URLS, RAW_DIR, ensure_dirs
from t20x.ingest.registry import build_name_to_id_map
from t20x.models.domain import Delivery, MatchInfo, ParsedMatch, Player


def download_cricsheet(league: str = "all_t20", force: bool = False) -> Path:
    """Download a Cricsheet JSON ZIP file.

    Args:
        league: League key (all_t20, ipl, bbl, psl, cpl).
        force: Re-download even if file exists.

    Returns:
        Path to the downloaded ZIP file.
    """
    ensure_dirs()
    url = CRICSHEET_LEAGUE_URLS.get(league)
    if url is None:
        raise ValueError(f"Unknown league: {league}. Options: {list(CRICSHEET_LEAGUE_URLS)}")

    filename = url.split("/")[-1]
    dest = RAW_DIR / filename
    if dest.exists() and not force:
        return dest

    print(f"Downloading {url} ...")
    with httpx.stream("GET", url, follow_redirects=True, timeout=300) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
            for chunk in resp.iter_bytes(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))

    return dest


def parse_match_json(match_data: dict, match_id: str) -> ParsedMatch | None:
    """Parse a single Cricsheet JSON match into domain objects.

    Args:
        match_data: Parsed JSON dict from a Cricsheet match file.
        match_id: The match identifier (usually filename without extension).

    Returns:
        ParsedMatch or None if the match can't be parsed.
    """
    info = match_data.get("info", {})

    # Only process T20 matches
    match_type = info.get("match_type", "")
    if match_type not in ("T20", "IT20"):
        return None

    # Build player name -> ID registry from this match
    registry_people = info.get("registry", {}).get("people", {})
    name_to_id = build_name_to_id_map(registry_people)

    # Parse match info
    teams = info.get("teams", [])
    dates = info.get("dates", [])
    outcome = info.get("outcome", {})
    toss = info.get("toss", {})
    event = info.get("event", {})

    winner = outcome.get("winner", "")
    win_by = outcome.get("by", {})
    win_margin = win_by.get("runs") or win_by.get("wickets")
    win_type = "runs" if "runs" in win_by else ("wickets" if "wickets" in win_by else "")

    pom_list = info.get("player_of_match", [])
    player_of_match = pom_list[0] if pom_list else ""

    match_info = MatchInfo(
        match_id=match_id,
        league=event.get("name", info.get("match_type", "")),
        season=str(info.get("season", "")),
        date=dates[0] if dates else "2000-01-01",
        venue=info.get("venue", ""),
        city=info.get("city", ""),
        team_1=teams[0] if len(teams) > 0 else "",
        team_2=teams[1] if len(teams) > 1 else "",
        toss_winner=toss.get("winner", ""),
        toss_decision=toss.get("decision", ""),
        winner=winner,
        win_margin=win_margin,
        win_type=win_type,
        player_of_match=player_of_match,
        gender=info.get("gender", "male"),
    )

    # Collect players from the match
    players: dict[str, Player] = {}
    players_by_team = info.get("players", {})
    for team_name, player_names in players_by_team.items():
        for pname in player_names:
            pid = name_to_id.get(pname, pname)
            if pid not in players:
                players[pid] = Player(player_id=pid, name=pname, country=team_name)

    # Parse innings and deliveries
    all_deliveries: list[Delivery] = []
    innings_list = match_data.get("innings", [])

    # Determine target for second innings
    first_innings_total = 0

    for innings_idx, innings_data in enumerate(innings_list):
        innings_num = innings_idx + 1
        overs = innings_data.get("overs", [])

        target = None
        if innings_num == 2:
            target = first_innings_total + 1  # Need to beat first innings total
            # Check if explicit target is set
            target_data = innings_data.get("target", {})
            if target_data:
                target = target_data.get("runs", target)

        cumulative_runs = 0
        cumulative_wickets = 0
        delivery_seq = 0

        for over_data in overs:
            over_number = over_data["over"]
            deliveries = over_data.get("deliveries", [])

            for ball_idx, ball_data in enumerate(deliveries):
                delivery = Delivery.from_cricsheet(
                    match_id=match_id,
                    innings=innings_num,
                    over_number=over_number,
                    ball_number=ball_idx,
                    delivery_seq=delivery_seq,
                    ball_data=ball_data,
                    player_registry=name_to_id,
                    cumulative_runs=cumulative_runs,
                    cumulative_wickets=cumulative_wickets,
                    target=target,
                )
                all_deliveries.append(delivery)

                # Update cumulative state
                cumulative_runs += delivery.total_runs
                if delivery.is_wicket:
                    cumulative_wickets += 1
                delivery_seq += 1

        # Store first innings total for target calculation
        if innings_num == 1:
            first_innings_total = cumulative_runs

    return ParsedMatch(info=match_info, deliveries=all_deliveries, players=players)


def iter_matches_from_zip(zip_path: Path) -> Iterator[ParsedMatch]:
    """Iterate over all matches in a Cricsheet JSON ZIP file.

    Yields:
        ParsedMatch objects for each successfully parsed T20 match.
    """
    with zipfile.ZipFile(zip_path, "r") as zf:
        json_files = [f for f in zf.namelist() if f.endswith(".json") and not f.startswith("__")]
        for filename in tqdm(json_files, desc="Parsing matches", unit="match"):
            try:
                with zf.open(filename) as f:
                    match_data = json.loads(f.read())
                match_id = Path(filename).stem
                parsed = parse_match_json(match_data, match_id)
                if parsed is not None:
                    yield parsed
            except Exception as e:
                # Log and skip malformed matches
                tqdm.write(f"Warning: skipping {filename}: {e}")
                continue


class CricsheetSource:
    """Cricsheet data source implementation."""

    def __init__(self, league: str = "all_t20"):
        self.league = league

    def parse(self, path: str | None = None) -> Iterator[ParsedMatch]:
        """Parse matches from Cricsheet.

        Args:
            path: Path to a local Cricsheet JSON ZIP. None = auto-download.
        """
        if path is not None:
            zip_path = Path(path)
            if not zip_path.exists():
                raise FileNotFoundError(f"File not found: {path}")
        else:
            zip_path = download_cricsheet(league=self.league)

        yield from iter_matches_from_zip(zip_path)
