"""Pydantic domain models for Cricsheet JSON data."""

from __future__ import annotations

from datetime import date
from typing import Any

from pydantic import BaseModel, Field

from t20x.models.enums import Phase, over_to_phase


class Player(BaseModel):
    """A cricket player."""

    player_id: str
    name: str
    full_name: str | None = None
    dob: date | None = None
    batting_style: str | None = None
    bowling_type: str | None = None
    bowling_arm: str | None = None
    bowling_pace: str | None = None
    bowling_style: str | None = None
    primary_role: str | None = None
    country: str | None = None
    espn_id: int | None = None


class Delivery(BaseModel):
    """A single delivery (ball) in a cricket match."""

    match_id: str
    innings: int
    over_number: int
    ball_number: int
    delivery_seq: int = 0
    batter_id: str
    bowler_id: str
    non_striker_id: str
    batter_runs: int = 0
    extra_runs: int = 0
    total_runs: int = 0
    is_wide: bool = False
    is_noball: bool = False
    is_bye: bool = False
    is_legbye: bool = False
    is_boundary_four: bool = False
    is_boundary_six: bool = False
    is_dot: bool = False
    is_wicket: bool = False
    dismissal_kind: str | None = None
    player_out_id: str | None = None
    phase: Phase = Phase.POWERPLAY
    innings_runs: int = 0
    innings_wickets: int = 0
    run_rate: float = 0.0
    required_rate: float | None = None
    balls_remaining: int = 120

    @classmethod
    def from_cricsheet(
        cls,
        match_id: str,
        innings: int,
        over_number: int,
        ball_number: int,
        delivery_seq: int,
        ball_data: dict[str, Any],
        player_registry: dict[str, str],
        cumulative_runs: int,
        cumulative_wickets: int,
        target: int | None,
    ) -> Delivery:
        """Parse a delivery from Cricsheet JSON ball data."""
        runs = ball_data.get("runs", {})
        batter_runs = runs.get("batter", 0)
        extra_runs = runs.get("extras", 0)
        total_runs = runs.get("total", 0)

        extras = ball_data.get("extras", {})
        is_wide = "wides" in extras
        is_noball = "noballs" in extras
        is_bye = "byes" in extras
        is_legbye = "legbyes" in extras

        is_boundary_four = batter_runs == 4 and not is_bye and not is_legbye
        is_boundary_six = batter_runs == 6 and not is_bye and not is_legbye
        is_dot = total_runs == 0

        # Wicket info
        wickets = ball_data.get("wickets", [])
        is_wicket = len(wickets) > 0
        dismissal_kind = wickets[0]["kind"] if wickets else None
        player_out_raw = wickets[0]["player_out"] if wickets else None
        player_out_id = player_registry.get(player_out_raw) if player_out_raw else None

        # Resolve player IDs
        batter_id = player_registry.get(ball_data["batter"], ball_data["batter"])
        bowler_id = player_registry.get(ball_data["bowler"], ball_data["bowler"])
        non_striker_id = player_registry.get(
            ball_data.get("non_striker", ""), ball_data.get("non_striker", "")
        )

        phase = over_to_phase(over_number)

        # Compute balls bowled so far (legal deliveries only)
        balls_bowled = over_number * 6 + ball_number + 1
        run_rate = cumulative_runs / (balls_bowled / 6) if balls_bowled > 0 else 0.0
        balls_remaining = 120 - balls_bowled

        required_rate = None
        if target is not None and balls_remaining > 0:
            runs_needed = target - cumulative_runs
            overs_remaining = balls_remaining / 6
            required_rate = runs_needed / overs_remaining if overs_remaining > 0 else 999.0

        return cls(
            match_id=match_id,
            innings=innings,
            over_number=over_number,
            ball_number=ball_number,
            delivery_seq=delivery_seq,
            batter_id=batter_id,
            bowler_id=bowler_id,
            non_striker_id=non_striker_id,
            batter_runs=batter_runs,
            extra_runs=extra_runs,
            total_runs=total_runs,
            is_wide=is_wide,
            is_noball=is_noball,
            is_bye=is_bye,
            is_legbye=is_legbye,
            is_boundary_four=is_boundary_four,
            is_boundary_six=is_boundary_six,
            is_dot=is_dot,
            is_wicket=is_wicket,
            dismissal_kind=dismissal_kind,
            player_out_id=player_out_id,
            phase=phase,
            innings_runs=cumulative_runs + total_runs,
            innings_wickets=cumulative_wickets + (1 if is_wicket else 0),
            run_rate=run_rate,
            required_rate=required_rate,
            balls_remaining=balls_remaining,
        )


class MatchInfo(BaseModel):
    """Match-level information from Cricsheet JSON."""

    match_id: str
    league: str = ""
    season: str = ""
    date: date
    venue: str = ""
    city: str = ""
    team_1: str = ""
    team_2: str = ""
    toss_winner: str = ""
    toss_decision: str = ""
    winner: str = ""
    win_margin: int | None = None
    win_type: str = ""
    player_of_match: str = ""
    gender: str = "male"


class ParsedMatch(BaseModel):
    """A fully parsed match with metadata and all deliveries."""

    info: MatchInfo
    deliveries: list[Delivery] = Field(default_factory=list)
    players: dict[str, Player] = Field(default_factory=dict)
