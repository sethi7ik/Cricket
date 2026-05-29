"""Match state and situational context models."""

from __future__ import annotations

from pydantic import BaseModel

from t20x.models.enums import Phase


class MatchState(BaseModel):
    """Current state of a T20 innings at a given point."""

    innings: int
    over_number: int
    ball_number: int
    total_runs: int
    wickets_fallen: int
    run_rate: float
    required_rate: float | None = None
    phase: Phase
    balls_remaining: int
    target: int | None = None  # Only for second innings


class SituationContext(BaseModel):
    """Extended context for situation-specific analysis."""

    match_state: MatchState
    venue: str = ""
    league: str = ""
    batter_rating: float | None = None
    bowler_rating: float | None = None
    bowler_type: str | None = None
    batting_position: int | None = None
    is_chasing: bool = False

    @property
    def pressure_index(self) -> float:
        """Compute a pressure index based on match situation.

        Higher values = more pressure. Factors:
        - Required rate vs current rate (chasing)
        - Wickets fallen
        - Balls remaining (less = more pressure)
        - Phase (death > middle > powerplay for batting pressure)
        """
        pressure = 0.0

        # Wicket pressure: each wicket adds pressure
        pressure += self.match_state.wickets_fallen * 0.1

        # Rate pressure (chasing only)
        if self.match_state.required_rate is not None and self.match_state.run_rate > 0:
            rate_diff = self.match_state.required_rate - self.match_state.run_rate
            pressure += max(0, rate_diff) * 0.15

        # Time pressure: fewer balls = more pressure
        balls_fraction = self.match_state.balls_remaining / 120
        pressure += (1 - balls_fraction) * 0.3

        # Phase multiplier
        phase_mult = {Phase.POWERPLAY: 0.8, Phase.MIDDLE: 1.0, Phase.DEATH: 1.3}
        pressure *= phase_mult.get(self.match_state.phase, 1.0)

        return round(min(pressure, 1.0), 3)
