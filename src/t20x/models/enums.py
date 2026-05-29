"""Enumerations for T20 cricket domain."""

from enum import Enum


class Phase(str, Enum):
    """T20 innings phases."""

    POWERPLAY = "powerplay"  # Overs 1-6
    MIDDLE = "middle"  # Overs 7-15
    DEATH = "death"  # Overs 16-20


class BowlingArm(str, Enum):
    """Bowling arm."""

    RIGHT = "right"
    LEFT = "left"


class BowlingPace(str, Enum):
    """Broad bowling pace category."""

    FAST = "fast"
    MEDIUM_FAST = "medium-fast"
    MEDIUM = "medium"
    SPIN = "spin"


class BowlingStyle(str, Enum):
    """Specific bowling style."""

    RIGHT_ARM_FAST = "right-arm fast"
    RIGHT_ARM_MEDIUM_FAST = "right-arm medium-fast"
    RIGHT_ARM_MEDIUM = "right-arm medium"
    LEFT_ARM_FAST = "left-arm fast"
    LEFT_ARM_MEDIUM_FAST = "left-arm medium-fast"
    LEFT_ARM_MEDIUM = "left-arm medium"
    RIGHT_ARM_OFFBREAK = "right-arm offbreak"
    RIGHT_ARM_LEGBREAK = "right-arm legbreak"
    LEFT_ARM_ORTHODOX = "left-arm orthodox"
    LEFT_ARM_WRIST_SPIN = "left-arm wrist spin"
    SLOW_LEFT_ARM_ORTHODOX = "slow left-arm orthodox"


class BattingStyle(str, Enum):
    """Batting hand."""

    RIGHT = "right-hand bat"
    LEFT = "left-hand bat"


class DismissalKind(str, Enum):
    """Types of dismissal."""

    BOWLED = "bowled"
    CAUGHT = "caught"
    LBW = "lbw"
    RUN_OUT = "run out"
    STUMPED = "stumped"
    CAUGHT_AND_BOWLED = "caught and bowled"
    HIT_WICKET = "hit wicket"
    OBSTRUCTING_FIELD = "obstructing the field"
    RETIRED_HURT = "retired hurt"
    RETIRED_OUT = "retired out"
    TIMED_OUT = "timed out"
    HANDLED_BALL = "handled the ball"


class PlayerRole(str, Enum):
    """Primary player role."""

    BATSMAN = "batsman"
    BOWLER = "bowler"
    ALLROUNDER = "allrounder"
    WICKETKEEPER = "wicketkeeper"


class RatingType(str, Enum):
    """Types of player ratings."""

    ELO_BAT = "elo_bat"
    ELO_BOWL = "elo_bowl"
    BT_BAT = "bt_bat"
    BT_BOWL = "bt_bowl"
    BAYESIAN_BAT = "bayesian_bat"
    BAYESIAN_BOWL = "bayesian_bowl"
    BLENDED_BAT = "blended_bat"
    BLENDED_BOWL = "blended_bowl"


def over_to_phase(over_number: int) -> Phase:
    """Convert 0-indexed over number to phase."""
    if over_number < 6:
        return Phase.POWERPLAY
    elif over_number < 15:
        return Phase.MIDDLE
    else:
        return Phase.DEATH
