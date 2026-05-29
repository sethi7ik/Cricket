"""Per-delivery Elo rating engine for batsman vs bowler matchups."""

from __future__ import annotations

from dataclasses import dataclass, field

from t20x.constants import ELO_INITIAL, ELO_K_FACTOR, ELO_SCALE, OUTCOME_SCORES


@dataclass
class EloRating:
    """A player's Elo rating with metadata."""

    rating: float = ELO_INITIAL
    deliveries: int = 0
    matches: int = 0


@dataclass
class EloEngine:
    """Per-delivery Elo engine treating each ball as a zero-sum game.

    Each delivery is a contest between batter and bowler. The outcome is
    mapped to a [0, 1] score, and ratings are updated using the standard
    Elo formula with configurable K-factor and opponent-quality weighting.
    """

    k_factor: float = ELO_K_FACTOR
    scale: float = ELO_SCALE
    initial_rating: float = ELO_INITIAL
    opponent_weight: bool = False
    outcome_scores: dict[str, float] = field(default_factory=lambda: dict(OUTCOME_SCORES))

    # Internal state: player_id -> EloRating
    bat_ratings: dict[str, EloRating] = field(default_factory=dict)
    bowl_ratings: dict[str, EloRating] = field(default_factory=dict)

    def _get_bat(self, player_id: str) -> EloRating:
        if player_id not in self.bat_ratings:
            self.bat_ratings[player_id] = EloRating(rating=self.initial_rating)
        return self.bat_ratings[player_id]

    def _get_bowl(self, player_id: str) -> EloRating:
        if player_id not in self.bowl_ratings:
            self.bowl_ratings[player_id] = EloRating(rating=self.initial_rating)
        return self.bowl_ratings[player_id]

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Expected score for player A against player B."""
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / self.scale))

    def delivery_outcome_score(
        self, batter_runs: int, is_wicket: bool, is_wide: bool, is_noball: bool
    ) -> float:
        """Map a delivery outcome to a [0, 1] score for the batsman.

        Returns:
            Score between 0 (worst for batter) and 1 (best for batter).
        """
        if is_wicket:
            return self.outcome_scores.get("W", 0.0)
        if is_wide or is_noball:
            # Extras from bowler errors — slightly positive for batter
            return 0.2
        return self.outcome_scores.get(str(batter_runs), 0.1)

    def compute_k(self, opponent_rating: float) -> float:
        """Compute adjusted K-factor based on opponent strength.

        Facing a stronger opponent amplifies rating changes.
        """
        if not self.opponent_weight:
            return self.k_factor

        # Scale K by how far opponent is from average
        strength_factor = (opponent_rating - self.initial_rating) / self.initial_rating
        # Clamp to avoid extreme K values
        multiplier = 1.0 + max(-0.5, min(0.5, strength_factor * 0.5))
        return self.k_factor * multiplier

    def update(
        self,
        batter_id: str,
        bowler_id: str,
        batter_runs: int,
        is_wicket: bool,
        is_wide: bool = False,
        is_noball: bool = False,
        expected_score: float | None = None,
    ) -> tuple[float, float]:
        """Update ratings for a single delivery.

        Args:
            batter_id: Batter's player ID.
            bowler_id: Bowler's player ID.
            batter_runs: Runs scored by the batter.
            is_wicket: Whether the batter was dismissed.
            is_wide: Whether it was a wide.
            is_noball: Whether it was a no-ball.
            expected_score: If provided, use this xR-driven expected score in [0, 1]
                instead of the rating-diff formula. Enables opponent-quality-adjusted
                Elo where the baseline is context-aware rather than only rating-based.

        Returns:
            Tuple of (new_batter_rating, new_bowler_rating).
        """
        bat = self._get_bat(batter_id)
        bowl = self._get_bowl(bowler_id)

        if expected_score is None:
            e_bat = self.expected_score(bat.rating, bowl.rating)
        else:
            e_bat = expected_score
        e_bowl = 1.0 - e_bat

        # Actual score (zero-sum)
        s_bat = self.delivery_outcome_score(batter_runs, is_wicket, is_wide, is_noball)
        s_bowl = 1.0 - s_bat

        # K-factor adjusted by opponent strength
        k_bat = self.compute_k(bowl.rating)
        k_bowl = self.compute_k(bat.rating)

        # Update ratings
        bat.rating += k_bat * (s_bat - e_bat)
        bowl.rating += k_bowl * (s_bowl - e_bowl)

        bat.deliveries += 1
        bowl.deliveries += 1

        return bat.rating, bowl.rating

    def sweep(
        self,
        deliveries: list[dict],
        prior_bat: dict[str, float] | None = None,
        prior_bowl: dict[str, float] | None = None,
        expected_scores: list[float] | None = None,
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Sweep through all deliveries chronologically, updating ratings.

        Args:
            deliveries: List of delivery dicts with keys:
                batter_id, bowler_id, batter_runs, is_wicket, is_wide, is_noball
            prior_bat: Optional prior batting ratings to initialize from.
            prior_bowl: Optional prior bowling ratings to initialize from.
            expected_scores: Optional per-delivery xR-derived expected scores in [0, 1],
                aligned to `deliveries`. When provided, Elo updates use the residual
                `actual − expected_score` instead of the rating-diff expected.

        Returns:
            Tuple of (bat_ratings_dict, bowl_ratings_dict) mapping player_id -> rating.
        """
        if prior_bat:
            for pid, rating in prior_bat.items():
                self.bat_ratings[pid] = EloRating(rating=rating)
        if prior_bowl:
            for pid, rating in prior_bowl.items():
                self.bowl_ratings[pid] = EloRating(rating=rating)

        if expected_scores is not None and len(expected_scores) != len(deliveries):
            raise ValueError(
                f"expected_scores length {len(expected_scores)} != deliveries {len(deliveries)}"
            )

        for i, d in enumerate(deliveries):
            self.update(
                batter_id=d["batter_id"],
                bowler_id=d["bowler_id"],
                batter_runs=d["batter_runs"],
                is_wicket=d["is_wicket"],
                is_wide=d.get("is_wide", False),
                is_noball=d.get("is_noball", False),
                expected_score=(None if expected_scores is None else float(expected_scores[i])),
            )

        bat_out = {pid: r.rating for pid, r in self.bat_ratings.items()}
        bowl_out = {pid: r.rating for pid, r in self.bowl_ratings.items()}
        return bat_out, bowl_out

    def get_ratings_snapshot(self) -> dict[str, dict]:
        """Get current ratings for all players.

        Returns:
            Dict with 'bat' and 'bowl' keys, each mapping player_id -> {rating, deliveries}.
        """
        return {
            "bat": {
                pid: {"rating": r.rating, "deliveries": r.deliveries}
                for pid, r in self.bat_ratings.items()
            },
            "bowl": {
                pid: {"rating": r.rating, "deliveries": r.deliveries}
                for pid, r in self.bowl_ratings.items()
            },
        }
