"""Bradley-Terry pairwise comparison model for cricket matchups.

Fits strength parameters for batsmen and bowlers such that:
    P(batter dominates | batter i, bowler j) = alpha_i / (alpha_i + beta_j)

This resolves circular dependencies in ratings by solving all matchups simultaneously.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from math import log

import numpy as np


@dataclass
class MatchupRecord:
    """Aggregated outcome record for a batter-bowler pair."""

    batter_id: str
    bowler_id: str
    batter_wins: float = 0.0  # Sum of delivery scores for batter
    bowler_wins: float = 0.0  # Sum of delivery scores for bowler
    count: int = 0


@dataclass
class BradleyTerryModel:
    """Bradley-Terry model fitted via iterative MLE.

    The classic algorithm iterates:
        alpha_i = sum_j(w_ij) / sum_j(n_ij / (alpha_i + beta_j))

    where w_ij = batter i's win count vs bowler j, n_ij = total contests.
    """

    max_iter: int = 100
    tol: float = 1e-6
    min_deliveries: int = 30  # Minimum deliveries to include a player

    # Fitted parameters
    bat_strength: dict[str, float] = field(default_factory=dict)
    bowl_strength: dict[str, float] = field(default_factory=dict)

    def aggregate_matchups(
        self,
        deliveries: list[dict],
        outcome_scores: dict[str, float] | None = None,
    ) -> dict[tuple[str, str], MatchupRecord]:
        """Aggregate delivery-level data into pairwise matchup records.

        Args:
            deliveries: List of delivery dicts.
            outcome_scores: Mapping of outcome to [0,1] score. Default used if None.

        Returns:
            Dict mapping (batter_id, bowler_id) -> MatchupRecord.
        """
        if outcome_scores is None:
            from t20x.constants import OUTCOME_SCORES
            outcome_scores = OUTCOME_SCORES

        matchups: dict[tuple[str, str], MatchupRecord] = {}

        for d in deliveries:
            bat_id = d["batter_id"]
            bowl_id = d["bowler_id"]
            key = (bat_id, bowl_id)

            if key not in matchups:
                matchups[key] = MatchupRecord(batter_id=bat_id, bowler_id=bowl_id)

            rec = matchups[key]

            # Compute score
            if d["is_wicket"]:
                score = outcome_scores.get("W", 0.0)
            elif d.get("is_wide", False) or d.get("is_noball", False):
                score = 0.2
            else:
                score = outcome_scores.get(str(d["batter_runs"]), 0.1)

            rec.batter_wins += score
            rec.bowler_wins += 1.0 - score
            rec.count += 1

        return matchups

    def fit(
        self,
        matchups: dict[tuple[str, str], MatchupRecord],
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Fit Bradley-Terry model via iterative MLE.

        Args:
            matchups: Pairwise matchup records from aggregate_matchups().

        Returns:
            Tuple of (bat_strength, bowl_strength) dicts mapping player_id -> strength.
        """
        # Collect all players with enough data
        bat_counts: dict[str, int] = defaultdict(int)
        bowl_counts: dict[str, int] = defaultdict(int)
        for (bat_id, bowl_id), rec in matchups.items():
            bat_counts[bat_id] += rec.count
            bowl_counts[bowl_id] += rec.count

        batters = [b for b, c in bat_counts.items() if c >= self.min_deliveries]
        bowlers = [b for b, c in bowl_counts.items() if c >= self.min_deliveries]

        if not batters or not bowlers:
            return {}, {}

        # Initialize strengths to 1.0
        alpha = {b: 1.0 for b in batters}
        beta = {b: 1.0 for b in bowlers}

        batter_set = set(batters)
        bowler_set = set(bowlers)

        # Pre-compute wins and matchup lists per player
        bat_wins: dict[str, float] = defaultdict(float)
        bowl_wins: dict[str, float] = defaultdict(float)
        bat_matchups: dict[str, list[tuple[str, int]]] = defaultdict(list)
        bowl_matchups: dict[str, list[tuple[str, int]]] = defaultdict(list)

        for (bat_id, bowl_id), rec in matchups.items():
            if bat_id in batter_set and bowl_id in bowler_set:
                bat_wins[bat_id] += rec.batter_wins
                bowl_wins[bowl_id] += rec.bowler_wins
                bat_matchups[bat_id].append((bowl_id, rec.count))
                bowl_matchups[bowl_id].append((bat_id, rec.count))

        # Iterative update
        for iteration in range(self.max_iter):
            max_change = 0.0

            # Update batter strengths
            for bat_id in batters:
                w = bat_wins[bat_id]
                if w <= 0:
                    continue
                denom = sum(
                    n / (alpha[bat_id] + beta[bowl_id])
                    for bowl_id, n in bat_matchups[bat_id]
                )
                if denom > 0:
                    new_alpha = w / denom
                    max_change = max(max_change, abs(new_alpha - alpha[bat_id]))
                    alpha[bat_id] = new_alpha

            # Update bowler strengths
            for bowl_id in bowlers:
                w = bowl_wins[bowl_id]
                if w <= 0:
                    continue
                denom = sum(
                    n / (alpha[bat_id] + beta[bowl_id])
                    for bat_id, n in bowl_matchups[bowl_id]
                )
                if denom > 0:
                    new_beta = w / denom
                    max_change = max(max_change, abs(new_beta - beta[bowl_id]))
                    beta[bowl_id] = new_beta

            # Normalize: geometric mean of all strengths = 1
            all_vals = list(alpha.values()) + list(beta.values())
            geo_mean = np.exp(np.mean(np.log(np.array(all_vals) + 1e-10)))
            if geo_mean > 0:
                for k in alpha:
                    alpha[k] /= geo_mean
                for k in beta:
                    beta[k] /= geo_mean

            if max_change < self.tol:
                break

        # Convert to Elo-like scale for interpretability: rating = 1500 + 400 * log10(strength)
        def to_elo(strength: float) -> float:
            return 1500.0 + 400.0 * log(max(strength, 1e-10)) / log(10)

        self.bat_strength = {pid: to_elo(s) for pid, s in alpha.items()}
        self.bowl_strength = {pid: to_elo(s) for pid, s in beta.items()}

        return self.bat_strength, self.bowl_strength

    def predict_matchup(self, batter_id: str, bowler_id: str) -> float | None:
        """Predict probability that batter dominates a delivery against bowler.

        Returns:
            P(batter wins) or None if either player not in model.
        """
        if batter_id not in self.bat_strength or bowler_id not in self.bowl_strength:
            return None

        # Convert back from Elo to raw strength
        alpha = 10 ** ((self.bat_strength[batter_id] - 1500) / 400)
        beta = 10 ** ((self.bowl_strength[bowler_id] - 1500) / 400)
        return alpha / (alpha + beta)
