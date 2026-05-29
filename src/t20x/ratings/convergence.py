"""Iterative convergence orchestrator for Elo + Bradley-Terry ratings.

The core algorithm: ratings for batsmen depend on bowler quality, and vice versa.
We iterate:
    1. Elo sweep through all deliveries (chronological)
    2. Bradley-Terry fit on aggregated matchups
    3. Blend Elo + BT ratings
    4. Check convergence
    5. Repeat with opponent-quality weighting from previous epoch
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import duckdb
import numpy as np
from tqdm import tqdm

from t20x.constants import OUTCOME_SCORES
from t20x.models.enums import Phase, over_to_phase
from t20x.ratings.bradley_terry import BradleyTerryModel
from t20x.ratings.elo import EloEngine
from t20x.ratings.expected_runs import ExpectedRunsModel


def _outcome_score_for(d: dict) -> float:
    if d["is_wicket"]:
        return OUTCOME_SCORES["W"]
    if d.get("is_wide", False) or d.get("is_noball", False):
        return 0.2
    return OUTCOME_SCORES.get(str(d["batter_runs"]), 0.1)


@dataclass
class EpochResult:
    """Results from one epoch of the convergence loop."""

    epoch: int
    bat_ratings: dict[str, float]
    bowl_ratings: dict[str, float]
    max_change: float
    converged: bool


@dataclass
class RatingOrchestrator:
    """Orchestrates iterative Elo + Bradley-Terry convergence.

    Runs the full rating computation pipeline:
    - Fetches deliveries from DuckDB
    - Runs iterative epochs until convergence
    - Stores results back to player_ratings table
    """

    max_epochs: int = 10
    epsilon: float = 1.0
    min_deliveries: int = 100
    elo_k: float = 0.5
    bt_blend_weight: float = 0.4  # Weight for BT in Elo/BT blend (Elo gets 1 - this)
    bt_min_deliveries: int = 30
    use_xr: bool = True  # If True, train xR and use residuals to drive Elo updates

    # Results per epoch
    history: list[EpochResult] = field(default_factory=list)
    # The xR model fitted once on the full slice (context-only features for v1)
    xr_model_: ExpectedRunsModel | None = None
    expected_scores_: np.ndarray | None = None

    def fetch_deliveries(
        self,
        conn: duckdb.DuckDBPyConnection,
        phase: Phase | None = None,
        league: str | None = None,
    ) -> list[dict]:
        """Fetch deliveries from DuckDB, ordered chronologically.

        Args:
            conn: DuckDB connection.
            phase: Filter to specific phase (None = all phases).
            league: Filter to specific league (None = all leagues).

        Returns:
            List of delivery dicts.
        """
        conditions = []
        if phase is not None:
            conditions.append(f"d.phase = '{phase.value}'")
        if league is not None:
            conditions.append(f"m.league = '{league}'")

        where = ""
        if conditions:
            where = "WHERE " + " AND ".join(conditions)

        query = f"""
            SELECT
                d.batter_id,
                d.bowler_id,
                d.batter_runs,
                d.is_wicket,
                d.is_wide,
                d.is_noball,
                d.over_number,
                d.phase,
                d.innings,
                d.innings_runs,
                d.innings_wickets,
                d.balls_remaining,
                d.required_rate,
                m.date
            FROM deliveries d
            JOIN matches m ON d.match_id = m.match_id
            {where}
            ORDER BY m.date, d.match_id, d.innings, d.over_number, d.ball_number, d.delivery_seq
        """
        rows = conn.execute(query).fetchall()
        columns = [
            "batter_id", "bowler_id", "batter_runs", "is_wicket",
            "is_wide", "is_noball", "over_number", "phase",
            "innings", "innings_runs", "innings_wickets",
            "balls_remaining", "required_rate", "date",
        ]
        return [dict(zip(columns, row)) for row in rows]

    def run(
        self,
        conn: duckdb.DuckDBPyConnection,
        phase: Phase | None = None,
        league: str | None = None,
        verbose: bool = True,
    ) -> list[EpochResult]:
        """Run the full iterative convergence loop.

        Args:
            conn: DuckDB connection with data loaded.
            phase: Filter to specific phase (None = overall).
            league: Filter to specific league (None = cross-league).
            verbose: Print progress.

        Returns:
            List of EpochResult for each epoch.
        """
        phase_label = phase.value if phase else "overall"
        league_label = league or "all"

        if verbose:
            print(f"Fetching deliveries (phase={phase_label}, league={league_label})...")

        deliveries = self.fetch_deliveries(conn, phase=phase, league=league)

        if not deliveries:
            if verbose:
                print("No deliveries found for the given filters.")
            return []

        if verbose:
            print(f"  {len(deliveries):,} deliveries loaded")

        # Train xR once on the full slice (features are context-only for v1).
        expected_scores: list[float] | None = None
        if self.use_xr:
            if verbose:
                print("Fitting xR model on full slice...")
            self.xr_model_ = ExpectedRunsModel().fit(deliveries)
            es = self.xr_model_.predict_expected_scores(deliveries)
            self.expected_scores_ = es
            expected_scores = es.tolist()
            if verbose:
                actual_mean = float(np.mean([
                    _outcome_score_for(d) for d in deliveries
                ]))
                print(
                    f"  xR fitted. expected_score mean={es.mean():.3f}  "
                    f"actual_score mean={actual_mean:.3f}"
                )

        bat_ratings: dict[str, float] = {}
        bowl_ratings: dict[str, float] = {}
        self.history = []

        for epoch in range(self.max_epochs):
            if verbose:
                print(f"\nEpoch {epoch + 1}/{self.max_epochs}")

            # --- Pass 1: Elo sweep ---
            elo = EloEngine(
                k_factor=self.elo_k,
                opponent_weight=(epoch > 0),  # Enable opponent weighting after first epoch
            )

            # Initialize from previous epoch ratings
            if bat_ratings and bowl_ratings:
                for pid, r in bat_ratings.items():
                    elo.bat_ratings[pid] = type(elo.bat_ratings.get(pid, elo._get_bat(pid)))(rating=r)
                for pid, r in bowl_ratings.items():
                    elo.bowl_ratings[pid] = type(elo.bowl_ratings.get(pid, elo._get_bowl(pid)))(rating=r)

            elo_bat, elo_bowl = elo.sweep(
                deliveries,
                prior_bat=bat_ratings or None,
                prior_bowl=bowl_ratings or None,
                expected_scores=expected_scores,
            )

            if verbose:
                print(f"  Elo: {len(elo_bat)} batters, {len(elo_bowl)} bowlers rated")

            # --- Pass 2: Bradley-Terry fit ---
            bt = BradleyTerryModel(min_deliveries=self.bt_min_deliveries)
            matchups = bt.aggregate_matchups(deliveries)
            bt_bat, bt_bowl = bt.fit(matchups)

            if verbose:
                print(f"  BT:  {len(bt_bat)} batters, {len(bt_bowl)} bowlers rated")

            # --- Pass 3: Blend ---
            new_bat: dict[str, float] = {}
            new_bowl: dict[str, float] = {}

            # All players from either model
            all_batters = set(elo_bat.keys()) | set(bt_bat.keys())
            all_bowlers = set(elo_bowl.keys()) | set(bt_bowl.keys())

            elo_w = 1.0 - self.bt_blend_weight
            bt_w = self.bt_blend_weight

            for pid in all_batters:
                e = elo_bat.get(pid)
                b = bt_bat.get(pid)
                if e is not None and b is not None:
                    new_bat[pid] = elo_w * e + bt_w * b
                elif e is not None:
                    new_bat[pid] = e
                else:
                    new_bat[pid] = b

            for pid in all_bowlers:
                e = elo_bowl.get(pid)
                b = bt_bowl.get(pid)
                if e is not None and b is not None:
                    new_bowl[pid] = elo_w * e + bt_w * b
                elif e is not None:
                    new_bowl[pid] = e
                else:
                    new_bowl[pid] = b

            # --- Pass 4: Check convergence ---
            max_change = 0.0
            if bat_ratings:
                for pid in new_bat:
                    if pid in bat_ratings:
                        # Only check players with enough data
                        elo_data = elo.bat_ratings.get(pid)
                        if elo_data and elo_data.deliveries >= self.min_deliveries:
                            change = abs(new_bat[pid] - bat_ratings[pid])
                            max_change = max(max_change, change)
                for pid in new_bowl:
                    if pid in bowl_ratings:
                        elo_data = elo.bowl_ratings.get(pid)
                        if elo_data and elo_data.deliveries >= self.min_deliveries:
                            change = abs(new_bowl[pid] - bowl_ratings[pid])
                            max_change = max(max_change, change)

            converged = epoch > 0 and max_change < self.epsilon

            result = EpochResult(
                epoch=epoch + 1,
                bat_ratings=dict(new_bat),
                bowl_ratings=dict(new_bowl),
                max_change=max_change,
                converged=converged,
            )
            self.history.append(result)

            if verbose:
                print(f"  Max rating change: {max_change:.2f} (threshold: {self.epsilon})")
                if converged:
                    print(f"  Converged after {epoch + 1} epochs!")

            bat_ratings = new_bat
            bowl_ratings = new_bowl

            if converged:
                break

        return self.history

    def save_ratings(
        self,
        conn: duckdb.DuckDBPyConnection,
        phase: Phase | None = None,
        league: str | None = None,
    ) -> int:
        """Save the final epoch's ratings to the player_ratings table.

        Returns:
            Number of rating rows inserted.
        """
        if not self.history:
            return 0

        final = self.history[-1]
        epoch = final.epoch
        phase_val = phase.value if phase else None
        league_val = league

        rows = []
        for pid, rating in final.bat_ratings.items():
            rows.append((pid, "blended_bat", phase_val, league_val, rating, None, None, None, epoch))
        for pid, rating in final.bowl_ratings.items():
            rows.append((pid, "blended_bowl", phase_val, league_val, rating, None, None, None, epoch))

        if rows:
            import pandas as pd
            df = pd.DataFrame(rows, columns=[
                "player_id", "rating_type", "phase", "league",
                "rating_value", "rating_variance", "matches_count",
                "deliveries_count", "epoch",
            ])
            conn.execute("INSERT INTO player_ratings SELECT * FROM df")

        return len(rows)

    def get_top_batters(self, n: int = 20, min_deliveries: int = 100) -> list[tuple[str, float]]:
        """Get top N batters from the final epoch."""
        if not self.history:
            return []
        final = self.history[-1]
        rated = [(pid, r) for pid, r in final.bat_ratings.items()]
        rated.sort(key=lambda x: x[1], reverse=True)
        return rated[:n]

    def get_top_bowlers(self, n: int = 20, min_deliveries: int = 100) -> list[tuple[str, float]]:
        """Get top N bowlers from the final epoch."""
        if not self.history:
            return []
        final = self.history[-1]
        rated = [(pid, r) for pid, r in final.bowl_ratings.items()]
        rated.sort(key=lambda x: x[1], reverse=True)
        return rated[:n]
