"""Expected Runs (xR) model for T20 deliveries.

Predicts the distribution over delivery outcomes {0, 1, 2, 3, 4, 6, W} given match
context, then maps that distribution to an expected outcome score in [0, 1] using
the same `OUTCOME_SCORES` mapping the Elo engine uses.

The residual `actual_score - expected_score` is the zero-mean signal that drives
opponent-quality-adjusted Elo updates: a batter who scores 1 off Bumrah in the death
gets credit for beating xR; a batter who blocks 0 off a part-timer in the powerplay
gets penalized.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from t20x.constants import OUTCOME_SCORES

OUTCOME_CLASSES: tuple[str, ...] = ("W", "0", "1", "2", "3", "4", "6")

FEATURE_COLS: tuple[str, ...] = (
    "over_number",
    "innings",
    "innings_runs",
    "innings_wickets",
    "balls_remaining",
    "required_rate",
)


def classify_delivery(batter_runs: int, is_wicket: bool, is_wide: bool, is_noball: bool) -> str:
    """Map a delivery to one of the OUTCOME_CLASSES.

    Wides/no-balls collapse to '1' (closest legal-delivery equivalent for scoring purposes).
    Rare 5-run balls (overthrows) collapse to '4'.
    """
    if is_wicket:
        return "W"
    if is_wide or is_noball:
        return "1"
    runs = int(batter_runs)
    if runs >= 6:
        return "6"
    if runs == 5:
        return "4"
    return str(runs)


@dataclass
class ExpectedRunsModel:
    """Context-aware xR model predicting outcome distributions per delivery."""

    max_iter: int = 200
    learning_rate: float = 0.05
    max_depth: int | None = 6
    min_samples_leaf: int = 200
    random_state: int = 42

    classifier_: HistGradientBoostingClassifier | None = None
    class_to_score_: np.ndarray | None = field(default=None, repr=False)

    def _build_features(self, deliveries: list[dict]) -> pd.DataFrame:
        df = pd.DataFrame(deliveries)
        for col in FEATURE_COLS:
            if col not in df.columns:
                df[col] = np.nan
        return df[list(FEATURE_COLS)].astype("float64")

    def _build_labels(self, deliveries: list[dict]) -> np.ndarray:
        return np.array(
            [
                classify_delivery(
                    d["batter_runs"],
                    d["is_wicket"],
                    d.get("is_wide", False),
                    d.get("is_noball", False),
                )
                for d in deliveries
            ]
        )

    def fit(self, deliveries: list[dict]) -> "ExpectedRunsModel":
        X = self._build_features(deliveries)
        y = self._build_labels(deliveries)

        clf = HistGradientBoostingClassifier(
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
        )
        clf.fit(X, y)
        self.classifier_ = clf

        # Precompute score lookup aligned to clf.classes_
        self.class_to_score_ = np.array(
            [OUTCOME_SCORES[c] for c in clf.classes_],
            dtype="float64",
        )
        return self

    def predict_expected_scores(self, deliveries: list[dict]) -> np.ndarray:
        """Return per-delivery expected outcome score in [0, 1]."""
        if self.classifier_ is None or self.class_to_score_ is None:
            raise RuntimeError("ExpectedRunsModel must be fit before prediction.")
        X = self._build_features(deliveries)
        proba = self.classifier_.predict_proba(X)
        return proba @ self.class_to_score_

    def predict_expected_runs(self, deliveries: list[dict]) -> np.ndarray:
        """Return per-delivery expected runs (probability-weighted)."""
        if self.classifier_ is None:
            raise RuntimeError("ExpectedRunsModel must be fit before prediction.")
        runs_lookup = np.array(
            [0 if c == "W" else int(c) for c in self.classifier_.classes_],
            dtype="float64",
        )
        X = self._build_features(deliveries)
        proba = self.classifier_.predict_proba(X)
        return proba @ runs_lookup
