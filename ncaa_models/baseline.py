"""
ncaa_models.baseline
====================
Logistic regression baseline for NCAA tournament prediction.

Architecture
------------
Input   : two raw team feature vectors (n_features each)
Diff    : diff = X_team1 - X_team2  (antisymmetric representation)
Scaler  : StandardScaler(with_mean=False) — preserves antisymmetry
Model   : LogisticRegression(fit_intercept=False, solver='saga')
Clip    : output probabilities clipped to [0.01, 0.99]

Symmetry guarantee
------------------
With with_mean=False and fit_intercept=False:
    P(A beats B) = sigmoid(w · diff(A,B))
    P(B beats A) = sigmoid(w · diff(B,A)) = sigmoid(-w · diff(A,B))
    P(A beats B) + P(B beats A) = 1  exactly.

Kaggle convention
-----------------
team1 = lower TeamId, team2 = higher TeamId.
Submission predicts P(lower-ID team wins).
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .base import MatchupPredictor

logger = logging.getLogger(__name__)

# All 28 feature columns produced by the Stage 01 ncaa_data pipeline.
# Order must match the columns in feat_df when building prediction arrays.
FEATURE_COLS: List[str] = [
    "games_played",
    "win_pct",
    "ppg_scored",
    "ppg_allowed",
    "scoring_margin",
    "home_win_pct",
    "away_win_pct",
    "neutral_win_pct",
    "win_pct_last10",
    "fg_pct",
    "fg3_pct",
    "ft_pct",
    "efg_pct",
    "ts_pct",
    "orb_rate",
    "drb_rate",
    "possessions_pg",
    "ortg",
    "drtg",
    "net_rtg",
    "tempo",
    "tov_rate",
    "ast_rate",
    "blk_rate",
    "stl_rate",
    "ot_rate",
    "sos",
    "seed",
]

# Probability clipping range — avoids extreme Brier/log-loss penalties.
PROB_CLIP_LOW: float = 0.01
PROB_CLIP_HIGH: float = 0.99


class LogisticBaseline(MatchupPredictor):
    """
    Logistic regression baseline implementing the MatchupPredictor interface.

    Parameters
    ----------
    C : float
        Inverse regularisation strength (sklearn convention). Smaller = more regularised.
    max_iter : int
        Maximum solver iterations.
    feature_cols : list[str] or None
        Feature columns to expect in input arrays (ordered).
        Defaults to the full 28-column FEATURE_COLS list.

    Examples
    --------
    ::

        model = LogisticBaseline()
        model.fit(X_team1_train, X_team2_train, y_train)
        probs = model.predict_proba(X_team1_test, X_team2_test)  # ∈ [0.01, 0.99]
    """

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 2000,
        feature_cols: Optional[List[str]] = None,
    ) -> None:
        self.feature_cols: List[str] = (
            list(feature_cols) if feature_cols is not None else list(FEATURE_COLS)
        )
        self._C = C
        self._max_iter = max_iter
        self._pipeline: Optional[Pipeline] = None
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # MatchupPredictor interface
    # ------------------------------------------------------------------

    def fit(
        self,
        X_team1: np.ndarray,
        X_team2: np.ndarray,
        y: np.ndarray,
    ) -> None:
        """
        Train on matchup feature arrays.

        Parameters
        ----------
        X_team1, X_team2 : np.ndarray, shape (n_matchups, n_features)
            Raw per-team feature vectors. team1 = lower TeamId.
        y : np.ndarray, shape (n_matchups,)
            y=1 if team1 wins, y=0 if team2 wins.
        """
        X_diff = X_team1 - X_team2  # antisymmetric differential representation

        self._pipeline = Pipeline([
            # with_mean=False keeps x_scaled(A,B) = -x_scaled(B,A) — antisymmetry.
            ("scaler", StandardScaler(with_mean=False)),
            ("lr", LogisticRegression(
                C=self._C,
                max_iter=self._max_iter,
                solver="saga",
                # fit_intercept=False ensures sigmoid(0)=0.5 for equal teams
                # and P(A,B) + P(B,A) = 1 exactly.
                fit_intercept=False,
                random_state=42,
            )),
        ])
        self._pipeline.fit(X_diff, y)
        self._is_fitted = True
        logger.info(
            "LogisticBaseline fitted on %d matchups (%d features).",
            len(y), X_diff.shape[1],
        )

    def predict_proba(
        self,
        X_team1: np.ndarray,
        X_team2: np.ndarray,
    ) -> np.ndarray:
        """
        Return P(team1 wins) clipped to [0.01, 0.99].

        Parameters
        ----------
        X_team1, X_team2 : np.ndarray, shape (n_matchups, n_features)

        Returns
        -------
        np.ndarray, shape (n_matchups,), values in [0.01, 0.99].
        """
        self._check_fitted()
        X_diff = X_team1 - X_team2
        raw_probs = self._pipeline.predict_proba(X_diff)[:, 1]
        return np.clip(raw_probs, PROB_CLIP_LOW, PROB_CLIP_HIGH)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Persist model to *path* using pickle."""
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "pipeline": self._pipeline,
            "feature_cols": self.feature_cols,
            "C": self._C,
            "max_iter": self._max_iter,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("LogisticBaseline saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "LogisticBaseline":
        """Load a model previously saved with ``save``."""
        with open(path, "rb") as f:
            payload = pickle.load(f)
        obj = cls(
            C=payload["C"],
            max_iter=payload["max_iter"],
            feature_cols=payload["feature_cols"],
        )
        obj._pipeline = payload["pipeline"]
        obj._is_fitted = True
        logger.info("LogisticBaseline loaded from %s", path)
        return obj

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "Model is not fitted. Call fit() before predict_proba()."
            )

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "unfitted"
        return (
            f"LogisticBaseline(C={self._C}, "
            f"n_features={len(self.feature_cols)}, {status})"
        )
