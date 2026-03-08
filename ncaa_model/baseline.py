"""
ncaa_model.baseline
===================
Logistic regression baseline for NCAA tournament prediction.

Architecture
------------
  Input   : matchup differential feature vector (feat_A - feat_B, 14 dims)
  Pre-proc: ``StandardScaler`` — zero-mean, unit-variance per column
  Model   : ``sklearn.linear_model.LogisticRegression``
             - L2 regularisation (C=1.0)
             - Saga solver (handles large datasets gracefully)
             - max_iter=2000 for convergence

Design notes
------------
* The matchup vector is defined so the model predicts
  P(lower-ID team wins) directly.  Symmetry is guaranteed:
  P(A beats B) = 1 − P(B beats A).

* NaN features are median-imputed per (gender, season) group
  *before* computing the differential, so the imputation is
  calibrated to the distribution of the training season.

* Serialisation: ``save``/``load`` use joblib so the full pipeline
  (scaler + LR) is stored in a single file.

Target performance on real Kaggle data
---------------------------------------
  * Train on 2021-2024 regular-season features + 2021-2024 tourney results
  * Evaluate on 2025 tournament (≈ 63 games)
  * Expected Brier score: 0.17–0.19  (target < 0.20)
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .matchup import (
    DEFAULT_FEATURE_COLS,
    build_training_data,
    impute_features,
    matchup_vector,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model class
# ---------------------------------------------------------------------------

class LogisticBaseline:
    """
    Logistic regression baseline model.

    Implements the ``NCAAPredictor`` protocol defined in ``evaluate.py``.

    Parameters
    ----------
    C : float
        Inverse regularisation strength (smaller = more regularised).
    max_iter : int
        Maximum LR solver iterations.
    feature_cols : list[str]
        Ordered list of feature columns to use (default: DEFAULT_FEATURE_COLS).

    Examples
    --------
    ::

        model = LogisticBaseline()
        model.fit(feat_df, tourney_df, train_seasons=range(2021, 2025))
        p = model.predict_proba_matchup(feat_a, feat_b)  # ∈ [0, 1]
    """

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 2000,
        feature_cols: Optional[List[str]] = None,
    ) -> None:
        self._feature_cols: List[str] = (
            list(feature_cols) if feature_cols is not None else list(DEFAULT_FEATURE_COLS)
        )
        self._pipeline: Optional[Pipeline] = None
        self._is_fitted: bool = False
        self._C = C
        self._max_iter = max_iter
        self._train_seasons: List[int] = []

    # ------------------------------------------------------------------
    # Protocol properties
    # ------------------------------------------------------------------

    @property
    def feature_cols(self) -> List[str]:
        """Ordered list of feature columns used by this model."""
        return self._feature_cols

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        feat_df: pd.DataFrame,
        tourney_df: pd.DataFrame,
        train_seasons: Iterable[int],
    ) -> "LogisticBaseline":
        """
        Fit the logistic regression on historical tournament matchups.

        Parameters
        ----------
        feat_df : pd.DataFrame
            Team-season feature matrix (output of ``compute_features``).
        tourney_df : pd.DataFrame
            Normalised tournament game rows with ``is_tourney=True``.
            Produced by ``normalize_all`` with ``games_df[games_df['is_tourney']]``.
        train_seasons : iterable of int
            Seasons to use for training labels.

        Returns
        -------
        self
        """
        seasons = list(train_seasons)
        X, y, meta = build_training_data(
            feat_df, tourney_df, seasons, self._feature_cols
        )

        self._pipeline = Pipeline([
            # with_mean=False: x_scaled(A,B) = -x_scaled(B,A) — antisymmetry preserved.
            # fit_intercept=False: no bias term, so sigmoid(0)=0.5 for equal teams
            # and P(A beats B) + P(B beats A) = 1 exactly (symmetry guarantee).
            ("scaler", StandardScaler(with_mean=False)),
            ("lr", LogisticRegression(
                C=self._C,
                max_iter=self._max_iter,
                solver="saga",
                fit_intercept=False,
                random_state=42,
            )),
        ])
        self._pipeline.fit(X, y)
        self._is_fitted = True
        self._train_seasons = seasons

        logger.info(
            "LogisticBaseline fitted on %d matchups from seasons %s",
            len(X), seasons,
        )
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_proba_matchup(
        self,
        feat_a: pd.Series,
        feat_b: pd.Series,
    ) -> float:
        """
        Return P(team A wins) given feature vectors for two teams.

        Team A is defined as the **lower raw TeamID** team (Kaggle convention).
        Symmetry is guaranteed by the differential representation:
        ``predict_proba_matchup(a, b) = 1 - predict_proba_matchup(b, a)``.

        Parameters
        ----------
        feat_a, feat_b : pd.Series
            Feature vectors indexed by ``self.feature_cols``.

        Returns
        -------
        float ∈ [0, 1]
        """
        self._assert_fitted()
        x = matchup_vector(feat_a, feat_b, self._feature_cols).reshape(1, -1)
        prob: float = float(self._pipeline.predict_proba(x)[0, 1])
        return prob

    def predict_proba_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Return P(lower-ID team wins) for a batch of pre-built matchup vectors.

        Parameters
        ----------
        X : np.ndarray  shape (n_matchups, n_features)

        Returns
        -------
        np.ndarray  shape (n_matchups,)  values ∈ [0, 1]
        """
        self._assert_fitted()
        return self._pipeline.predict_proba(X)[:, 1]

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Persist model to *path* using pickle."""
        self._assert_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "pipeline": self._pipeline,
            "feature_cols": self._feature_cols,
            "train_seasons": self._train_seasons,
            "C": self._C,
            "max_iter": self._max_iter,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Model saved to %s", path)

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
        obj._train_seasons = payload["train_seasons"]
        logger.info("Model loaded from %s (trained on %s)", path, obj._train_seasons)
        return obj

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _assert_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "Model is not fitted. Call fit() before predict_proba_matchup()."
            )

    def __repr__(self) -> str:
        status = f"fitted on {self._train_seasons}" if self._is_fitted else "unfitted"
        return (
            f"LogisticBaseline(C={self._C}, n_features={len(self._feature_cols)}, "
            f"status={status})"
        )
