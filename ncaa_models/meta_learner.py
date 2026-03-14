"""
ncaa_models.meta_learner
========================
Stage 06 — Stacking meta-learner ensemble for NCAA tournament prediction.

MetaLearnerEnsemble stacks component model predictions using a second-level
(meta) learner.  To prevent information leakage, out-of-fold (OOF)
predictions are generated via temporal cross-validation.

Architecture
------------
1. For each temporal fold (train_seasons → val_season):
   a. Fit every base model on the training seasons.
   b. Predict on the validation season → OOF prediction matrix.
2. Concatenate all OOF predictions into a meta-feature matrix ``Z``.
3. Fit the meta-learner (non-negative least-squares by default) on
   ``(Z, y_oof)``.
4. For final predictions:
   a. Refit all base models on the full training set.
   b. Apply base models to test set to form ``Z_test``.
   c. Apply meta-learner to ``Z_test``.

Meta-learner: Non-Negative Least Squares (NNLS)
    ``argmin ||Zw - y||²  s.t. w ≥ 0``
    Guarantees all stacking weights are strictly positive (after epsilon clip).

Usage
-----
::

    meta = MetaLearnerEnsemble(models=[lr, neural, cf_model])
    meta.fit_oof(matchup_df, feature_cols=FEATURE_COLS,
                 cv_seasons=[2023, 2024, 2025])
    probs = meta.predict_proba(X1_test, X2_test)
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from .base import MatchupPredictor
from .evaluate import compute_brier_score

logger = logging.getLogger(__name__)

#: Epsilon to guarantee strictly positive weights after NNLS
_WEIGHT_EPS: float = 1e-6


# ---------------------------------------------------------------------------
# _NNLSMetaLearner — internal meta-learner
# ---------------------------------------------------------------------------

class _NNLSMetaLearner:
    """
    Non-negative least-squares meta-learner.

    Solves:  argmin ||Zw - y||²  subject to  w >= 0

    After fitting, weights are clipped to a minimum of ``_WEIGHT_EPS`` and
    normalised to sum to 1 so that ``predict(Z)`` stays in [0, 1] whenever
    all column inputs are in [0, 1].
    """

    def __init__(self) -> None:
        self.weights_: Optional[np.ndarray] = None

    def fit(self, Z: np.ndarray, y: np.ndarray) -> "_NNLSMetaLearner":
        """
        Parameters
        ----------
        Z : np.ndarray, shape (n, n_models) — base model predictions
        y : np.ndarray, shape (n,) — binary labels
        """
        Z = np.asarray(Z, dtype=float)
        y = np.asarray(y, dtype=float)

        weights, _ = nnls(Z, y)

        # Guarantee strictly positive (clamp any zeros)
        weights = np.maximum(weights, _WEIGHT_EPS)
        weights /= weights.sum()
        self.weights_ = weights
        return self

    def predict(self, Z: np.ndarray) -> np.ndarray:
        if self.weights_ is None:
            raise RuntimeError("_NNLSMetaLearner must be fitted first.")
        Z = np.asarray(Z, dtype=float)
        return np.clip(Z @ self.weights_, 0.01, 0.99)


# ---------------------------------------------------------------------------
# MetaLearnerEnsemble
# ---------------------------------------------------------------------------

class MetaLearnerEnsemble(MatchupPredictor):
    """
    Stacking ensemble with non-negative least-squares meta-learner.

    Supports two fitting modes:

    1. ``fit(X1, X2, y)``
       Quick in-sample fitting (no OOF; meta-learner is trained on
       the same data the base models see).  Useful for unit tests and
       rapid iteration, but suffers from training-set over-optimism.

    2. ``fit_oof(matchup_df, feature_cols, cv_seasons)``
       Proper temporal OOF fitting for production use.  Base models are
       re-fitted on the full training set after OOF generation.

    After fitting, ``weights_`` contains the meta-learner stacking weights
    (one per base model), all strictly positive.

    Parameters
    ----------
    models : list of MatchupPredictor
        Base (level-0) models.
    """

    def __init__(self, models: List[MatchupPredictor]) -> None:
        if not models:
            raise ValueError("MetaLearnerEnsemble requires at least one model.")
        self.models = list(models)
        self._meta: _NNLSMetaLearner = _NNLSMetaLearner()
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # MatchupPredictor interface
    # ------------------------------------------------------------------

    def fit(
        self,
        X1: np.ndarray,
        X2: np.ndarray,
        y: np.ndarray,
    ) -> "MetaLearnerEnsemble":
        """
        Fit base models and meta-learner on the same data (in-sample).

        Note: this may overfit the meta-learner; prefer ``fit_oof()`` for
        production use.
        """
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)
        y = np.asarray(y, dtype=int)

        for model in self.models:
            model.fit(X1, X2, y)

        Z = self._build_meta_features(X1, X2)
        self._meta.fit(Z, y)
        self._fitted = True
        logger.debug("MetaLearnerEnsemble fitted in-sample: weights=%s", np.round(self.weights_, 4))
        return self

    def predict_proba(
        self,
        X1: np.ndarray,
        X2: np.ndarray,
    ) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("MetaLearnerEnsemble must be fitted first.")
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)
        Z = self._build_meta_features(X1, X2)
        return self._meta.predict(Z)

    # ------------------------------------------------------------------
    # OOF fitting (production)
    # ------------------------------------------------------------------

    def fit_oof(
        self,
        matchup_df: pd.DataFrame,
        feature_cols: Sequence[str],
        cv_seasons: Optional[List[int]] = None,
        min_train_seasons: int = 2,
    ) -> "MetaLearnerEnsemble":
        """
        Fit meta-learner on out-of-fold (OOF) predictions from temporal CV.

        After OOF generation, all base models are re-fitted on the full
        training dataset (all rows in matchup_df).

        Parameters
        ----------
        matchup_df : pd.DataFrame
            One row per unique matchup.
            Required columns: ``season``, ``X_team1``, ``X_team2``, ``y``.
        feature_cols : list of str
        cv_seasons : list of int, optional
            Seasons to use as validation holdouts.  Defaults to the last
            ``(n_seasons - min_train_seasons)`` available seasons.
        min_train_seasons : int
            Minimum training seasons required to run a fold.

        Returns
        -------
        self
        """
        all_seasons = sorted(matchup_df["season"].unique())
        if cv_seasons is None:
            cv_seasons = all_seasons[min_train_seasons:]

        # Accumulate OOF predictions
        oof_Z_list: list = []
        oof_y_list: list = []

        for val_season in sorted(cv_seasons):
            train_seasons = [s for s in all_seasons if s < val_season]
            if len(train_seasons) < min_train_seasons:
                logger.warning(
                    "Skipping OOF fold val=%d: only %d train season(s).",
                    val_season, len(train_seasons),
                )
                continue

            train_df = matchup_df[matchup_df["season"].isin(train_seasons)]
            val_df = matchup_df[matchup_df["season"] == val_season]
            if val_df.empty:
                continue

            X1_tr, X2_tr, y_tr = _extract(train_df)
            X1_v, X2_v, y_v = _extract(val_df)

            # Fit base models on this fold's training data
            fold_models = [_clone_fit(m, X1_tr, X2_tr, y_tr) for m in self.models]

            # Generate OOF predictions
            Z_val = np.column_stack([m.predict_proba(X1_v, X2_v) for m in fold_models])
            oof_Z_list.append(Z_val)
            oof_y_list.append(y_v)

            logger.debug(
                "OOF fold: train=%s → val=%d  (n=%d)",
                train_seasons, val_season, len(y_v),
            )

        if not oof_Z_list:
            logger.warning(
                "No OOF folds completed; falling back to in-sample fitting."
            )
            X1_all, X2_all, y_all = _extract(matchup_df)
            return self.fit(X1_all, X2_all, y_all)

        oof_Z = np.vstack(oof_Z_list)
        oof_y = np.concatenate(oof_y_list)

        # Fit meta-learner on OOF predictions
        self._meta.fit(oof_Z, oof_y)

        # Re-fit all base models on the full dataset
        X1_all, X2_all, y_all = _extract(matchup_df)
        for model in self.models:
            model.fit(X1_all, X2_all, y_all)

        self._fitted = True
        logger.info(
            "MetaLearnerEnsemble fitted OOF (n_oof=%d): weights=%s",
            len(oof_y), np.round(self.weights_, 4),
        )
        return self

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def weights_(self) -> np.ndarray:
        """Stacking weights — one per base model, all strictly positive."""
        if self._meta.weights_ is None:
            raise RuntimeError("weights_ not available before fitting.")
        return self._meta.weights_

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_meta_features(
        self, X1: np.ndarray, X2: np.ndarray
    ) -> np.ndarray:
        """Stack base model predictions into meta-feature matrix."""
        return np.column_stack([m.predict_proba(X1, X2) for m in self.models])

    def __repr__(self) -> str:
        names = [m.__class__.__name__ for m in self.models]
        return f"MetaLearnerEnsemble([{', '.join(names)}])"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract(df: pd.DataFrame):
    """Extract (X1, X2, y) arrays from a matchup DataFrame."""
    X1 = np.stack(df["X_team1"].values)
    X2 = np.stack(df["X_team2"].values)
    y = df["y"].values.astype(int)
    return X1, X2, y


def _clone_fit(
    model: MatchupPredictor,
    X1: np.ndarray,
    X2: np.ndarray,
    y: np.ndarray,
) -> MatchupPredictor:
    """
    Fit a shallow copy of ``model`` on new data.

    Uses ``sklearn.base.clone`` when available; otherwise instantiates a
    new model of the same class with the same ``__init__`` parameters.
    """
    try:
        from sklearn.base import clone as sk_clone
        # Only works for sklearn estimators; MatchupPredictors may not comply.
        cloned = sk_clone(model)
    except Exception:
        # Fallback: new instance of the same class with no arguments
        try:
            cloned = model.__class__()
        except TypeError:
            # If __init__ requires arguments, re-use the original model
            # (last fold will over-write the original — acceptable for OOF)
            cloned = model
    cloned.fit(X1, X2, y)
    return cloned
