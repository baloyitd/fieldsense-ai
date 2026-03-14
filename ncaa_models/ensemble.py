"""
ncaa_models.ensemble
====================
Stage 06 — Ensemble stacking of component NCAA tournament predictors.

Four aggregation strategies are provided:

SimpleAverageEnsemble
    Arithmetic mean of all component model predictions.  Equal weights.

BrierWeightedEnsemble
    Weights inversely proportional to each model's Brier score on a
    validation (or training) set.  Better models receive more weight.
    Weights are stored in ``weights_`` after fitting.

ContextualEnsemble
    Per-matchup context-dependent weighting driven by the
    :class:`~ncaa_models.upset_detector.UpsetDetector`:

    - ``blowout_likely``   → lean on conservative models (LR + baseline)
    - ``competitive``      → blend all models evenly
    - ``upset_plausible``  → lean on counterfactual-aware models

MetaLearnerEnsemble
    Logistic-regression stacking using out-of-fold (OOF) predictions to
    avoid information leakage.  See :mod:`ncaa_models.meta_learner`.

Usage
-----
::

    ensemble = BrierWeightedEnsemble([lr_model, neural_model, cf_model])
    ensemble.fit(X1_train, X2_train, y_train,
                 X1_val=X1_val, X2_val=X2_val, y_val=y_val)
    probs = ensemble.predict_proba(X1_test, X2_test)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from .base import MatchupPredictor
from .evaluate import compute_brier_score
from .upset_detector import BLOWOUT_LIKELY, COMPETITIVE, UPSET_PLAUSIBLE

logger = logging.getLogger(__name__)

#: Small epsilon to prevent zero weights
_WEIGHT_EPS: float = 1e-6


# ---------------------------------------------------------------------------
# SimpleAverageEnsemble
# ---------------------------------------------------------------------------

class SimpleAverageEnsemble(MatchupPredictor):
    """
    Ensemble with equal (simple average) weights.

    Parameters
    ----------
    models : list of MatchupPredictor
        Component models.  At least one required.
    """

    def __init__(self, models: List[MatchupPredictor]) -> None:
        if not models:
            raise ValueError("SimpleAverageEnsemble requires at least one model.")
        self.models = list(models)

    def fit(
        self,
        X1: np.ndarray,
        X2: np.ndarray,
        y: np.ndarray,
    ) -> "SimpleAverageEnsemble":
        """Fit all component models on the same training data."""
        for model in self.models:
            model.fit(np.atleast_2d(X1), np.atleast_2d(X2), np.asarray(y))
        return self

    def predict_proba(
        self,
        X1: np.ndarray,
        X2: np.ndarray,
    ) -> np.ndarray:
        """Return the arithmetic mean of component model predictions."""
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)
        preds = np.array([m.predict_proba(X1, X2) for m in self.models])
        return np.mean(preds, axis=0)

    def __repr__(self) -> str:
        names = [m.__class__.__name__ for m in self.models]
        return f"SimpleAverageEnsemble([{', '.join(names)}])"


# ---------------------------------------------------------------------------
# BrierWeightedEnsemble
# ---------------------------------------------------------------------------

class BrierWeightedEnsemble(MatchupPredictor):
    """
    Ensemble weighted by inverse Brier score.

    Each model's weight is proportional to ``1 / brier(model, val_data)``.
    Models with lower Brier receive higher weight.

    When ``X1_val`` is provided to ``fit()``, validation Brier is used.
    Otherwise, in-sample (training) Brier is used as an approximation.

    Parameters
    ----------
    models : list of MatchupPredictor
    """

    def __init__(self, models: List[MatchupPredictor]) -> None:
        if not models:
            raise ValueError("BrierWeightedEnsemble requires at least one model.")
        self.models = list(models)
        self.weights_: Optional[np.ndarray] = None

    def fit(
        self,
        X1: np.ndarray,
        X2: np.ndarray,
        y: np.ndarray,
        X1_val: Optional[np.ndarray] = None,
        X2_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "BrierWeightedEnsemble":
        """
        Fit all component models, then compute inverse-Brier weights.

        Parameters
        ----------
        X1, X2, y : training data
        X1_val, X2_val, y_val : optional validation data for weight computation.
            If not provided, training data Brier is used (may overfit).
        """
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)
        y = np.asarray(y, dtype=int)

        for model in self.models:
            model.fit(X1, X2, y)

        # Choose evaluation set
        if X1_val is not None:
            eval_X1 = np.atleast_2d(X1_val)
            eval_X2 = np.atleast_2d(X2_val)
            eval_y = np.asarray(y_val, dtype=int)
        else:
            eval_X1, eval_X2, eval_y = X1, X2, y

        briers = []
        for model in self.models:
            preds = model.predict_proba(eval_X1, eval_X2)
            b = compute_brier_score(preds, eval_y)
            briers.append(max(b, _WEIGHT_EPS))

        inv_briers = np.array([1.0 / b for b in briers])
        self.weights_ = inv_briers / inv_briers.sum()

        logger.debug(
            "BrierWeightedEnsemble weights: %s (briers: %s)",
            np.round(self.weights_, 4),
            np.round(briers, 4),
        )
        return self

    def predict_proba(
        self,
        X1: np.ndarray,
        X2: np.ndarray,
    ) -> np.ndarray:
        if self.weights_ is None:
            raise RuntimeError("BrierWeightedEnsemble must be fitted first.")
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)
        preds = np.array([m.predict_proba(X1, X2) for m in self.models])
        return np.average(preds, axis=0, weights=self.weights_)

    def __repr__(self) -> str:
        w = (
            "[" + ", ".join(f"{wi:.3f}" for wi in self.weights_) + "]"
            if self.weights_ is not None
            else "None"
        )
        return f"BrierWeightedEnsemble(weights={w})"


# ---------------------------------------------------------------------------
# ContextualEnsemble
# ---------------------------------------------------------------------------

# Default weight profile keys
_CTX_KEYS = (BLOWOUT_LIKELY, COMPETITIVE, UPSET_PLAUSIBLE)


def _build_default_profiles(n_models: int) -> Dict[str, np.ndarray]:
    """
    Return default weight profiles for each matchup context.

    Default philosophy:
    - blowout  : up-weight first model (conservative), down-weight last
    - competitive : equal weights (all models contribute)
    - upset    : up-weight last model (exploratory), down-weight first

    If n_models == 1, all profiles are uniform.
    """
    if n_models == 1:
        return {k: np.array([1.0]) for k in _CTX_KEYS}

    # Linear ramp profiles
    blowout_raw = np.linspace(2.0, 1.0, n_models)
    upset_raw = np.linspace(1.0, 2.0, n_models)
    equal_raw = np.ones(n_models)

    def _norm(w: np.ndarray) -> np.ndarray:
        return w / w.sum()

    return {
        BLOWOUT_LIKELY: _norm(blowout_raw),
        COMPETITIVE: _norm(equal_raw),
        UPSET_PLAUSIBLE: _norm(upset_raw),
    }


class ContextualEnsemble(MatchupPredictor):
    """
    Context-dependent ensemble using :class:`~ncaa_models.upset_detector.UpsetDetector`.

    Each matchup is classified as blowout / competitive / upset-plausible and
    a corresponding weight profile selects how to combine component models.

    Parameters
    ----------
    models : list of MatchupPredictor
    upset_detector : UpsetDetector
        Used to classify each matchup at prediction time.
    weight_profiles : dict, optional
        Maps context label → weight array (length = n_models).
        Weights need not be normalised; they will be normalised internally.
        If not provided, default profiles are used (see ``_build_default_profiles``).
    """

    def __init__(
        self,
        models: List[MatchupPredictor],
        upset_detector,
        weight_profiles: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        if not models:
            raise ValueError("ContextualEnsemble requires at least one model.")
        self.models = list(models)
        self.upset_detector = upset_detector
        n = len(models)

        if weight_profiles is not None:
            # Normalise supplied profiles
            self.weight_profiles = {
                k: np.asarray(v, dtype=float) / np.asarray(v, dtype=float).sum()
                for k, v in weight_profiles.items()
            }
        else:
            self.weight_profiles = _build_default_profiles(n)

        # Ensure all three keys exist
        for k in _CTX_KEYS:
            if k not in self.weight_profiles:
                self.weight_profiles[k] = np.ones(n) / n

    def fit(
        self,
        X1: np.ndarray,
        X2: np.ndarray,
        y: np.ndarray,
    ) -> "ContextualEnsemble":
        """Fit all component models."""
        for model in self.models:
            model.fit(np.atleast_2d(X1), np.atleast_2d(X2), np.asarray(y))
        return self

    def predict_proba(
        self,
        X1: np.ndarray,
        X2: np.ndarray,
    ) -> np.ndarray:
        """
        For each matchup, classify context, then return context-weighted average.
        """
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)
        n = len(X1)
        out = np.empty(n, dtype=float)

        for i in range(n):
            x1, x2 = X1[i], X2[i]
            # Classify context
            result = self.upset_detector.analyze(x1, x2)
            ctx = result["classification"]
            weights = self.weight_profiles.get(ctx, np.ones(len(self.models)) / len(self.models))

            # Weighted average of component predictions
            preds = np.array([
                m.predict_proba(x1.reshape(1, -1), x2.reshape(1, -1))[0]
                for m in self.models
            ])
            out[i] = float(np.dot(weights, preds))

        return out

    def __repr__(self) -> str:
        return (
            f"ContextualEnsemble("
            f"n_models={len(self.models)}, "
            f"detector={self.upset_detector.__class__.__name__})"
        )


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def build_simple_ensemble(
    models: List[MatchupPredictor],
    X1_train: np.ndarray,
    X2_train: np.ndarray,
    y_train: np.ndarray,
) -> SimpleAverageEnsemble:
    """Fit and return a SimpleAverageEnsemble."""
    ens = SimpleAverageEnsemble(models)
    ens.fit(X1_train, X2_train, y_train)
    return ens


def build_weighted_ensemble(
    models: List[MatchupPredictor],
    X1_train: np.ndarray,
    X2_train: np.ndarray,
    y_train: np.ndarray,
    X1_val: Optional[np.ndarray] = None,
    X2_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
) -> BrierWeightedEnsemble:
    """Fit and return a BrierWeightedEnsemble."""
    ens = BrierWeightedEnsemble(models)
    ens.fit(X1_train, X2_train, y_train, X1_val, X2_val, y_val)
    return ens
