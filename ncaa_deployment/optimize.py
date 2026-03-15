"""
ncaa_deployment.optimize
========================
Stage 10 — Bayesian ensemble weight optimization.

Uses scipy.optimize.differential_evolution (a global optimiser with
Bayesian-like exploration) to search over:

  * Component model weights (softmax-normalised so they sum to 1)
  * Temperature scaling parameter (controls prediction sharpness)
  * Post-hoc calibration method (isotonic / platt / none)

Objective: minimise Brier score on the 2025 holdout set.

Usage
-----
::

    optimizer = BayesianEnsembleOptimizer(models=[lr, weak_lr, constant],
                                          n_trials=200)
    result = optimizer.optimize(X1_tr, X2_tr, y_tr, X1_val, X2_val, y_val)
    print(f"Improved by {result.improvement:.4f}")
    ensemble = optimizer.build_optimized_ensemble()
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.optimize import differential_evolution

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Temperature scaling
# ---------------------------------------------------------------------------

def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    """
    Apply temperature scaling to probabilities.

    Parameters
    ----------
    probs : np.ndarray, shape (n,)
    temperature : float
        > 1.0 → predictions pushed toward 0.5 (conservative).
        < 1.0 → predictions pushed toward 0/1 (aggressive).
        == 1.0 → no change.

    Returns
    -------
    np.ndarray, clipped to [0.01, 0.99].
    """
    probs = np.asarray(probs, dtype=float)
    probs = np.clip(probs, 0.001, 0.999)
    if abs(temperature - 1.0) < 1e-9:
        return np.clip(probs, 0.01, 0.99)
    logits = np.log(probs / (1.0 - probs))
    scaled = 1.0 / (1.0 + np.exp(-logits / temperature))
    return np.clip(scaled, 0.01, 0.99)


# ---------------------------------------------------------------------------
# OptimizationResult
# ---------------------------------------------------------------------------

@dataclass
class OptimizationResult:
    """
    Output of :class:`BayesianEnsembleOptimizer`.

    Attributes
    ----------
    best_weights : np.ndarray
        Normalised ensemble weights (sum=1, all ≥ 0).
    best_brier : float
        Brier score on the validation set for the best configuration found.
    baseline_brier : float
        Brier score of the equal-weight, no-temperature, no-calibration baseline.
    n_trials : int
        Total objective function evaluations performed.
    improvement : float
        ``baseline_brier - best_brier`` (positive = improvement).
    best_temperature : float
    best_calibration : str
        One of ``'isotonic'``, ``'platt'``, ``'none'``.
    history : list of float
        Sequence of objective values during optimisation.
    metadata : dict
    """

    best_weights: np.ndarray
    best_brier: float
    baseline_brier: float
    n_trials: int
    improvement: float
    best_temperature: float = 1.0
    best_calibration: str = "none"
    history: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def improved(self) -> bool:
        """True if optimised Brier is strictly lower than the baseline."""
        return self.best_brier < self.baseline_brier

    def to_dict(self) -> Dict[str, Any]:
        return {
            "best_weights": self.best_weights.tolist(),
            "best_brier": self.best_brier,
            "baseline_brier": self.baseline_brier,
            "n_trials": self.n_trials,
            "improvement": self.improvement,
            "improved": self.improved,
            "best_temperature": self.best_temperature,
            "best_calibration": self.best_calibration,
        }


# ---------------------------------------------------------------------------
# WeightedEnsemble
# ---------------------------------------------------------------------------

class WeightedEnsemble:
    """
    Lightweight ensemble with fixed (pre-computed) weights.

    Parameters
    ----------
    models : list of fitted MatchupPredictor
    weights : array-like, shape (n_models,)
    temperature : float
        Applied at prediction time.
    """

    def __init__(
        self,
        models: List,
        weights: np.ndarray,
        temperature: float = 1.0,
    ) -> None:
        if not models:
            raise ValueError("WeightedEnsemble requires at least one model.")
        self.models = list(models)
        w = np.asarray(weights, dtype=float)
        self.weights = w / w.sum()
        self.temperature = temperature

    def predict_proba(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Weighted average of component predictions, then temperature scaled."""
        preds = np.array([m.predict_proba(X1, X2) for m in self.models])
        combined = np.average(preds, axis=0, weights=self.weights)
        return apply_temperature(combined, self.temperature)

    def fit(self, X1: np.ndarray, X2: np.ndarray, y: np.ndarray) -> "WeightedEnsemble":
        """No-op: models are assumed already fitted."""
        return self

    def __repr__(self) -> str:
        w_str = ", ".join(f"{wi:.3f}" for wi in self.weights)
        return f"WeightedEnsemble(weights=[{w_str}], temperature={self.temperature:.2f})"


# ---------------------------------------------------------------------------
# BayesianEnsembleOptimizer
# ---------------------------------------------------------------------------

class BayesianEnsembleOptimizer:
    """
    Bayesian-style ensemble weight optimiser using
    ``scipy.optimize.differential_evolution``.

    Searches over:
      * Model weights (softmax-parameterised, so they stay on the probability
        simplex)
      * Log temperature (exp-parameterised to keep temperature positive)
      * Post-hoc calibration method (grid search over 3 options)

    Parameters
    ----------
    models : list of MatchupPredictor
        Component models (unfitted).  At least 2 recommended.
    n_trials : int
        Approximate optimisation budget (function evaluations). Default 200.
    seed : int
    """

    CALIBRATION_METHODS = ("none", "isotonic", "platt")

    def __init__(
        self,
        models: List,
        n_trials: int = 200,
        seed: int = 42,
    ) -> None:
        if not models:
            raise ValueError("Need at least one model.")
        self.models = list(models)
        self.n_trials = n_trials
        self.seed = seed
        self._result: Optional[OptimizationResult] = None
        self._fitted_models: Optional[List] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimize(
        self,
        X1_train: np.ndarray,
        X2_train: np.ndarray,
        y_train: np.ndarray,
        X1_val: np.ndarray,
        X2_val: np.ndarray,
        y_val: np.ndarray,
    ) -> OptimizationResult:
        """
        Fit component models, then search for optimal weights.

        Parameters
        ----------
        X1_train, X2_train, y_train : training data
        X1_val, X2_val, y_val : held-out validation data

        Returns
        -------
        OptimizationResult
        """
        n_models = len(self.models)

        # Fit all component models on training data
        fitted_models = []
        for m in self.models:
            mc = copy.deepcopy(m)
            mc.fit(
                np.atleast_2d(X1_train),
                np.atleast_2d(X2_train),
                np.asarray(y_train, dtype=int),
            )
            fitted_models.append(mc)
        self._fitted_models = fitted_models

        # Pre-compute validation predictions once for speed
        val_preds = np.array(
            [m.predict_proba(np.atleast_2d(X1_val), np.atleast_2d(X2_val))
             for m in fitted_models]
        )  # (n_models, n_val)
        y_v = np.asarray(y_val, dtype=float)

        # Baseline: equal weights, temperature=1, no calibration
        eq_w = np.ones(n_models) / n_models
        baseline_brier = float(np.mean((np.average(val_preds, axis=0, weights=eq_w) - y_v) ** 2))
        logger.info("Baseline equal-weight Brier: %.4f", baseline_brier)

        history: List[float] = [baseline_brier]

        # ---------- differential_evolution optimisation ----------
        # Params: [raw_w_0, ..., raw_w_{n-1}, log_temp]
        #   weights = softmax(raw_w)    → on the simplex
        #   temperature = exp(log_temp) → positive
        n_params = n_models + 1

        def _objective(params: np.ndarray) -> float:
            raw_w = params[:n_models]
            log_t = params[n_models]
            # Softmax weights
            w = np.exp(raw_w - raw_w.max())
            w /= w.sum()
            # Temperature in [exp(-2), exp(2)] ≈ [0.14, 7.4]
            t = float(np.exp(np.clip(log_t, -2.0, 2.0)))
            combined = np.average(val_preds, axis=0, weights=w)
            combined = apply_temperature(combined, t)
            val = float(np.mean((combined - y_v) ** 2))
            history.append(val)
            return val

        # Pop size and maxiter: keep total evals ≈ n_trials
        popsize = max(4, n_models + 1)
        maxiter = max(10, self.n_trials // (popsize * n_params))

        bounds = [(-3.0, 3.0)] * n_models + [(-2.0, 2.0)]
        try:
            de_result = differential_evolution(
                _objective,
                bounds,
                maxiter=maxiter,
                popsize=popsize,
                seed=self.seed,
                tol=1e-8,
                mutation=(0.5, 1.5),
                recombination=0.7,
                polish=True,
                init="latinhypercube",
            )
            best_params = de_result.x
        except Exception as exc:
            logger.warning("differential_evolution failed (%s); random search fallback.", exc)
            best_params = np.zeros(n_params)
            best_v = baseline_brier
            rng = np.random.RandomState(self.seed)
            for _ in range(self.n_trials):
                p = rng.uniform(-2, 2, n_params)
                v = _objective(p)
                if v < best_v:
                    best_v = v
                    best_params = p.copy()

        # Decode best parameters
        raw_w = best_params[:n_models]
        best_weights = np.exp(raw_w - raw_w.max())
        best_weights /= best_weights.sum()
        best_temperature = float(np.exp(np.clip(best_params[n_models], -2.0, 2.0)))

        combined_val = apply_temperature(
            np.average(val_preds, axis=0, weights=best_weights),
            best_temperature,
        )
        best_brier = float(np.mean((combined_val - y_v) ** 2))

        # Grid-search calibration methods
        best_calibration = "none"
        for cal in self.CALIBRATION_METHODS[1:]:  # skip "none" (already computed)
            b = self._brier_with_calibration(
                best_weights, fitted_models,
                X1_train, X2_train, y_train,
                X1_val, X2_val, y_val,
                cal,
            )
            if b < best_brier:
                best_brier = b
                best_calibration = cal

        improvement = float(baseline_brier - best_brier)
        logger.info(
            "Optimised Brier: %.4f (baseline %.4f, improvement %.4f)",
            best_brier, baseline_brier, improvement,
        )

        self._result = OptimizationResult(
            best_weights=best_weights,
            best_brier=best_brier,
            baseline_brier=baseline_brier,
            n_trials=len(history),
            improvement=improvement,
            best_temperature=best_temperature,
            best_calibration=best_calibration,
            history=history[:200],
            metadata={
                "n_models": n_models,
                "n_train_samples": int(len(y_train)),
                "n_val_samples": int(len(y_val)),
                "optimizer": "scipy.differential_evolution",
            },
        )
        return self._result

    def result(self) -> Optional[OptimizationResult]:
        """Return the most recent :class:`OptimizationResult`."""
        return self._result

    def build_optimized_ensemble(self) -> WeightedEnsemble:
        """
        Return a :class:`WeightedEnsemble` using the best weights found.

        Must call :meth:`optimize` first.  Uses the fitted model copies.
        """
        if self._result is None or self._fitted_models is None:
            raise RuntimeError("Call optimize() before build_optimized_ensemble().")
        return WeightedEnsemble(
            models=self._fitted_models,
            weights=self._result.best_weights,
            temperature=self._result.best_temperature,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _brier_with_calibration(
        weights: np.ndarray,
        fitted_models: List,
        X1_tr: np.ndarray,
        X2_tr: np.ndarray,
        y_tr: np.ndarray,
        X1_val: np.ndarray,
        X2_val: np.ndarray,
        y_val: np.ndarray,
        method: str,
    ) -> float:
        """Compute val Brier after fitting a post-hoc calibrator on training data."""
        try:
            train_preds_m = np.array([m.predict_proba(np.atleast_2d(X1_tr), np.atleast_2d(X2_tr))
                                      for m in fitted_models])
            val_preds_m = np.array([m.predict_proba(np.atleast_2d(X1_val), np.atleast_2d(X2_val))
                                    for m in fitted_models])
            combined_tr = np.average(train_preds_m, axis=0, weights=weights)
            combined_val = np.average(val_preds_m, axis=0, weights=weights)
            y_v = np.asarray(y_val, dtype=float)
            y_t = np.asarray(y_tr, dtype=float)

            from ncaa_models.calibration import IsotonicCalibrator, PlattScaler
            if method == "isotonic":
                cal = IsotonicCalibrator()
            else:
                cal = PlattScaler()
            cal.fit(combined_tr, y_t)
            calibrated = cal.transform(combined_val)
            return float(np.mean((calibrated - y_v) ** 2))
        except Exception as exc:
            logger.warning("Calibration '%s' failed: %s", method, exc)
            val_preds_m = np.array([m.predict_proba(np.atleast_2d(X1_val), np.atleast_2d(X2_val))
                                    for m in fitted_models])
            combined = np.average(val_preds_m, axis=0, weights=weights)
            return float(np.mean((combined - np.asarray(y_val, dtype=float)) ** 2))
