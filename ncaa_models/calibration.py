"""
ncaa_models.calibration
=======================
Stage 06 — Post-hoc probability calibration for NCAA tournament predictions.

Two calibration methods are provided:

PlattScaler
    Logistic regression fitted on raw model probabilities → actual outcomes.
    Parameters: a single sigmoid mapping  P_cal = σ(a·p + b).

IsotonicCalibrator
    Non-parametric monotone calibration via isotonic regression.
    More flexible than Platt but requires more data.

PostHocCalibrator
    MatchupPredictor wrapper that fits a base model and then calibrates its
    output on a held-out fold (or an explicitly supplied calibration set).

compare_calibration
    Utility that fits both methods and returns ECE for each.

Usage
-----
::

    cal = PostHocCalibrator(base_model=LogisticBaseline(), method='platt')
    cal.fit(X1_train, X2_train, y_train,
            X1_cal=X1_val, X2_cal=X2_val, y_cal=y_val)
    probs = cal.predict_proba(X1_test, X2_test)
"""

from __future__ import annotations

import logging
from typing import Dict, Literal, Optional, Tuple

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from .base import MatchupPredictor
from .evaluate import calibration_report

logger = logging.getLogger(__name__)

#: Hard clip range applied to all calibrated probabilities.
PROB_CLIP_LOW: float = 0.01
PROB_CLIP_HIGH: float = 0.99

CalibrationMethod = Literal["platt", "isotonic"]


# ---------------------------------------------------------------------------
# PlattScaler
# ---------------------------------------------------------------------------

class PlattScaler:
    """
    Platt scaling — logistic regression on raw probabilities.

    Maps raw probabilities through a sigmoid:  P_cal = σ(a·p + b)
    where (a, b) are fitted by logistic regression.

    Parameters
    ----------
    C : float
        Inverse regularization strength for the logistic regression.
    """

    def __init__(self, C: float = 1.0) -> None:
        self.C = C
        self._lr: Optional[LogisticRegression] = None

    def fit(self, raw_probs: np.ndarray, y: np.ndarray) -> "PlattScaler":
        """
        Fit on raw (uncalibrated) probabilities and binary labels.

        Parameters
        ----------
        raw_probs : np.ndarray, shape (n,)
        y : np.ndarray, shape (n,), dtype int in {0, 1}
        """
        raw = np.asarray(raw_probs, dtype=float).reshape(-1, 1)
        y_arr = np.asarray(y, dtype=int)
        self._lr = LogisticRegression(C=self.C, solver="lbfgs", max_iter=300).fit(
            raw, y_arr
        )
        return self

    def transform(self, raw_probs: np.ndarray) -> np.ndarray:
        """
        Apply Platt scaling to raw probabilities.

        Returns
        -------
        np.ndarray, shape (n,), clipped to [0.01, 0.99]
        """
        if self._lr is None:
            raise RuntimeError("PlattScaler must be fitted before transform().")
        raw = np.asarray(raw_probs, dtype=float).reshape(-1, 1)
        probs = self._lr.predict_proba(raw)[:, 1]
        return np.clip(probs, PROB_CLIP_LOW, PROB_CLIP_HIGH)

    def fit_transform(
        self, raw_probs: np.ndarray, y: np.ndarray
    ) -> Tuple["PlattScaler", np.ndarray]:
        """Convenience: fit and return calibrated probabilities."""
        self.fit(raw_probs, y)
        return self, self.transform(raw_probs)


# ---------------------------------------------------------------------------
# IsotonicCalibrator
# ---------------------------------------------------------------------------

class IsotonicCalibrator:
    """
    Isotonic regression calibrator.

    Fits a piecewise-constant, non-decreasing function from raw
    probabilities to actual win rates.  More flexible than Platt but
    can over-fit on small datasets.
    """

    def __init__(self) -> None:
        self._ir: Optional[IsotonicRegression] = None

    def fit(self, raw_probs: np.ndarray, y: np.ndarray) -> "IsotonicCalibrator":
        raw = np.asarray(raw_probs, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        self._ir = IsotonicRegression(out_of_bounds="clip").fit(raw, y_arr)
        return self

    def transform(self, raw_probs: np.ndarray) -> np.ndarray:
        if self._ir is None:
            raise RuntimeError("IsotonicCalibrator must be fitted before transform().")
        raw = np.asarray(raw_probs, dtype=float)
        probs = self._ir.predict(raw)
        return np.clip(probs.astype(float), PROB_CLIP_LOW, PROB_CLIP_HIGH)

    def fit_transform(
        self, raw_probs: np.ndarray, y: np.ndarray
    ) -> Tuple["IsotonicCalibrator", np.ndarray]:
        self.fit(raw_probs, y)
        return self, self.transform(raw_probs)


# ---------------------------------------------------------------------------
# PostHocCalibrator — MatchupPredictor wrapper
# ---------------------------------------------------------------------------

class PostHocCalibrator(MatchupPredictor):
    """
    Wraps any MatchupPredictor with post-hoc probability calibration.

    The calibrator is fitted either on an explicit calibration set or,
    if none is provided, on the last ``val_fraction`` of the training data.

    Parameters
    ----------
    base_model : MatchupPredictor
        The underlying prediction model.
    method : {'platt', 'isotonic'}
        Calibration method.  Default ``'platt'``.
    val_fraction : float
        Fraction of training data to hold out for calibration fitting
        when no explicit calibration set is provided.  Default 0.2.
    """

    def __init__(
        self,
        base_model: MatchupPredictor,
        method: CalibrationMethod = "platt",
        val_fraction: float = 0.2,
    ) -> None:
        self.base_model = base_model
        self.method = method
        self.val_fraction = val_fraction
        self.calibrator_: Optional[PlattScaler | IsotonicCalibrator] = None

    def fit(
        self,
        X1: np.ndarray,
        X2: np.ndarray,
        y: np.ndarray,
        X1_cal: Optional[np.ndarray] = None,
        X2_cal: Optional[np.ndarray] = None,
        y_cal: Optional[np.ndarray] = None,
    ) -> "PostHocCalibrator":
        """
        Fit base model and calibrator.

        If ``X1_cal`` is provided, the base model is trained on ``(X1, X2, y)``
        and calibrated on ``(X1_cal, X2_cal, y_cal)``.

        If not provided, the last ``val_fraction`` of the training rows are
        withheld for calibration.
        """
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)
        y = np.asarray(y, dtype=int)

        if X1_cal is not None:
            # Explicit calibration set
            self.base_model.fit(X1, X2, y)
            X1_c = np.atleast_2d(X1_cal)
            X2_c = np.atleast_2d(X2_cal)
            y_c = np.asarray(y_cal, dtype=int)
        else:
            # Split training data
            n = len(y)
            n_cal = max(1, int(n * self.val_fraction))
            n_tr = n - n_cal
            if n_tr < 1:
                n_tr, n_cal = n, n  # degenerate: calibrate on full training set
            self.base_model.fit(X1[:n_tr], X2[:n_tr], y[:n_tr])
            X1_c, X2_c, y_c = X1[n_tr:], X2[n_tr:], y[n_tr:]

        raw_cal = self.base_model.predict_proba(X1_c, X2_c)

        if self.method == "platt":
            self.calibrator_ = PlattScaler().fit(raw_cal, y_c)
        elif self.method == "isotonic":
            self.calibrator_ = IsotonicCalibrator().fit(raw_cal, y_c)
        else:
            raise ValueError(
                f"Unknown calibration method '{self.method}'. "
                "Choose 'platt' or 'isotonic'."
            )

        logger.debug(
            "PostHocCalibrator(%s) fitted: cal_n=%d.", self.method, len(y_c)
        )
        return self

    def predict_proba(
        self, X1: np.ndarray, X2: np.ndarray
    ) -> np.ndarray:
        """Return calibrated P(team1 wins)."""
        if self.calibrator_ is None:
            raise RuntimeError("PostHocCalibrator must be fitted before predict_proba().")
        raw = self.base_model.predict_proba(np.atleast_2d(X1), np.atleast_2d(X2))
        return self.calibrator_.transform(raw)

    def __repr__(self) -> str:
        return (
            f"PostHocCalibrator(method={self.method!r}, "
            f"base={self.base_model.__class__.__name__})"
        )


# ---------------------------------------------------------------------------
# Utility: compare calibration methods
# ---------------------------------------------------------------------------

def compare_calibration(
    raw_probs: np.ndarray,
    y: np.ndarray,
    n_bins: int = 5,
) -> Dict:
    """
    Compare Platt scaling, isotonic regression, and uncalibrated predictions.

    All three are evaluated on the same data (in-sample).  Use this as a
    diagnostic after fitting calibrators on a held-out fold.

    Parameters
    ----------
    raw_probs : np.ndarray, shape (n,)
        Raw (uncalibrated) model probabilities.
    y : np.ndarray, shape (n,)
        Binary labels.
    n_bins : int
        Number of calibration bins for the report.  Default 5.

    Returns
    -------
    dict with keys ``'raw'``, ``'platt'``, ``'isotonic'``,
    each containing ``{'ece': float, 'brier': float}``.
    """
    raw = np.asarray(raw_probs, dtype=float)
    y_arr = np.asarray(y, dtype=int)

    def _ece_brier(probs: np.ndarray) -> Dict:
        report = calibration_report(probs, y_arr, n_bins=n_bins)
        return {"ece": report["ece"], "brier": report["brier"]}

    result = {"raw": _ece_brier(raw)}

    if len(y_arr) >= 2 and len(np.unique(y_arr)) == 2:
        _, platt_probs = PlattScaler().fit_transform(raw, y_arr)
        result["platt"] = _ece_brier(platt_probs)

        _, iso_probs = IsotonicCalibrator().fit_transform(raw, y_arr)
        result["isotonic"] = _ece_brier(iso_probs)
    else:
        # Not enough diversity to fit calibrators
        result["platt"] = result["raw"].copy()
        result["isotonic"] = result["raw"].copy()

    return result
