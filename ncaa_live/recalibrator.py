"""
ncaa_live.recalibrator
======================
Stage 08 — Per-round model recalibration on live tournament outcomes.

RoundRecalibrator retrains (or fine-tunes) the prediction model after each
tournament round using the completed game results as additional training
signal.  It wraps the Stage 04 LoRA calibration pipeline when a neural
model is present, and falls back to full LogisticBaseline retraining
otherwise.

Usage
-----
::

    recal = RoundRecalibrator(base_model=fitted_logistic_model)
    recal.set_baseline(base_model, X1_val, X2_val, y_val, round_num=0)

    version = recal.recalibrate(
        results_X1=X1_tourney,
        results_X2=X2_tourney,
        results_y=y_tourney,
        X1_val=X1_val,
        X2_val=X2_val,
        y_val=y_val,
        X1_hist=X1_train,
        X2_hist=X2_train,
        y_hist=y_train,
        round_num=1,
    )
    print(version.brier)
"""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def brier_score(probs: np.ndarray, y: np.ndarray) -> float:
    """Mean squared error between probability forecasts and binary outcomes."""
    return float(np.mean((np.asarray(probs, dtype=float) - np.asarray(y, dtype=float)) ** 2))


# ---------------------------------------------------------------------------
# AdapterVersion
# ---------------------------------------------------------------------------

@dataclass
class AdapterVersion:
    """
    Snapshot of the prediction model at a specific tournament round.

    Attributes
    ----------
    round_num : int
        0 = pre-tournament baseline.
    model : MatchupPredictor
        Deep copy of the fitted model.
    brier : float
        Brier score on the validation set when this version was created.
    elapsed_seconds : float
        Wall-clock time to produce this version.
    timestamp : str
        ISO-8601 UTC timestamp.
    metadata : dict
    """

    round_num: int
    model: Any
    brier: float
    elapsed_seconds: float = 0.0
    timestamp: str = field(
        default_factory=lambda: __import__("datetime").datetime.utcnow().isoformat()
    )
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# RoundRecalibrator
# ---------------------------------------------------------------------------

class RoundRecalibrator:
    """
    Recalibrates the prediction model after each tournament round.

    Parameters
    ----------
    base_model : MatchupPredictor
        Pre-tournament model implementing ``fit(X1, X2, y)`` and
        ``predict_proba(X1, X2) → np.ndarray``.
    feature_cols : list of str, optional
    """

    #: Maximum wall-clock seconds allowed for one recalibration cycle.
    MAX_RECAL_SECONDS: float = 30 * 60  # 30 minutes

    def __init__(self, base_model, feature_cols=None) -> None:
        from ncaa_models.baseline import FEATURE_COLS
        self.feature_cols = list(feature_cols or FEATURE_COLS)
        self._base_model = base_model
        self._versions: List[AdapterVersion] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_baseline(
        self,
        model,
        X1_val: np.ndarray,
        X2_val: np.ndarray,
        y_val: np.ndarray,
        round_num: int = 0,
    ) -> AdapterVersion:
        """
        Record the pre-tournament baseline version.

        Parameters
        ----------
        model : MatchupPredictor
        X1_val, X2_val, y_val : np.ndarray
            Validation arrays for Brier computation.
        round_num : int

        Returns
        -------
        AdapterVersion
        """
        b = brier_score(model.predict_proba(X1_val, X2_val), y_val)
        version = AdapterVersion(
            round_num=round_num,
            model=copy.deepcopy(model),
            brier=b,
            metadata={"label": "baseline"},
        )
        self._versions.append(version)
        logger.info("Baseline recorded: Brier=%.4f", b)
        return version

    def recalibrate(
        self,
        results_X1: np.ndarray,
        results_X2: np.ndarray,
        results_y: np.ndarray,
        X1_val: np.ndarray,
        X2_val: np.ndarray,
        y_val: np.ndarray,
        X1_hist: Optional[np.ndarray] = None,
        X2_hist: Optional[np.ndarray] = None,
        y_hist: Optional[np.ndarray] = None,
        round_num: int = 1,
    ) -> AdapterVersion:
        """
        Fit a new model version on historical + live tournament data.

        Parameters
        ----------
        results_X1, results_X2 : np.ndarray
            Feature vectors for completed tournament games.
        results_y : np.ndarray
            Outcomes (1 = team1 won, 0 = team2 won).
        X1_val, X2_val, y_val : np.ndarray
            Validation set for Brier evaluation.
        X1_hist, X2_hist, y_hist : np.ndarray, optional
            Historical (pre-tournament) training data.
        round_num : int

        Returns
        -------
        AdapterVersion
        """
        t0 = time.time()

        # Combine historical and new tournament data
        if X1_hist is not None and len(X1_hist) > 0:
            X1_tr = np.vstack([X1_hist, results_X1])
            X2_tr = np.vstack([X2_hist, results_X2])
            y_tr = np.concatenate([y_hist, results_y])
        else:
            X1_tr = np.asarray(results_X1, dtype=float)
            X2_tr = np.asarray(results_X2, dtype=float)
            y_tr = np.asarray(results_y, dtype=int)

        new_model = self._clone_and_fit(X1_tr, X2_tr, y_tr)
        elapsed = time.time() - t0

        b = brier_score(new_model.predict_proba(X1_val, X2_val), y_val)
        version = AdapterVersion(
            round_num=round_num,
            model=new_model,
            brier=b,
            elapsed_seconds=elapsed,
            metadata={
                "n_tournament_games": int(len(results_y)),
                "n_hist_games": int(len(y_hist)) if y_hist is not None else 0,
                "label": f"round_{round_num}",
            },
        )
        self._versions.append(version)

        if elapsed > self.MAX_RECAL_SECONDS:
            logger.warning(
                "Round %d recalibration took %.1f s (limit %.0f s)",
                round_num, elapsed, self.MAX_RECAL_SECONDS,
            )
        else:
            logger.info(
                "Round %d recalibrated: Brier=%.4f in %.3f s",
                round_num, b, elapsed,
            )
        return version

    def current_version(self) -> Optional[AdapterVersion]:
        """Most recent AdapterVersion, or None."""
        return self._versions[-1] if self._versions else None

    def version_history(self) -> List[AdapterVersion]:
        return list(self._versions)

    def compute_brier(self, model, X1: np.ndarray, X2: np.ndarray, y: np.ndarray) -> float:
        return brier_score(model.predict_proba(X1, X2), y)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _clone_and_fit(self, X1: np.ndarray, X2: np.ndarray, y: np.ndarray):
        """Deep-copy the base model and refit on the given data."""
        new_model = copy.deepcopy(self._base_model)
        # Reset any cached fit state on LogisticBaseline
        if hasattr(new_model, "_pipeline"):
            new_model._pipeline = None
        new_model.fit(X1, X2, y)
        return new_model
