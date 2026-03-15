"""
ncaa_deployment.submission
==========================
Stage 10 — Generate two competition-ready Kaggle submission variants.

Conservative Submission
    Isotonic calibration + temperature > 1.0 (predictions closer to 0.5).
    Optimised for worst-case stability; sacrifices peak accuracy for
    robustness against upsets.

Aggressive Submission
    Optimised ensemble weights + temperature ≤ 1.0 (sharper predictions).
    Targets minimum expected Brier score; accepts higher variance.

Both variants:
  * Pass :func:`ncaa_models.submit.validate_submission` schema checks.
  * Produce predictions in [0.01, 0.99].
  * Use the standard ``ID`` / ``Pred`` Kaggle column format.

Usage
-----
::

    builder = SubmissionBuilder(feat_df=feat_df, team_ids=TEAM_IDS,
                                season=2025, feature_cols=FEATURE_COLS)
    conservative = builder.build_conservative(model, temperature=5.0)
    aggressive   = builder.build_aggressive(model)
    kl = kl_divergence(conservative, aggressive)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import entropy

from ncaa_deployment.optimize import WeightedEnsemble, apply_temperature
from ncaa_models.submit import build_submission, validate_submission

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SubmissionVariant
# ---------------------------------------------------------------------------

@dataclass
class SubmissionVariant:
    """
    A named Kaggle submission variant.

    Attributes
    ----------
    name : str
        ``'conservative'`` or ``'aggressive'``.
    df : pd.DataFrame
        Kaggle-format DataFrame with columns ``ID`` and ``Pred``.
    temperature : float
    calibration : str
    brier : float or None
        If ground-truth labels are available.
    metadata : dict
    """

    name: str
    df: pd.DataFrame
    temperature: float = 1.0
    calibration: str = "none"
    brier: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self, expected_season: Optional[int] = None) -> Dict[str, Any]:
        """Run Kaggle schema validation."""
        return validate_submission(self.df, expected_season=expected_season)

    def save_csv(self, path: str) -> None:
        """Save submission to CSV."""
        self.df.to_csv(path, index=False)
        logger.info("Submission '%s' saved to %s", self.name, path)

    def __len__(self) -> int:
        return len(self.df)


# ---------------------------------------------------------------------------
# KL divergence helper
# ---------------------------------------------------------------------------

def kl_divergence(
    sub_p: "SubmissionVariant | pd.DataFrame",
    sub_q: "SubmissionVariant | pd.DataFrame",
    eps: float = 1e-9,
) -> float:
    """
    Compute KL divergence KL(P || Q) between two submission prediction vectors,
    treating the normalised prediction vectors as discrete probability distributions.

    Parameters
    ----------
    sub_p : SubmissionVariant or pd.DataFrame with 'ID' and 'Pred' columns
    sub_q : SubmissionVariant or pd.DataFrame
    eps : float
        Small constant added before normalisation to avoid log(0).

    Returns
    -------
    float  (≥ 0; 0 means identical distributions)
    """
    df_p = sub_p.df if isinstance(sub_p, SubmissionVariant) else sub_p
    df_q = sub_q.df if isinstance(sub_q, SubmissionVariant) else sub_q

    # Align on ID
    merged = df_p[["ID", "Pred"]].merge(
        df_q[["ID", "Pred"]], on="ID", suffixes=("_p", "_q")
    )
    p = np.asarray(merged["Pred_p"].values, dtype=float) + eps
    q = np.asarray(merged["Pred_q"].values, dtype=float) + eps

    # Normalise to distributions
    p_norm = p / p.sum()
    q_norm = q / q.sum()

    return float(entropy(p_norm, q_norm))


# ---------------------------------------------------------------------------
# SubmissionBuilder
# ---------------------------------------------------------------------------

class SubmissionBuilder:
    """
    Builds conservative and aggressive Kaggle submission CSVs.

    Parameters
    ----------
    feat_df : pd.DataFrame
        Full feature matrix (all seasons, all teams).
    team_ids : sequence of int
    season : int
    feature_cols : list of str
    """

    def __init__(
        self,
        feat_df: pd.DataFrame,
        team_ids: Sequence[int],
        season: int = 2025,
        feature_cols: Optional[List[str]] = None,
    ) -> None:
        from ncaa_models.baseline import FEATURE_COLS
        self.feat_df = feat_df
        self.team_ids = list(team_ids)
        self.season = season
        self.feature_cols = list(feature_cols or FEATURE_COLS)

    # ------------------------------------------------------------------
    # Conservative
    # ------------------------------------------------------------------

    def build_conservative(
        self,
        model,
        temperature: float = 5.0,
        calibration: str = "isotonic",
        X1_cal: Optional[np.ndarray] = None,
        X2_cal: Optional[np.ndarray] = None,
        y_cal: Optional[np.ndarray] = None,
    ) -> SubmissionVariant:
        """
        Build the conservative submission.

        Strategy:
          1. Optionally apply isotonic calibration (trained on ``X1_cal``).
          2. Apply temperature scaling with ``temperature > 1.0`` to push
             predictions toward 0.5.

        Parameters
        ----------
        model : MatchupPredictor (fitted)
        temperature : float
            Default 5.0 (strongly conservative).
        calibration : str
            ``'isotonic'``, ``'platt'``, or ``'none'``.
        X1_cal, X2_cal, y_cal : calibration data (optional).
        """
        cal_model = self._maybe_calibrate(model, calibration, X1_cal, X2_cal, y_cal)
        wrapped = _TemperatureModel(cal_model, temperature)
        df = build_submission(
            model=wrapped,
            team_features=self.feat_df,
            team_ids=self.team_ids,
            season=self.season,
            feature_cols=self.feature_cols,
            clip_probs=True,
        )
        logger.info(
            "Conservative submission: n=%d, mean_pred=%.4f",
            len(df), df["Pred"].mean(),
        )
        return SubmissionVariant(
            name="conservative",
            df=df,
            temperature=temperature,
            calibration=calibration,
            metadata={
                "strategy": "isotonic_calibration+temperature_scaling",
                "season": self.season,
            },
        )

    # ------------------------------------------------------------------
    # Aggressive
    # ------------------------------------------------------------------

    def build_aggressive(
        self,
        model,
        temperature: float = 1.0,
        calibration: str = "none",
    ) -> SubmissionVariant:
        """
        Build the aggressive submission.

        Strategy:
          1. Use the model's native predictions (from optimised ensemble weights).
          2. Apply temperature ≤ 1.0 for sharper predictions (or 1.0 = unchanged).

        Parameters
        ----------
        model : MatchupPredictor (fitted)
        temperature : float
            Default 1.0.  Use < 1.0 for even sharper predictions.
        """
        cal_model = self._maybe_calibrate(model, calibration)
        wrapped = _TemperatureModel(cal_model, temperature)
        df = build_submission(
            model=wrapped,
            team_features=self.feat_df,
            team_ids=self.team_ids,
            season=self.season,
            feature_cols=self.feature_cols,
            clip_probs=True,
        )
        logger.info(
            "Aggressive submission: n=%d, mean_pred=%.4f",
            len(df), df["Pred"].mean(),
        )
        return SubmissionVariant(
            name="aggressive",
            df=df,
            temperature=temperature,
            calibration=calibration,
            metadata={
                "strategy": "optimised_weights",
                "season": self.season,
            },
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _maybe_calibrate(
        self,
        model,
        calibration: str,
        X1_cal: Optional[np.ndarray] = None,
        X2_cal: Optional[np.ndarray] = None,
        y_cal: Optional[np.ndarray] = None,
    ):
        """Return a (possibly calibrated) model wrapper."""
        if calibration == "none" or X1_cal is None:
            return model
        try:
            from ncaa_models.calibration import PostHocCalibrator
            cal = PostHocCalibrator(base_model=model, method=calibration)
            # Use passed-in calibration data
            cal.fit(
                np.atleast_2d(X1_cal),
                np.atleast_2d(X2_cal),
                np.asarray(y_cal, dtype=int),
                X1_cal=np.atleast_2d(X1_cal),
                X2_cal=np.atleast_2d(X2_cal),
                y_cal=np.asarray(y_cal, dtype=int),
            )
            return cal
        except Exception as exc:
            logger.warning("Calibration '%s' failed (%s); using uncalibrated model.", calibration, exc)
            return model


# ---------------------------------------------------------------------------
# _TemperatureModel — thin wrapper applying temperature scaling
# ---------------------------------------------------------------------------

class _TemperatureModel:
    """Wraps any MatchupPredictor with temperature scaling at predict time."""

    def __init__(self, base, temperature: float = 1.0) -> None:
        self._base = base
        self.temperature = temperature

    def predict_proba(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        raw = self._base.predict_proba(X1, X2)
        return apply_temperature(raw, self.temperature)

    def fit(self, X1: np.ndarray, X2: np.ndarray, y: np.ndarray) -> "_TemperatureModel":
        return self
