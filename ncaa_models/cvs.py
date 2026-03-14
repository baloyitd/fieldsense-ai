"""
ncaa_models.cvs
===============
Stage 05 — Counterfactual Validity Score (CVS) computation.

A CVS quantifies how plausible a perturbed feature vector is, given
historical basketball data.  It is a number in [0, 1] where:

    1.0  — perfectly plausible (within all historical bounds, no contradictions)
    0.0  — completely implausible (extreme out-of-bounds values)

CVS formula
-----------
CVS = w_bounds * bounds_score
    + w_dist   * distribution_score
    + w_contra * contradiction_score

where:
    bounds_score       = fraction of features within historical D1 bounds
    distribution_score = penalty for features beyond 3 σ of training distribution
    contradiction_score = penalty for physically-contradictory feature combos

Default weights: 0.60 / 0.30 / 0.10 (bounds most important).

Threshold: CVS ≥ 0.92 for a scenario to be included in prediction adjustments.

Usage
-----
::

    from ncaa_models.cvs import CVSComputer, TrainingDistribution

    dist = TrainingDistribution.from_data(X_train, feature_cols)
    cvs = CVSComputer(dist)
    score = cvs.compute(X_original, X_perturbed)
    valid = cvs.is_valid(X_original, X_perturbed)  # score >= 0.92
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .baseline import FEATURE_COLS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hard-coded D1 basketball feature bounds
# Used when no training data is available.
# ---------------------------------------------------------------------------

#: (lower, upper) bounds for each feature in FEATURE_COLS.
_D1_BOUNDS: Dict[str, Tuple[float, float]] = {
    "games_played":    (15.0, 40.0),
    "win_pct":         (0.05, 1.00),
    "ppg_scored":      (50.0, 102.0),
    "ppg_allowed":     (50.0, 102.0),
    "scoring_margin":  (-42.0, 42.0),
    "home_win_pct":    (0.00, 1.00),
    "away_win_pct":    (0.00, 1.00),
    "neutral_win_pct": (0.00, 1.00),
    "win_pct_last10":  (0.00, 1.00),
    "fg_pct":          (0.26, 0.60),
    "fg3_pct":         (0.20, 0.50),   # 80% is NOT in this range → CVS drops
    "ft_pct":          (0.50, 0.95),
    "efg_pct":         (0.30, 0.65),
    "ts_pct":          (0.35, 0.70),
    "orb_rate":        (0.10, 0.55),
    "drb_rate":        (0.45, 0.90),
    "possessions_pg":  (55.0, 90.0),
    "ortg":            (75.0, 135.0),
    "drtg":            (75.0, 135.0),
    "net_rtg":         (-45.0, 45.0),
    "tempo":           (55.0, 90.0),
    "tov_rate":        (0.07, 0.35),   # 30 turnovers ≈ 0.43+ → NOT in range
    "ast_rate":        (0.35, 0.80),
    "blk_rate":        (0.02, 0.22),
    "stl_rate":        (0.02, 0.20),
    "ot_rate":         (0.00, 0.30),
    "sos":             (0.28, 0.78),
    "seed":            (1.00, 16.00),
}

# Default mean and standard deviation (derived from D1 bounds assuming midpoint ± 3σ).
_D1_MEANS: Dict[str, float] = {k: (lo + hi) / 2.0 for k, (lo, hi) in _D1_BOUNDS.items()}
_D1_STDS: Dict[str, float]  = {k: (hi - lo) / 6.0 for k, (lo, hi) in _D1_BOUNDS.items()}

# Default CVS threshold
DEFAULT_CVS_THRESHOLD: float = 0.92

# Default CVS component weights (must sum to 1.0)
DEFAULT_WEIGHTS: Tuple[float, float, float] = (0.60, 0.30, 0.10)


# ---------------------------------------------------------------------------
# TrainingDistribution
# ---------------------------------------------------------------------------

@dataclass
class TrainingDistribution:
    """
    Per-feature statistics computed from a training set.

    Attributes
    ----------
    feature_cols : list of str
    bounds_lo : np.ndarray, shape (feature_dim,)
        Minimum observed value per feature.
    bounds_hi : np.ndarray, shape (feature_dim,)
        Maximum observed value per feature.
    mean : np.ndarray, shape (feature_dim,)
    std  : np.ndarray, shape (feature_dim,)
        Standard deviation; floored at 1e-6 to avoid division by zero.
    """

    feature_cols: List[str]
    bounds_lo: np.ndarray
    bounds_hi: np.ndarray
    mean: np.ndarray
    std:  np.ndarray

    # ------------------------------------------------------------------

    @classmethod
    def from_data(
        cls,
        X: np.ndarray,
        feature_cols: Optional[List[str]] = None,
    ) -> "TrainingDistribution":
        """
        Fit distribution statistics from a feature matrix.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, feature_dim)
        feature_cols : list of str, optional

        Returns
        -------
        TrainingDistribution
        """
        feature_cols = list(feature_cols or FEATURE_COLS)
        if X.ndim != 2 or X.shape[1] != len(feature_cols):
            raise ValueError(
                f"X must be shape (n_samples, {len(feature_cols)}), "
                f"got {X.shape}."
            )
        return cls(
            feature_cols=feature_cols,
            bounds_lo=X.min(axis=0),
            bounds_hi=X.max(axis=0),
            mean=X.mean(axis=0),
            std=np.maximum(X.std(axis=0), 1e-6),
        )

    @classmethod
    def default(cls, feature_cols: Optional[List[str]] = None) -> "TrainingDistribution":
        """
        Return a ``TrainingDistribution`` built from hard-coded D1 bounds.

        Useful when no training data is available.

        Parameters
        ----------
        feature_cols : list of str, optional
            Defaults to :data:`~ncaa_models.baseline.FEATURE_COLS`.
        """
        feature_cols = list(feature_cols or FEATURE_COLS)
        lo = np.array([_D1_BOUNDS.get(c, (-1e9, 1e9))[0] for c in feature_cols])
        hi = np.array([_D1_BOUNDS.get(c, (-1e9, 1e9))[1] for c in feature_cols])
        mu = np.array([_D1_MEANS.get(c, 0.0) for c in feature_cols])
        sd = np.array([_D1_STDS.get(c, 1.0) for c in feature_cols])
        return cls(
            feature_cols=feature_cols,
            bounds_lo=lo,
            bounds_hi=hi,
            mean=mu,
            std=np.maximum(sd, 1e-6),
        )


# ---------------------------------------------------------------------------
# CVSComputer
# ---------------------------------------------------------------------------

class CVSComputer:
    """
    Compute the Counterfactual Validity Score (CVS) for a perturbed feature vector.

    Parameters
    ----------
    training_dist : TrainingDistribution, optional
        Feature statistics.  If ``None``, uses the hard-coded D1 defaults.
    weights : tuple of 3 floats
        Weights for (bounds, distribution, contradiction) components.
        Must sum to 1.0.  Default ``(0.60, 0.30, 0.10)``.
    threshold : float
        CVS ≥ threshold → scenario is valid.  Default 0.92.
    """

    def __init__(
        self,
        training_dist: Optional[TrainingDistribution] = None,
        weights: Tuple[float, float, float] = DEFAULT_WEIGHTS,
        threshold: float = DEFAULT_CVS_THRESHOLD,
        feature_cols: Optional[List[str]] = None,
    ) -> None:
        if abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError(f"weights must sum to 1.0, got {sum(weights):.4f}")

        fc = list(feature_cols or FEATURE_COLS)
        if training_dist is None:
            training_dist = TrainingDistribution.default(fc)
        self.dist = training_dist
        self.w_bounds, self.w_dist, self.w_contra = weights
        self.threshold = threshold
        self.feat_idx = {c: i for i, c in enumerate(self.dist.feature_cols)}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def compute(
        self,
        X_original: np.ndarray,
        X_perturbed: np.ndarray,
    ) -> float:
        """
        Compute CVS for a single perturbed feature vector.

        Parameters
        ----------
        X_original : np.ndarray, shape (feature_dim,)
            Original (unperturbed) feature vector.
        X_perturbed : np.ndarray, shape (feature_dim,)
            Perturbed feature vector to score.

        Returns
        -------
        float in [0, 1]
        """
        X_p = np.asarray(X_perturbed, dtype=float).ravel()
        bs = self._bounds_score(X_p)
        ds = self._distribution_score(X_p)
        cs = self._contradiction_score(X_p)
        cvs = self.w_bounds * bs + self.w_dist * ds + self.w_contra * cs
        return float(np.clip(cvs, 0.0, 1.0))

    def is_valid(
        self,
        X_original: np.ndarray,
        X_perturbed: np.ndarray,
    ) -> bool:
        """Return ``True`` iff ``compute(...) >= self.threshold``."""
        return self.compute(X_original, X_perturbed) >= self.threshold

    def batch_compute(
        self,
        X_original: np.ndarray,
        X_perturbed_list: Sequence[np.ndarray],
    ) -> List[float]:
        """
        Compute CVS for multiple perturbed vectors against one original.

        Returns
        -------
        list of float
        """
        return [self.compute(X_original, xp) for xp in X_perturbed_list]

    # ------------------------------------------------------------------
    # Component scores
    # ------------------------------------------------------------------

    def _bounds_score(self, X: np.ndarray) -> float:
        """
        Fraction of features that lie within [bounds_lo, bounds_hi].

        Any feature outside the historical D1 range contributes 0 to the
        score; inside contributes 1.
        """
        lo = self.dist.bounds_lo
        hi = self.dist.bounds_hi
        in_bounds = np.logical_and(X >= lo, X <= hi)
        return float(np.mean(in_bounds))

    def _distribution_score(self, X: np.ndarray) -> float:
        """
        Score based on z-score distance from training mean.

        Full score (1.0) if all z-scores ≤ 3.  Linearly penalised for
        the worst-offending feature beyond 3 σ::

            score = max(0, 1 - (z_max - 3) / 3)
        """
        z = np.abs(X - self.dist.mean) / self.dist.std
        z_max = float(z.max())
        if z_max <= 3.0:
            return 1.0
        return float(max(0.0, 1.0 - (z_max - 3.0) / 3.0))

    def _contradiction_score(self, X: np.ndarray) -> float:
        """
        Penalty for physically contradictory feature combinations.

        Checks:
        1. FG3% > 0.55 (no sustained D1 team has ever shot 55%+ from 3)
        2. FG%  > 0.62 (hard physical upper limit)
        3. TOV% > 0.40 (30+ turnovers per game is impossible to sustain)
        4. FG3% > 0.50 AND TOV% > 0.30 (elite shooting + chaos = contradiction)
        5. ORTG > 130 AND DRTG > 120 (can't be both elite offensively and
           terrible defensively at the same level of play)
        """
        fi = self.feat_idx
        penalty = 0.0

        # 1. Physically impossible FG3%
        if fi.get("fg3_pct") is not None and X[fi["fg3_pct"]] > 0.55:
            penalty += 0.50

        # 2. Physically impossible FG%
        if fi.get("fg_pct") is not None and X[fi["fg_pct"]] > 0.62:
            penalty += 0.30

        # 3. Impossible turnover rate
        if fi.get("tov_rate") is not None and X[fi["tov_rate"]] > 0.40:
            penalty += 0.40

        # 4. Elite shooting + chaos
        if (fi.get("fg3_pct") is not None
                and fi.get("tov_rate") is not None
                and X[fi["fg3_pct"]] > 0.50
                and X[fi["tov_rate"]] > 0.30):
            penalty += 0.25

        # 5. Impossible ortg/drtg combo
        if (fi.get("ortg") is not None
                and fi.get("drtg") is not None
                and X[fi["ortg"]] > 130.0
                and X[fi["drtg"]] > 120.0):
            penalty += 0.20

        return float(max(0.0, 1.0 - penalty))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"CVSComputer("
            f"threshold={self.threshold}, "
            f"weights=({self.w_bounds:.2f}, {self.w_dist:.2f}, {self.w_contra:.2f}))"
        )
