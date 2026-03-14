"""
ncaa_agent.anomaly
==================
Stage 07 — Anomaly detection for NCAA tournament predictions.

A prediction is flagged as anomalous when the ensemble probability deviates
more than ``threshold`` (default 0.15) from the *seed-baseline* probability —
the historical win rate for that seed matchup type.

AnomalyDetector
---------------
- ``detect(traces)``       → all anomalous traces
- ``flag_top_k(traces, k)`` → top-k most surprising predictions
- ``prediction_error_recall`` → fraction of actual errors that are flagged

Seed baselines
--------------
Drawn from historical NCAA Division I tournament data (first-round
win rates for common seed matchups; later rounds use a formula-based
estimate).

Target: ≥70% of actual prediction errors (|prob - outcome| > 0.20)
should be among the flagged anomalies.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Historical seed win-rate table
# ---------------------------------------------------------------------------
# Key: (better_seed, worse_seed)  — better_seed < worse_seed
# Value: historical win probability for the better-seeded team

SEED_WIN_RATES: Dict[tuple, float] = {
    # First-round matchups (standard bracket)
    (1, 16): 0.993,
    (2, 15): 0.940,
    (3, 14): 0.850,
    (4, 13): 0.790,
    (5, 12): 0.650,
    (6, 11): 0.633,
    (7, 10): 0.603,
    (8,  9): 0.510,
    # Second-round matchups (common pairings)
    (1,  8): 0.770,
    (1,  9): 0.755,
    (2,  7): 0.720,
    (2, 10): 0.735,
    (3,  6): 0.640,
    (3, 11): 0.700,
    (4,  5): 0.590,
    (4, 12): 0.690,
    # Sweet 16 and beyond
    (1,  4): 0.780,
    (1,  5): 0.750,
    (1, 12): 0.840,
    (2,  3): 0.600,
    (2,  6): 0.680,
    (2, 11): 0.730,
    (1,  3): 0.700,
    (1,  2): 0.543,
    (3,  4): 0.540,
}

#: Default anomaly detection threshold
DEFAULT_THRESHOLD: float = 0.15

#: Default number of flagged anomalies to return
DEFAULT_TOP_K: int = 20


# ---------------------------------------------------------------------------
# Seed baseline utility
# ---------------------------------------------------------------------------

def seed_baseline_win_rate(seed1: int, seed2: int) -> float:
    """
    Return P(team1 wins) given seed matchup, using historical win rates.

    If ``seed1 < seed2`` (team1 is better-seeded), looks up the table or
    uses a formula-based estimate.  If ``seed1 > seed2``, returns the
    complement.

    Parameters
    ----------
    seed1, seed2 : int
        Seeds of team1 and team2 (1 = best).

    Returns
    -------
    float in [0.01, 0.99]
    """
    if seed1 == seed2:
        return 0.50

    # Normalise: always look up (lower, higher)
    if seed1 < seed2:
        lo, hi = seed1, seed2
        flip = False
    else:
        lo, hi = seed2, seed1
        flip = True

    # Look up exact entry first
    rate = SEED_WIN_RATES.get((lo, hi))
    if rate is None:
        # Formula fallback: logistic model calibrated to historical data
        # sigma(0.25 * seed_diff) gives reasonable approximations
        diff = hi - lo
        rate = float(1.0 / (1.0 + np.exp(-0.25 * diff)))

    rate = float(np.clip(rate, 0.01, 0.99))
    return 1.0 - rate if flip else rate


# ---------------------------------------------------------------------------
# AnomalyFlag dataclass
# ---------------------------------------------------------------------------

@dataclass
class AnomalyFlag:
    """
    Anomaly flag for a single matchup prediction.

    Attributes
    ----------
    matchup_id : str
    team1_name, team2_name : str
    seed1, seed2 : int
    ensemble_prob : float
    seed_baseline_prob : float
    deviation : float
        ``|ensemble_prob - seed_baseline_prob|``
    is_anomalous : bool
        ``True`` iff ``deviation > threshold``.
    rank : int
        Rank in the sorted top-k list (1 = largest deviation).
    """

    matchup_id: str
    team1_name: str
    team2_name: str
    seed1: int
    seed2: int
    ensemble_prob: float
    seed_baseline_prob: float
    deviation: float
    is_anomalous: bool
    rank: int = 0

    def to_dict(self) -> Dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# AnomalyDetector
# ---------------------------------------------------------------------------

class AnomalyDetector:
    """
    Detect anomalous model predictions by comparing to seed baselines.

    Parameters
    ----------
    threshold : float
        Minimum absolute deviation from seed baseline to flag as anomalous.
        Default 0.15.
    top_k : int
        Number of anomalies returned by ``flag_top_k()``.  Default 20.
    """

    def __init__(
        self,
        threshold: float = DEFAULT_THRESHOLD,
        top_k: int = DEFAULT_TOP_K,
    ) -> None:
        self.threshold = threshold
        self.top_k = top_k

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    def seed_baseline(self, seed1: int, seed2: int) -> float:
        """Return historical win probability for team1 given seeds."""
        return seed_baseline_win_rate(seed1, seed2)

    def compute_deviation(
        self,
        ensemble_prob: float,
        seed1: int,
        seed2: int,
    ) -> float:
        """Return |ensemble_prob - seed_baseline|."""
        baseline = self.seed_baseline(seed1, seed2)
        return abs(float(ensemble_prob) - baseline)

    def detect(
        self,
        traces,
    ) -> List[AnomalyFlag]:
        """
        Return all anomalous traces (deviation > threshold).

        Parameters
        ----------
        traces : list of ReasoningTrace

        Returns
        -------
        list of AnomalyFlag, sorted by deviation descending
        """
        flags = self._compute_flags(traces)
        anomalous = [f for f in flags if f.is_anomalous]
        anomalous.sort(key=lambda f: f.deviation, reverse=True)
        for rank, f in enumerate(anomalous, start=1):
            f.rank = rank
        return anomalous

    def flag_top_k(
        self,
        traces,
        k: Optional[int] = None,
    ) -> List[AnomalyFlag]:
        """
        Return the top-k most surprising predictions (by deviation).

        Parameters
        ----------
        traces : list of ReasoningTrace
        k : int, optional
            Overrides ``self.top_k`` if provided.

        Returns
        -------
        list of AnomalyFlag, length min(k, len(traces))
        """
        k = k if k is not None else self.top_k
        flags = self._compute_flags(traces)
        flags.sort(key=lambda f: f.deviation, reverse=True)
        top = flags[:k]
        for rank, f in enumerate(top, start=1):
            f.rank = rank
        return top

    def prediction_error_recall(
        self,
        traces,
        outcomes: Sequence[int],
        error_threshold: float = 0.20,
    ) -> float:
        """
        Fraction of actual prediction errors that are flagged as anomalous.

        Parameters
        ----------
        traces : list of ReasoningTrace (length n)
        outcomes : sequence of int (length n)
            Actual match outcomes: 1 if team1 won, 0 if team2 won.
        error_threshold : float
            |ensemble_prob - outcome| > this value counts as a prediction error.

        Returns
        -------
        float in [0, 1]
        """
        if not traces:
            return 0.0

        flags = self._compute_flags(traces)
        flag_by_id = {f.matchup_id: f for f in flags}

        n_errors = 0
        n_flagged_errors = 0
        for trace, outcome in zip(traces, outcomes):
            err = abs(trace.ensemble_prob - float(outcome))
            if err > error_threshold:
                n_errors += 1
                f = flag_by_id.get(trace.matchup_id)
                if f is not None and f.is_anomalous:
                    n_flagged_errors += 1

        if n_errors == 0:
            logger.warning("No prediction errors found; returning 0.0.")
            return 0.0

        recall = n_flagged_errors / n_errors
        logger.debug(
            "Anomaly recall: %d/%d errors flagged (%.1f%%)",
            n_flagged_errors, n_errors, 100 * recall,
        )
        return recall

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_flags(self, traces) -> List[AnomalyFlag]:
        """Compute AnomalyFlag for every trace."""
        flags = []
        for trace in traces:
            baseline = self.seed_baseline(trace.seed1, trace.seed2)
            deviation = abs(trace.ensemble_prob - baseline)
            flags.append(AnomalyFlag(
                matchup_id=trace.matchup_id,
                team1_name=trace.team1_name,
                team2_name=trace.team2_name,
                seed1=trace.seed1,
                seed2=trace.seed2,
                ensemble_prob=trace.ensemble_prob,
                seed_baseline_prob=baseline,
                deviation=deviation,
                is_anomalous=deviation > self.threshold,
            ))
        return flags

    def __repr__(self) -> str:
        return (
            f"AnomalyDetector(threshold={self.threshold}, top_k={self.top_k})"
        )
