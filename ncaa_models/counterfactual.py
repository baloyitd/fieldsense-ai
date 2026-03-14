"""
ncaa_models.counterfactual
==========================
Stage 05 — Counterfactual scenario generator for NCAA matchup analysis.

A CounterfactualGenerator applies basketball-specific perturbations to team
feature vectors and records how the model's probability estimate changes.
This enables "what-if" analysis: What if a star player were injured? What if
the underdog went on a hot-shooting streak?

Perturbation types
------------------
injury          Remove star player → reduce ORTG 5–20%, propagate to ppg/margin
foul_trouble    Key player limited → reduce ORTG/FG%, reduces team efficiency
hot_shooting    Shoot above season average → increase FG%/3P% 5–15 pp
cold_shooting   Shoot below average → decrease FG%/3P% 5–15 pp
rest_advantage  Extra rest → boost efficiency 2–5%, reduce turnover rate

Each scenario is described by a CounterfactualScenario dataclass with the
perturbed feature arrays, the model probability, and a Counterfactual Validity
Score (CVS) supplied by an optional CVSComputer.

Usage
-----
::

    gen = CounterfactualGenerator(model=my_model, cvs_computer=cvs)
    scenarios = gen.generate(X1, X2, seed=42)
    valid = [s for s in scenarios if s.is_valid]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from .baseline import FEATURE_COLS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Perturbation type constants
# ---------------------------------------------------------------------------

INJURY = "injury"
FOUL_TROUBLE = "foul_trouble"
HOT_SHOOTING = "hot_shooting"
COLD_SHOOTING = "cold_shooting"
REST_ADVANTAGE = "rest_advantage"

ALL_PERTURBATION_TYPES: Tuple[str, ...] = (
    INJURY, FOUL_TROUBLE, HOT_SHOOTING, COLD_SHOOTING, REST_ADVANTAGE,
)

#: Teams that can be perturbed in each scenario
BOTH_TEAMS: Tuple[str, str] = ("team1", "team2")

#: Default severity levels sampled per perturbation
DEFAULT_SEVERITIES: Tuple[float, float, float] = (0.25, 0.50, 0.75)

#: Default CVS threshold for a scenario to be considered valid
DEFAULT_CVS_THRESHOLD: float = 0.92


# ---------------------------------------------------------------------------
# Scenario dataclass
# ---------------------------------------------------------------------------

@dataclass
class CounterfactualScenario:
    """
    One counterfactual scenario for a matchup.

    Attributes
    ----------
    perturbation : str
        Name of the perturbation applied (e.g. ``'hot_shooting'``).
    team : str
        Which team was perturbed: ``'team1'`` or ``'team2'``.
    severity : float
        Perturbation intensity in [0, 1].  0 = minimal, 1 = maximal.
    X1 : np.ndarray, shape (feature_dim,)
        (Potentially perturbed) team-1 feature vector.
    X2 : np.ndarray, shape (feature_dim,)
        (Potentially perturbed) team-2 feature vector.
    base_prob : float
        Model P(team1 wins) before any perturbation.
    prob : float
        Model P(team1 wins) after perturbation.
    delta_prob : float
        ``prob - base_prob``.
    cvs : float
        Counterfactual Validity Score for the perturbed team (0–1).
    is_valid : bool
        ``True`` iff ``cvs >= cvs_threshold``.
    """
    perturbation: str
    team: str
    severity: float
    X1: np.ndarray
    X2: np.ndarray
    base_prob: float
    prob: float
    delta_prob: float
    cvs: float
    is_valid: bool

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def underdog_prob(self) -> float:
        """
        Win probability for whichever team has the lower base probability.

        If team1 is the underdog (base_prob < 0.5): ``underdog_prob = prob``.
        If team2 is the underdog (base_prob >= 0.5): ``underdog_prob = 1 - prob``.
        """
        return self.prob if self.base_prob < 0.5 else (1.0 - self.prob)

    def __repr__(self) -> str:
        return (
            f"CF({self.perturbation} on {self.team}, "
            f"sev={self.severity:.2f}, "
            f"Δp={self.delta_prob:+.3f}, "
            f"cvs={self.cvs:.3f}, valid={self.is_valid})"
        )


# ---------------------------------------------------------------------------
# Perturbation implementations
# ---------------------------------------------------------------------------

def _apply_injury(X: np.ndarray, severity: float, fi: dict) -> np.ndarray:
    """
    Star player injury: reduce ORTG by 5–20% (severity-dependent).

    Propagates to: ppg_scored, scoring_margin, net_rtg, fg_pct, efg_pct, ts_pct.
    """
    X = X.copy()
    ortg_drop = 0.05 + 0.15 * severity          # 5 – 20 %

    old_ortg = X[fi["ortg"]]
    new_ortg = max(75.0, old_ortg * (1.0 - ortg_drop))
    X[fi["ortg"]] = new_ortg

    # ppg_scored ~ ortg * possessions / 100
    X[fi["ppg_scored"]] = new_ortg * X[fi["possessions_pg"]] / 100.0
    X[fi["net_rtg"]] = new_ortg - X[fi["drtg"]]
    X[fi["scoring_margin"]] = X[fi["ppg_scored"]] - X[fi["ppg_allowed"]]

    # Shooting efficiency also drops (star player was efficient)
    shoot_drop = 0.02 + 0.05 * severity         # 2 – 7 pp
    X[fi["fg_pct"]]  = max(0.26, X[fi["fg_pct"]]  - shoot_drop * 0.6)
    X[fi["efg_pct"]] = max(0.30, X[fi["efg_pct"]] - shoot_drop * 0.7)
    X[fi["ts_pct"]]  = max(0.35, X[fi["ts_pct"]]  - shoot_drop * 0.7)

    return X


def _apply_foul_trouble(X: np.ndarray, severity: float, fi: dict) -> np.ndarray:
    """
    Foul trouble: key player limited → ORTG –3–12%, FG% –1–5 pp.

    Opponent FTA opportunity is modelled implicitly as lower team efficiency
    (the limited player avoids drives, reducing ORTG without purely a shooting drop).
    """
    X = X.copy()
    ortg_drop = 0.03 + 0.09 * severity          # 3 – 12 %
    new_ortg = max(75.0, X[fi["ortg"]] * (1.0 - ortg_drop))
    X[fi["ortg"]] = new_ortg
    X[fi["ppg_scored"]] = new_ortg * X[fi["possessions_pg"]] / 100.0
    X[fi["net_rtg"]] = new_ortg - X[fi["drtg"]]
    X[fi["scoring_margin"]] = X[fi["ppg_scored"]] - X[fi["ppg_allowed"]]

    fg_drop = 0.01 + 0.04 * severity            # 1 – 5 pp
    X[fi["fg_pct"]]  = max(0.26, X[fi["fg_pct"]] - fg_drop)
    X[fi["ast_rate"]] = max(0.35, X[fi["ast_rate"]] - 0.03 * severity)

    return X


def _apply_hot_shooting(X: np.ndarray, severity: float, fi: dict) -> np.ndarray:
    """
    Hot shooting: FG3% +5–15 pp, FG% +3–10 pp; ORTG and scoring improve.
    """
    X = X.copy()
    fg3_boost = 0.05 + 0.10 * severity          # 5 – 15 pp
    fg_boost  = fg3_boost * 0.65

    X[fi["fg3_pct"]] = min(0.60, X[fi["fg3_pct"]] + fg3_boost)
    X[fi["fg_pct"]]  = min(0.62, X[fi["fg_pct"]]  + fg_boost)
    X[fi["efg_pct"]] = min(0.68, X[fi["efg_pct"]] + fg3_boost * 0.70)
    X[fi["ts_pct"]]  = min(0.72, X[fi["ts_pct"]]  + fg3_boost * 0.70)

    # ORTG improves: +1 pp FG3% ≈ +1.5 ORTG points (league average estimate)
    ortg_gain = fg3_boost * 15.0
    new_ortg = min(135.0, X[fi["ortg"]] + ortg_gain)
    X[fi["ortg"]] = new_ortg
    X[fi["ppg_scored"]] = new_ortg * X[fi["possessions_pg"]] / 100.0
    X[fi["net_rtg"]] = new_ortg - X[fi["drtg"]]
    X[fi["scoring_margin"]] = X[fi["ppg_scored"]] - X[fi["ppg_allowed"]]

    return X


def _apply_cold_shooting(X: np.ndarray, severity: float, fi: dict) -> np.ndarray:
    """
    Cold shooting: FG3% –5–15 pp, FG% –3–10 pp; ORTG and scoring drop.
    """
    X = X.copy()
    fg3_drop = 0.05 + 0.10 * severity
    fg_drop  = fg3_drop * 0.65

    X[fi["fg3_pct"]] = max(0.20, X[fi["fg3_pct"]] - fg3_drop)
    X[fi["fg_pct"]]  = max(0.26, X[fi["fg_pct"]]  - fg_drop)
    X[fi["efg_pct"]] = max(0.30, X[fi["efg_pct"]] - fg3_drop * 0.70)
    X[fi["ts_pct"]]  = max(0.35, X[fi["ts_pct"]]  - fg3_drop * 0.70)

    ortg_loss = fg3_drop * 15.0
    new_ortg = max(75.0, X[fi["ortg"]] - ortg_loss)
    X[fi["ortg"]] = new_ortg
    X[fi["ppg_scored"]] = new_ortg * X[fi["possessions_pg"]] / 100.0
    X[fi["net_rtg"]] = new_ortg - X[fi["drtg"]]
    X[fi["scoring_margin"]] = X[fi["ppg_scored"]] - X[fi["ppg_allowed"]]

    return X


def _apply_rest_advantage(X: np.ndarray, severity: float, fi: dict) -> np.ndarray:
    """
    Rest advantage: extra rest → ORTG +2–5%, DRTG –1–2.5%, TOV –3–8%.
    """
    X = X.copy()
    off_boost = 0.02 + 0.03 * severity          # 2 – 5 %
    def_boost = off_boost * 0.50                  # half on defense

    new_ortg = min(135.0, X[fi["ortg"]] * (1.0 + off_boost))
    new_drtg = max(75.0,  X[fi["drtg"]] * (1.0 - def_boost))
    X[fi["ortg"]] = new_ortg
    X[fi["drtg"]] = new_drtg
    X[fi["net_rtg"]] = new_ortg - new_drtg

    X[fi["ppg_scored"]]  = new_ortg * X[fi["possessions_pg"]] / 100.0
    X[fi["ppg_allowed"]] = new_drtg * X[fi["possessions_pg"]] / 100.0
    X[fi["scoring_margin"]] = X[fi["ppg_scored"]] - X[fi["ppg_allowed"]]

    tov_reduction = 0.03 + 0.05 * severity
    X[fi["tov_rate"]] = max(0.07, X[fi["tov_rate"]] * (1.0 - tov_reduction))

    return X


_PERTURBATION_FNS = {
    INJURY:         _apply_injury,
    FOUL_TROUBLE:   _apply_foul_trouble,
    HOT_SHOOTING:   _apply_hot_shooting,
    COLD_SHOOTING:  _apply_cold_shooting,
    REST_ADVANTAGE: _apply_rest_advantage,
}


# ---------------------------------------------------------------------------
# CounterfactualGenerator
# ---------------------------------------------------------------------------

class CounterfactualGenerator:
    """
    Generate counterfactual matchup scenarios by perturbing team features.

    Parameters
    ----------
    model : MatchupPredictor or callable
        Must expose ``predict_proba(X1, X2) -> np.ndarray`` or be directly
        callable as ``model(X1, X2) -> np.ndarray``.
    feature_cols : list of str, optional
        Feature column names.  Defaults to :data:`~ncaa_models.baseline.FEATURE_COLS`.
    cvs_computer : CVSComputer, optional
        If provided, each scenario's CVS and ``is_valid`` flag are populated.
        If ``None``, CVS is set to 1.0 and all scenarios are marked valid.
    severities : tuple of float, optional
        Severity levels to test.  Default ``(0.25, 0.50, 0.75)``.
    cvs_threshold : float
        Threshold above which a scenario is marked valid.  Default 0.92.
    """

    def __init__(
        self,
        model,
        feature_cols: Optional[List[str]] = None,
        cvs_computer=None,
        severities: Tuple[float, ...] = DEFAULT_SEVERITIES,
        cvs_threshold: float = DEFAULT_CVS_THRESHOLD,
    ) -> None:
        self.feature_cols = list(feature_cols or FEATURE_COLS)
        self.feat_idx = {c: i for i, c in enumerate(self.feature_cols)}
        self.cvs_computer = cvs_computer
        self.severities = tuple(severities)
        self.cvs_threshold = cvs_threshold

        # Normalise model call interface
        if hasattr(model, "predict_proba"):
            self._predict = model.predict_proba
        elif callable(model):
            self._predict = model
        else:
            raise TypeError(
                "model must be a MatchupPredictor (with predict_proba) "
                "or a plain callable (X1, X2) -> np.ndarray."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        X1: np.ndarray,
        X2: np.ndarray,
        perturbation_types: Optional[Sequence[str]] = None,
        teams: Tuple[str, ...] = BOTH_TEAMS,
        seed: int = 42,
    ) -> List[CounterfactualScenario]:
        """
        Generate all counterfactual scenarios for a **single** matchup.

        Parameters
        ----------
        X1, X2 : np.ndarray, shape (feature_dim,) or (1, feature_dim)
            Team feature vectors.
        perturbation_types : list of str, optional
            Subset of perturbations to generate.  Defaults to all 5.
        teams : tuple of str
            Which teams to perturb.  Default both.
        seed : int
            Random seed for reproducibility.  Identical ``(seed, X1, X2)``
            always produces identical scenario traces.

        Returns
        -------
        list of CounterfactualScenario
            One entry per (perturbation, team, severity) combination.
        """
        X1 = np.asarray(X1, dtype=float).ravel()
        X2 = np.asarray(X2, dtype=float).ravel()

        if perturbation_types is None:
            perturbation_types = ALL_PERTURBATION_TYPES

        # Base prediction
        base_prob = float(
            self._predict(X1.reshape(1, -1), X2.reshape(1, -1))[0]
        )

        scenarios: List[CounterfactualScenario] = []
        for ptype in perturbation_types:
            if ptype not in _PERTURBATION_FNS:
                raise ValueError(
                    f"Unknown perturbation '{ptype}'. "
                    f"Valid: {ALL_PERTURBATION_TYPES}"
                )
            fn = _PERTURBATION_FNS[ptype]
            fi = self.feat_idx

            for team in teams:
                for severity in self.severities:
                    # Apply perturbation to the chosen team
                    if team == "team1":
                        X1_p = fn(X1, severity, fi)
                        X2_p = X2.copy()
                        X_orig = X1
                        X_pert = X1_p
                    else:
                        X1_p = X1.copy()
                        X2_p = fn(X2, severity, fi)
                        X_orig = X2
                        X_pert = X2_p

                    prob = float(
                        self._predict(X1_p.reshape(1, -1), X2_p.reshape(1, -1))[0]
                    )
                    delta_prob = prob - base_prob

                    # CVS
                    if self.cvs_computer is not None:
                        cvs = self.cvs_computer.compute(X_orig, X_pert)
                    else:
                        cvs = 1.0

                    scenarios.append(
                        CounterfactualScenario(
                            perturbation=ptype,
                            team=team,
                            severity=severity,
                            X1=X1_p,
                            X2=X2_p,
                            base_prob=base_prob,
                            prob=prob,
                            delta_prob=delta_prob,
                            cvs=cvs,
                            is_valid=(cvs >= self.cvs_threshold),
                        )
                    )

        logger.debug(
            "Generated %d scenarios (base_prob=%.3f, %d valid).",
            len(scenarios),
            base_prob,
            sum(s.is_valid for s in scenarios),
        )
        return scenarios

    def generate_single(
        self,
        X1: np.ndarray,
        X2: np.ndarray,
        perturbation_type: str,
        team: str,
        severity: float,
    ) -> CounterfactualScenario:
        """
        Generate one specific counterfactual scenario.

        Parameters
        ----------
        X1, X2 : np.ndarray, shape (feature_dim,)
        perturbation_type : str
        team : str, ``'team1'`` or ``'team2'``
        severity : float, in [0, 1]
        """
        scenarios = self.generate(
            X1, X2,
            perturbation_types=[perturbation_type],
            teams=(team,),
            seed=42,
        )
        # Override severity (generate() uses self.severities; we want exactly this one)
        X1 = np.asarray(X1, dtype=float).ravel()
        X2 = np.asarray(X2, dtype=float).ravel()
        base_prob = float(
            self._predict(X1.reshape(1, -1), X2.reshape(1, -1))[0]
        )
        fn = _PERTURBATION_FNS[perturbation_type]
        fi = self.feat_idx

        if team == "team1":
            X1_p = fn(X1, severity, fi)
            X2_p = X2.copy()
            X_orig, X_pert = X1, X1_p
        else:
            X1_p = X1.copy()
            X2_p = fn(X2, severity, fi)
            X_orig, X_pert = X2, X2_p

        prob = float(self._predict(X1_p.reshape(1, -1), X2_p.reshape(1, -1))[0])
        cvs = self.cvs_computer.compute(X_orig, X_pert) if self.cvs_computer else 1.0

        return CounterfactualScenario(
            perturbation=perturbation_type,
            team=team,
            severity=severity,
            X1=X1_p,
            X2=X2_p,
            base_prob=base_prob,
            prob=prob,
            delta_prob=prob - base_prob,
            cvs=cvs,
            is_valid=(cvs >= self.cvs_threshold),
        )

    def adjusted_probability(
        self,
        X1: np.ndarray,
        X2: np.ndarray,
        seed: int = 42,
        blend_weight: float = 0.20,
    ) -> float:
        """
        Return a counterfactual-adjusted win probability for team1.

        The adjusted probability blends the base prediction with the mean of
        **valid** counterfactual scenario probabilities:

            adjusted = (1 - w) * base_prob + w * mean(valid_cf_probs)

        If no valid scenarios exist, the base probability is returned unchanged.

        Parameters
        ----------
        X1, X2 : np.ndarray, shape (feature_dim,)
        seed : int
        blend_weight : float
            Weight given to the counterfactual mean.  Default 0.20.

        Returns
        -------
        float
            Adjusted P(team1 wins).
        """
        scenarios = self.generate(X1, X2, seed=seed)
        valid = [s for s in scenarios if s.is_valid]
        if not valid:
            base_prob = float(
                self._predict(
                    np.asarray(X1, dtype=float).reshape(1, -1),
                    np.asarray(X2, dtype=float).reshape(1, -1),
                )[0]
            )
            return base_prob
        base_prob = valid[0].base_prob
        cf_mean = float(np.mean([s.prob for s in valid]))
        return (1.0 - blend_weight) * base_prob + blend_weight * cf_mean
