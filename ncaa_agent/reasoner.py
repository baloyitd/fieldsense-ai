"""
ncaa_agent.reasoner
===================
Stage 07 — Structured reasoning engine for NCAA tournament matchup analysis.

NCAAReasoner produces 4-step reasoning traces for every matchup:

  Step 1 — Efficiency Analysis
      Compare ORTG, DRTG, net rating, and tempo.  The efficiency delta
      between teams is the single strongest predictor of tournament outcomes.

  Step 2 — Historical Patterns
      Seed matchup win rates and similar efficiency-profile comparisons.
      Grounds the prediction in historical tournament outcomes.

  Step 3 — Contextual Factors
      Recent form (L10 win%), strength of schedule (SOS), turnover rate,
      and assist rate capture late-season trajectory and team quality.

  Step 4 — Refined Probability
      Aggregates adjustments from steps 1–3, blends lightly with the
      ensemble probability, and computes a confidence score.

The reasoner is OFFLINE — it never touches the prediction hot path.
Results are cached via :class:`~ncaa_agent.cache.ResultCache`.

Usage
-----
::

    reasoner = NCAAReasoner()
    trace = reasoner.reason(X1, X2, ensemble_prob=0.82,
                            meta={"team1_name": "Kansas", "seed1": 1,
                                  "team2_name": "Howard", "seed2": 16})
    print(trace.steps[0].analysis)
"""

from __future__ import annotations

import datetime
import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ncaa_models.baseline import FEATURE_COLS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature index lookup (built once at module load)
# ---------------------------------------------------------------------------

_FI: Dict[str, int] = {c: i for i, c in enumerate(FEATURE_COLS)}

# ---------------------------------------------------------------------------
# Probability adjustment caps
# ---------------------------------------------------------------------------

_MAX_STEP_ADJ: float = 0.08   # per-step cap
_MAX_TOTAL_ADJ: float = 0.12  # total adjustment cap (blend stays close to ensemble)
_BLEND_WEIGHT: float = 0.20   # fraction of total adjustment applied to ensemble_prob

# ---------------------------------------------------------------------------
# Confidence thresholds
# ---------------------------------------------------------------------------

_HIGH_CONF: float = 0.30   # |prob - 0.5| > 0.30  → HIGH
_MED_CONF: float = 0.12    # |prob - 0.5| > 0.12  → MEDIUM, else LOW


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ReasoningStep:
    """One step in a 4-step reasoning trace."""

    step: int
    """Step number: 1–4."""

    name: str
    """Human-readable step label."""

    analysis: str
    """Prose explanation of this step's findings."""

    key_metrics: Dict[str, Any]
    """Numeric metrics surfaced by this step."""

    adjustment: float
    """Probability adjustment contributed by this step (in [-0.08, +0.08])."""

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ReasoningTrace:
    """
    Full 4-step reasoning trace for one matchup.

    Attributes
    ----------
    matchup_id : str
    team1_name, team2_name : str
    seed1, seed2 : int
    ensemble_prob : float
        Raw ensemble model probability for team1.
    adjusted_prob : float
        ensemble_prob lightly adjusted by reasoning steps.
    confidence : str
        One of ``'HIGH'``, ``'MEDIUM'``, ``'LOW'``.
    steps : list of ReasoningStep
        Exactly 4 steps.
    season : int or None
    metadata : dict
        Any extra key-value pairs passed through from ``meta``.
    timestamp : str
        ISO-8601 UTC timestamp of trace creation.
    """

    matchup_id: str
    team1_name: str
    team2_name: str
    seed1: int
    seed2: int
    ensemble_prob: float
    adjusted_prob: float
    confidence: str
    steps: List[ReasoningStep]
    season: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.datetime.utcnow().isoformat()
    )

    def to_dict(self) -> Dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> "ReasoningTrace":
        steps = [ReasoningStep(**s) for s in d.pop("steps", [])]
        return cls(**d, steps=steps)


# ---------------------------------------------------------------------------
# NCAAReasoner
# ---------------------------------------------------------------------------

class NCAAReasoner:
    """
    Offline reasoning engine for NCAA tournament matchups.

    Parameters
    ----------
    feature_cols : list of str, optional
        Feature column names.  Defaults to :data:`ncaa_models.baseline.FEATURE_COLS`.
    cache : ResultCache, optional
        If provided, completed traces are cached and retrieved on repeated calls.
    """

    STEP_NAMES = [
        "Efficiency Analysis",
        "Historical Patterns",
        "Contextual Factors",
        "Refined Probability",
    ]

    def __init__(
        self,
        feature_cols: Optional[List[str]] = None,
        cache=None,
    ) -> None:
        self.feature_cols = list(feature_cols or FEATURE_COLS)
        self.fi: Dict[str, int] = {c: i for i, c in enumerate(self.feature_cols)}
        self.cache = cache

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reason(
        self,
        X1: np.ndarray,
        X2: np.ndarray,
        ensemble_prob: float,
        meta: Optional[Dict[str, Any]] = None,
    ) -> ReasoningTrace:
        """
        Generate a 4-step reasoning trace for a single matchup.

        Parameters
        ----------
        X1, X2 : np.ndarray, shape (feature_dim,) or (1, feature_dim)
        ensemble_prob : float
            Pre-computed ensemble win probability for team1.
        meta : dict, optional
            Keys: ``team1_name``, ``team2_name``, ``seed1``, ``seed2``,
            ``matchup_id``, ``season``.

        Returns
        -------
        ReasoningTrace
        """
        X1 = np.asarray(X1, dtype=float).ravel()
        X2 = np.asarray(X2, dtype=float).ravel()
        ensemble_prob = float(ensemble_prob)

        meta = meta or {}
        team1_name = str(meta.get("team1_name", "Team A"))
        team2_name = str(meta.get("team2_name", "Team B"))
        seed1 = int(meta.get("seed1", self._extract_seed(X1)))
        seed2 = int(meta.get("seed2", self._extract_seed(X2)))
        matchup_id = str(meta.get("matchup_id", f"{team1_name}_vs_{team2_name}"))
        season = meta.get("season")

        # Cache lookup
        if self.cache is not None:
            cache_key = self.cache.make_key(
                X1.tolist(), X2.tolist(), ensemble_prob, seed1, seed2,
                team1_name, team2_name,
            )
            cached = self.cache.get(cache_key)
            if cached is not None:
                return ReasoningTrace.from_dict(cached)

        # Compute 4 steps
        step1 = self._step1_efficiency(X1, X2, team1_name, team2_name)
        step2 = self._step2_historical(X1, X2, seed1, seed2, team1_name, team2_name)
        step3 = self._step3_contextual(X1, X2, team1_name, team2_name)
        step4, adjusted_prob = self._step4_refined(
            ensemble_prob, [step1, step2, step3], seed1, seed2, team1_name
        )

        confidence = self._confidence(adjusted_prob)
        trace = ReasoningTrace(
            matchup_id=matchup_id,
            team1_name=team1_name,
            team2_name=team2_name,
            seed1=seed1,
            seed2=seed2,
            ensemble_prob=ensemble_prob,
            adjusted_prob=adjusted_prob,
            confidence=confidence,
            steps=[step1, step2, step3, step4],
            season=season,
            metadata={k: v for k, v in meta.items()
                       if k not in ("team1_name", "team2_name", "seed1", "seed2",
                                   "matchup_id", "season")},
        )

        if self.cache is not None:
            self.cache.set(cache_key, trace.to_dict())

        return trace

    def reason_batch(
        self,
        matchups: Sequence[Dict[str, Any]],
    ) -> List[ReasoningTrace]:
        """
        Generate reasoning traces for a list of matchups.

        Parameters
        ----------
        matchups : list of dict
            Each dict must have keys ``X1``, ``X2``, ``ensemble_prob``,
            and optionally ``meta``.

        Returns
        -------
        list of ReasoningTrace
        """
        traces = []
        for m in matchups:
            trace = self.reason(
                m["X1"],
                m["X2"],
                m["ensemble_prob"],
                meta=m.get("meta"),
            )
            traces.append(trace)
        return traces

    # ------------------------------------------------------------------
    # Step implementations
    # ------------------------------------------------------------------

    def _step1_efficiency(
        self,
        X1: np.ndarray,
        X2: np.ndarray,
        team1_name: str,
        team2_name: str,
    ) -> ReasoningStep:
        """Step 1: Compare offensive/defensive efficiency metrics."""
        fi = self.fi

        ortg1 = float(X1[fi["ortg"]])
        drtg1 = float(X1[fi["drtg"]])
        net1  = float(X1[fi["net_rtg"]])
        tempo1 = float(X1[fi["tempo"]])

        ortg2 = float(X2[fi["ortg"]])
        drtg2 = float(X2[fi["drtg"]])
        net2  = float(X2[fi["net_rtg"]])
        tempo2 = float(X2[fi["tempo"]])

        net_diff   = net1 - net2
        ortg_diff  = ortg1 - ortg2
        drtg_diff  = drtg2 - drtg1   # positive = team1 has better defence
        tempo_diff = tempo1 - tempo2

        # Adjustment: ±0.03 per 10 net_rtg point advantage
        adj = float(np.clip(0.003 * net_diff, -_MAX_STEP_ADJ, _MAX_STEP_ADJ))

        # Build analysis text
        net_adv = team1_name if net_diff > 0 else team2_name
        tempo_desc = (
            "faster pace" if tempo_diff > 2 else
            "slower pace" if tempo_diff < -2 else
            "similar tempo"
        )
        analysis = (
            f"{team1_name}: ORTG {ortg1:.1f}, DRTG {drtg1:.1f}, "
            f"NetRtg {net1:+.1f}  |  "
            f"{team2_name}: ORTG {ortg2:.1f}, DRTG {drtg2:.1f}, "
            f"NetRtg {net2:+.1f}.  "
            f"Net rating edge: {net_adv} by {abs(net_diff):.1f} pts.  "
            f"Tempo: {tempo1:.1f} vs {tempo2:.1f} ({tempo_desc})."
        )

        return ReasoningStep(
            step=1,
            name=self.STEP_NAMES[0],
            analysis=analysis,
            key_metrics={
                "ortg1": ortg1, "drtg1": drtg1, "net_rtg1": net1,
                "ortg2": ortg2, "drtg2": drtg2, "net_rtg2": net2,
                "net_rtg_diff": net_diff,
                "ortg_diff": ortg_diff,
                "drtg_diff": drtg_diff,
                "tempo1": tempo1, "tempo2": tempo2,
            },
            adjustment=adj,
        )

    def _step2_historical(
        self,
        X1: np.ndarray,
        X2: np.ndarray,
        seed1: int,
        seed2: int,
        team1_name: str,
        team2_name: str,
    ) -> ReasoningStep:
        """Step 2: Seed matchup historical win rates."""
        from ncaa_agent.anomaly import seed_baseline_win_rate

        sos1 = float(X1[self.fi["sos"]])
        sos2 = float(X2[self.fi["sos"]])
        win_pct1 = float(X1[self.fi["win_pct"]])
        win_pct2 = float(X2[self.fi["win_pct"]])

        # Historical baseline for team1 as lower seed
        hist_rate = seed_baseline_win_rate(seed1, seed2)
        seed_diff = seed2 - seed1

        # Adjustment: pull slightly toward historical rate
        hist_adj = float(np.clip(0.10 * (hist_rate - 0.5), -_MAX_STEP_ADJ, _MAX_STEP_ADJ))

        # Describe the seed advantage
        if abs(seed_diff) >= 5:
            upset_label = "major upset alert" if seed1 > seed2 else "significant favourite"
        elif abs(seed_diff) >= 3:
            upset_label = "moderate upset potential" if seed1 > seed2 else "moderate favourite"
        elif abs(seed_diff) >= 1:
            upset_label = "slight upset potential" if seed1 > seed2 else "slight favourite"
        else:
            upset_label = "pick-em"

        analysis = (
            f"Seed matchup: #{seed1} {team1_name} vs #{seed2} {team2_name} "
            f"({upset_label}).  "
            f"Historical win rate for #{min(seed1, seed2)} seed: "
            f"{hist_rate:.1%}.  "
            f"Season win%: {team1_name} {win_pct1:.0%} vs {team2_name} {win_pct2:.0%}.  "
            f"SOS: {team1_name} {sos1:.3f} vs {team2_name} {sos2:.3f}."
        )

        return ReasoningStep(
            step=2,
            name=self.STEP_NAMES[1],
            analysis=analysis,
            key_metrics={
                "seed1": seed1, "seed2": seed2,
                "seed_diff": seed_diff,
                "historical_win_rate": hist_rate,
                "win_pct1": win_pct1, "win_pct2": win_pct2,
                "sos1": sos1, "sos2": sos2,
            },
            adjustment=hist_adj,
        )

    def _step3_contextual(
        self,
        X1: np.ndarray,
        X2: np.ndarray,
        team1_name: str,
        team2_name: str,
    ) -> ReasoningStep:
        """Step 3: Recent form, turnover rate, assist rate, strength of schedule."""
        fi = self.fi

        l10_1 = float(X1[fi["win_pct_last10"]])
        l10_2 = float(X2[fi["win_pct_last10"]])
        tov1  = float(X1[fi["tov_rate"]])
        tov2  = float(X2[fi["tov_rate"]])
        ast1  = float(X1[fi["ast_rate"]])
        ast2  = float(X2[fi["ast_rate"]])
        fg3_1 = float(X1[fi["fg3_pct"]])
        fg3_2 = float(X2[fi["fg3_pct"]])

        l10_diff = l10_1 - l10_2
        tov_edge = tov2 - tov1   # positive = team1 takes care of ball better

        adj = float(np.clip(0.04 * l10_diff + 0.05 * tov_edge,
                            -_MAX_STEP_ADJ, _MAX_STEP_ADJ))

        form_label1 = "hot" if l10_1 >= 0.70 else "cold" if l10_1 <= 0.40 else "steady"
        form_label2 = "hot" if l10_2 >= 0.70 else "cold" if l10_2 <= 0.40 else "steady"

        analysis = (
            f"Recent form (L10): {team1_name} {l10_1:.0%} ({form_label1}) vs "
            f"{team2_name} {l10_2:.0%} ({form_label2}).  "
            f"Turnover rate: {team1_name} {tov1:.1%} vs {team2_name} {tov2:.1%}.  "
            f"Assist rate: {team1_name} {ast1:.1%} vs {team2_name} {ast2:.1%}.  "
            f"3-pt shooting: {team1_name} {fg3_1:.1%} vs {team2_name} {fg3_2:.1%}."
        )

        return ReasoningStep(
            step=3,
            name=self.STEP_NAMES[2],
            analysis=analysis,
            key_metrics={
                "win_pct_last10_1": l10_1, "win_pct_last10_2": l10_2,
                "l10_diff": l10_diff,
                "tov_rate1": tov1, "tov_rate2": tov2,
                "ast_rate1": ast1, "ast_rate2": ast2,
                "fg3_pct1": fg3_1, "fg3_pct2": fg3_2,
            },
            adjustment=adj,
        )

    def _step4_refined(
        self,
        ensemble_prob: float,
        prior_steps: List[ReasoningStep],
        seed1: int,
        seed2: int,
        team1_name: str,
    ) -> tuple:
        """Step 4: Aggregate adjustments, blend with ensemble, compute confidence."""
        total_adj = sum(s.adjustment for s in prior_steps)
        total_adj = float(np.clip(total_adj, -_MAX_TOTAL_ADJ, _MAX_TOTAL_ADJ))

        # Light blend: ensemble carries 80% of weight, reasoning 20%
        adjusted_prob = ensemble_prob + _BLEND_WEIGHT * total_adj
        adjusted_prob = float(np.clip(adjusted_prob, 0.01, 0.99))

        confidence = self._confidence(adjusted_prob)

        adj_desc = f"{total_adj:+.3f}"
        dir_label = "upward" if total_adj > 0 else "downward" if total_adj < 0 else "no"

        analysis = (
            f"Ensemble base probability: {ensemble_prob:.3f}.  "
            f"Reasoning adjustment: {adj_desc} ({dir_label} revision).  "
            f"Adjusted probability: {adjusted_prob:.3f}.  "
            f"Confidence: {confidence}."
        )

        step = ReasoningStep(
            step=4,
            name=self.STEP_NAMES[3],
            analysis=analysis,
            key_metrics={
                "ensemble_prob": ensemble_prob,
                "step1_adj": prior_steps[0].adjustment,
                "step2_adj": prior_steps[1].adjustment,
                "step3_adj": prior_steps[2].adjustment,
                "total_adjustment": total_adj,
                "adjusted_prob": adjusted_prob,
                "confidence": confidence,
            },
            adjustment=total_adj,
        )
        return step, adjusted_prob

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_seed(self, X: np.ndarray) -> int:
        """Extract seed from feature vector (defaults to 8 if unavailable)."""
        try:
            return max(1, int(round(float(X[self.fi["seed"]]))))
        except (IndexError, KeyError):
            return 8

    @staticmethod
    def _confidence(prob: float) -> str:
        dist = abs(prob - 0.5)
        if dist > _HIGH_CONF:
            return "HIGH"
        if dist > _MED_CONF:
            return "MEDIUM"
        return "LOW"
