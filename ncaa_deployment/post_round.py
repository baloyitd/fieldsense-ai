"""
ncaa_deployment.post_round
==========================
Stage 10 — Post-round analysis framework for live tournament monitoring.

After each tournament round completes, this module:
  1. Ingests actual game results.
  2. Computes Brier scores for that round's predictions.
  3. Compares conservative vs aggressive variant performance.
  4. Compares static (pre-tournament) vs live-adapted (Stage 08) performance.
  5. Generates a structured round summary report.

Usage
-----
::

    analyzer = PostRoundAnalyzer()
    analyzer.register_variant("conservative", conservative_submission)
    analyzer.register_variant("aggressive",   aggressive_submission)

    for round_num, (results, labels) in enumerate(round_data, 1):
        report = analyzer.analyze_round(round_num, results, labels)
        print(report.summary())
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# VariantComparison
# ---------------------------------------------------------------------------

@dataclass
class VariantComparison:
    """
    Brier comparison between two submission variants for one round.

    Attributes
    ----------
    round_num : int
    variant_a : str
    variant_b : str
    brier_a : float
    brier_b : float
    winner : str
        Variant name with lower Brier, or ``'tie'``.
    delta : float
        brier_b - brier_a  (negative → a is better)
    """

    round_num: int
    variant_a: str
    variant_b: str
    brier_a: float
    brier_b: float

    @property
    def delta(self) -> float:
        return self.brier_b - self.brier_a

    @property
    def winner(self) -> str:
        if abs(self.delta) < 1e-9:
            return "tie"
        return self.variant_a if self.brier_a <= self.brier_b else self.variant_b

    def to_dict(self) -> Dict[str, Any]:
        return {
            "round_num": self.round_num,
            "variant_a": self.variant_a,
            "variant_b": self.variant_b,
            "brier_a": self.brier_a,
            "brier_b": self.brier_b,
            "delta": self.delta,
            "winner": self.winner,
        }


# ---------------------------------------------------------------------------
# RoundReport
# ---------------------------------------------------------------------------

@dataclass
class RoundReport:
    """
    Summary of post-round analysis.

    Attributes
    ----------
    round_num : int
    n_games : int
    brier_by_variant : dict
        Maps variant name → Brier score.
    comparisons : list of VariantComparison
    cumulative_brier : dict
        Maps variant name → cumulative Brier across all rounds so far.
    details : str
    metadata : dict
    """

    round_num: int
    n_games: int
    brier_by_variant: Dict[str, float]
    comparisons: List[VariantComparison]
    cumulative_brier: Dict[str, float]
    details: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"=== Round {self.round_num} Report ({self.n_games} games) ===",
        ]
        for name, brier in sorted(self.brier_by_variant.items()):
            lines.append(f"  {name:<20}: Brier={brier:.4f}  (cumulative={self.cumulative_brier.get(name, brier):.4f})")
        for cmp in self.comparisons:
            lines.append(f"  {cmp.variant_a} vs {cmp.variant_b}: winner={cmp.winner}  delta={cmp.delta:+.4f}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "round_num": self.round_num,
            "n_games": self.n_games,
            "brier_by_variant": self.brier_by_variant,
            "comparisons": [c.to_dict() for c in self.comparisons],
            "cumulative_brier": self.cumulative_brier,
            "details": self.details,
        }


# ---------------------------------------------------------------------------
# PostRoundAnalyzer
# ---------------------------------------------------------------------------

class PostRoundAnalyzer:
    """
    Tracks per-round predictions and computes post-round Brier analysis.

    Parameters
    ----------
    season : int
    """

    def __init__(self, season: int = 2025) -> None:
        self.season = season
        # Maps variant_name → SubmissionVariant (or DataFrame)
        self._variants: Dict[str, Any] = {}
        # Accumulates results: list of (matchup_id, y_true) per round
        self._round_results: List[Tuple[int, List[str], List[int]]] = []
        # Per-variant cumulative data
        self._cumulative_preds: Dict[str, List[float]] = {}
        self._cumulative_labels: List[int] = []
        # History of round reports
        self._reports: List[RoundReport] = []

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_variant(self, name: str, submission) -> None:
        """
        Register a submission variant for tracking.

        Parameters
        ----------
        name : str
            Unique name (e.g. ``'conservative'``, ``'aggressive'``).
        submission : SubmissionVariant or pd.DataFrame
            DataFrame must have ``ID`` and ``Pred`` columns.
        """
        self._variants[name] = submission
        self._cumulative_preds[name] = []
        logger.info("Registered variant '%s' with %d predictions.", name, len(self._get_df(name)))

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def analyze_round(
        self,
        round_num: int,
        matchup_ids: Sequence[str],
        y_true: Sequence[int],
    ) -> RoundReport:
        """
        Compute per-round Brier scores for all registered variants.

        Parameters
        ----------
        round_num : int
            Tournament round number (1 = first round).
        matchup_ids : sequence of str
            Kaggle-format IDs like ``'2025_1001_1008'``.
        y_true : sequence of int
            Actual outcomes (1 = team1 won, 0 = team2 won).

        Returns
        -------
        RoundReport
        """
        ids = list(matchup_ids)
        labels = list(y_true)
        n_games = len(labels)

        brier_by_variant: Dict[str, float] = {}
        for name in self._variants:
            df = self._get_df(name)
            preds = self._lookup_predictions(df, ids)
            self._cumulative_preds[name].extend(preds)
            brier_by_variant[name] = float(
                np.mean((np.asarray(preds) - np.asarray(labels, dtype=float)) ** 2)
            )

        self._cumulative_labels.extend(labels)

        # Cumulative Brier
        cumulative_brier: Dict[str, float] = {}
        for name, cpreds in self._cumulative_preds.items():
            if cpreds:
                all_p = np.asarray(cpreds, dtype=float)
                all_y = np.asarray(self._cumulative_labels[: len(cpreds)], dtype=float)
                cumulative_brier[name] = float(np.mean((all_p - all_y) ** 2))

        # Pairwise comparisons (all pairs)
        names = sorted(brier_by_variant.keys())
        comparisons: List[VariantComparison] = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                comparisons.append(VariantComparison(
                    round_num=round_num,
                    variant_a=a,
                    variant_b=b,
                    brier_a=brier_by_variant[a],
                    brier_b=brier_by_variant[b],
                ))

        details = self._build_details(round_num, brier_by_variant, matchup_ids, y_true)
        report = RoundReport(
            round_num=round_num,
            n_games=n_games,
            brier_by_variant=brier_by_variant,
            comparisons=comparisons,
            cumulative_brier=cumulative_brier,
            details=details,
            metadata={"season": self.season, "matchup_ids": ids},
        )
        self._round_results.append((round_num, ids, labels))
        self._reports.append(report)
        logger.info(
            "Round %d analysed: %d games, variants=%s",
            round_num, n_games, list(brier_by_variant.keys()),
        )
        return report

    def cumulative_brier(self, variant_name: str) -> Optional[float]:
        """Return cumulative Brier for a variant across all analysed rounds."""
        preds = self._cumulative_preds.get(variant_name, [])
        if not preds:
            return None
        n = len(preds)
        labels = self._cumulative_labels[:n]
        return float(np.mean(
            (np.asarray(preds, dtype=float) - np.asarray(labels, dtype=float)) ** 2
        ))

    def best_variant(self) -> Optional[str]:
        """Return the variant name with the lowest cumulative Brier."""
        scores = {
            name: self.cumulative_brier(name)
            for name in self._variants
            if self.cumulative_brier(name) is not None
        }
        if not scores:
            return None
        return min(scores, key=lambda k: scores[k])

    def history(self) -> List[RoundReport]:
        return list(self._reports)

    def n_rounds_analyzed(self) -> int:
        return len(self._reports)

    def variant_names(self) -> List[str]:
        return list(self._variants.keys())

    def compute_round_brier(
        self,
        variant_name: str,
        matchup_ids: Sequence[str],
        y_true: Sequence[int],
    ) -> float:
        """
        Convenience: compute Brier for a single variant on a specific round's games.

        Parameters
        ----------
        variant_name : str
        matchup_ids : sequence of str
        y_true : sequence of int

        Returns
        -------
        float
        """
        df = self._get_df(variant_name)
        preds = self._lookup_predictions(df, list(matchup_ids))
        return float(
            np.mean((np.asarray(preds) - np.asarray(y_true, dtype=float)) ** 2)
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_df(self, name: str) -> pd.DataFrame:
        v = self._variants[name]
        if isinstance(v, pd.DataFrame):
            return v
        return v.df  # SubmissionVariant

    def _lookup_predictions(
        self, df: pd.DataFrame, ids: List[str]
    ) -> List[float]:
        """Return predictions for the given IDs (0.5 fallback if missing)."""
        id_to_pred: Dict[str, float] = dict(zip(df["ID"], df["Pred"]))
        preds = [float(id_to_pred.get(mid, 0.5)) for mid in ids]
        return preds

    def _build_details(
        self,
        round_num: int,
        brier_by_variant: Dict[str, float],
        matchup_ids: Sequence[str],
        y_true: Sequence[int],
    ) -> str:
        lines = [f"Round {round_num}: {len(y_true)} games"]
        for name, b in sorted(brier_by_variant.items()):
            lines.append(f"  {name}: Brier={b:.4f}")
        outcomes = dict(zip(matchup_ids, y_true))
        upsets = sum(1 for v in y_true if v == 0)
        lines.append(f"  Upsets this round: {upsets}/{len(y_true)}")
        return "\n".join(lines)
