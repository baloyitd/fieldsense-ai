"""
ncaa_live.ingester
==================
Stage 08 — Box score ingestion and live feature-store updates.

BoxScoreIngester accepts completed-game results as they arrive during the
tournament and maintains an up-to-date feature DataFrame for downstream
recalibration and submission regeneration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ncaa_models.baseline import FEATURE_COLS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GameResult dataclass
# ---------------------------------------------------------------------------

@dataclass
class GameResult:
    """
    One completed tournament game.

    Parameters
    ----------
    game_id : str
    season : int
    round_num : int
        1 = Round of 64, 2 = Round of 32, 3 = Sweet 16, …
    team1_id, team2_id : int
    team1_score, team2_score : int
    box_scores : dict, optional
        Per-team box-score overrides keyed by team_id.
        Example: ``{1001: {"fg_pct": 0.52, "tov_rate": 0.14}}``.
    """

    game_id: str
    season: int
    round_num: int
    team1_id: int
    team2_id: int
    team1_score: int
    team2_score: int
    box_scores: Dict[int, Dict[str, float]] = field(default_factory=dict)

    @property
    def winner_id(self) -> int:
        return self.team1_id if self.team1_score > self.team2_score else self.team2_id

    @property
    def loser_id(self) -> int:
        return self.team2_id if self.team1_score > self.team2_score else self.team1_id

    @property
    def team1_won(self) -> bool:
        return self.team1_score > self.team2_score

    def to_dict(self) -> dict:
        return {
            "game_id": self.game_id,
            "season": self.season,
            "round_num": self.round_num,
            "team1_id": self.team1_id,
            "team2_id": self.team2_id,
            "team1_score": self.team1_score,
            "team2_score": self.team2_score,
            "winner_id": self.winner_id,
            "loser_id": self.loser_id,
        }


# ---------------------------------------------------------------------------
# BoxScoreIngester
# ---------------------------------------------------------------------------

class BoxScoreIngester:
    """
    Ingests completed game results and updates the feature store.

    Parameters
    ----------
    feature_store : pd.DataFrame
        Columns: ``['team_id', 'season'] + feature_cols``.
        Copied internally; caller's DataFrame is not mutated.
    feature_cols : list of str, optional
    """

    def __init__(
        self,
        feature_store: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
    ) -> None:
        self.feature_cols = list(feature_cols or FEATURE_COLS)
        self._store: pd.DataFrame = feature_store.copy()
        self._completed: List[GameResult] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest(self, result: GameResult) -> None:
        """
        Ingest one game result.

        Updates rolling season statistics (win_pct, scoring_margin,
        ppg_scored, ppg_allowed, win_pct_last10) for both teams.
        Also applies any box-score overrides in ``result.box_scores``.
        """
        self._completed.append(result)
        self._update_team_stats(
            result, result.team1_id,
            scored=result.team1_score, allowed=result.team2_score,
            won=result.team1_won,
        )
        self._update_team_stats(
            result, result.team2_id,
            scored=result.team2_score, allowed=result.team1_score,
            won=not result.team1_won,
        )
        for tid, stats in result.box_scores.items():
            self.update_feature_store(tid, stats, season=result.season)

        logger.info(
            "Ingested %s: %d (%d) vs %d (%d) → winner %d",
            result.game_id,
            result.team1_id, result.team1_score,
            result.team2_id, result.team2_score,
            result.winner_id,
        )

    def ingest_batch(self, results: Sequence[GameResult]) -> None:
        """Ingest multiple results in order."""
        for r in results:
            self.ingest(r)

    def update_feature_store(
        self,
        team_id: int,
        stats: Dict[str, float],
        season: Optional[int] = None,
    ) -> None:
        """
        Directly overwrite arbitrary feature values for a team.

        Parameters
        ----------
        team_id : int
        stats : dict
        season : int, optional
            If None, updates all rows for this team regardless of season.
        """
        mask = self._store["team_id"] == team_id
        if season is not None:
            mask &= self._store["season"] == season
        if not mask.any():
            logger.warning("team_id=%d not found in feature store", team_id)
            return
        for col, val in stats.items():
            if col in self._store.columns:
                self._store.loc[mask, col] = float(val)

    def get_completed_games(self) -> List[GameResult]:
        return list(self._completed)

    def get_updated_features(self) -> pd.DataFrame:
        return self._store.copy()

    def get_winner_ids(self) -> List[int]:
        return [r.winner_id for r in self._completed]

    def get_loser_ids(self) -> List[int]:
        return [r.loser_id for r in self._completed]

    def completed_matchup_ids(self, season: Optional[int] = None) -> List[str]:
        """Return Kaggle-format submission IDs for every completed game."""
        from ncaa_models.submit import make_submission_id
        ids = []
        for r in self._completed:
            s = season if season is not None else r.season
            lo = min(r.team1_id, r.team2_id)
            hi = max(r.team1_id, r.team2_id)
            ids.append(make_submission_id(s, lo, hi))
        return ids

    def build_training_arrays(
        self,
        season: Optional[int] = None,
    ):
        """
        Convert ingested game results into (X1, X2, y) numpy arrays.

        Parameters
        ----------
        season : int, optional
            If provided, filter to games of this season only.

        Returns
        -------
        X1, X2 : np.ndarray, shape (n, n_features)
        y : np.ndarray, shape (n,)
        """
        results = self._completed
        if season is not None:
            results = [r for r in results if r.season == season]

        if not results:
            n_feat = len(self.feature_cols)
            return np.empty((0, n_feat)), np.empty((0, n_feat)), np.array([])

        store = self._store
        X1_rows, X2_rows, ys = [], [], []

        for r in results:
            feat1 = self._lookup_features(store, r.team1_id, r.season)
            feat2 = self._lookup_features(store, r.team2_id, r.season)
            if feat1 is None or feat2 is None:
                continue
            X1_rows.append(feat1)
            X2_rows.append(feat2)
            ys.append(1 if r.team1_won else 0)

        if not X1_rows:
            n_feat = len(self.feature_cols)
            return np.empty((0, n_feat)), np.empty((0, n_feat)), np.array([])

        return (
            np.array(X1_rows, dtype=float),
            np.array(X2_rows, dtype=float),
            np.array(ys, dtype=int),
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _lookup_features(
        self,
        store: pd.DataFrame,
        team_id: int,
        season: int,
    ) -> Optional[np.ndarray]:
        """Return feature vector for a team in a given season."""
        mask = (store["team_id"] == team_id) & (store["season"] == season)
        rows = store.loc[mask, self.feature_cols]
        if rows.empty:
            # Fallback: any season for this team
            rows = store.loc[store["team_id"] == team_id, self.feature_cols]
        if rows.empty:
            return None
        return rows.iloc[0].values.astype(float)

    def _update_team_stats(
        self,
        result: GameResult,
        team_id: int,
        scored: int,
        allowed: int,
        won: bool,
    ) -> None:
        """Update rolling statistics for one team after a game."""
        mask = (
            (self._store["team_id"] == team_id)
            & (self._store["season"] == result.season)
        )
        if not mask.any():
            return

        gp = float(self._store.loc[mask, "games_played"].iloc[0])
        win_pct = float(self._store.loc[mask, "win_pct"].iloc[0])
        ppg_s = float(self._store.loc[mask, "ppg_scored"].iloc[0])
        ppg_a = float(self._store.loc[mask, "ppg_allowed"].iloc[0])
        l10 = float(self._store.loc[mask, "win_pct_last10"].iloc[0])

        new_gp = gp + 1
        new_wins = win_pct * gp + (1 if won else 0)
        new_win_pct = new_wins / new_gp
        new_ppg_s = (ppg_s * gp + scored) / new_gp
        new_ppg_a = (ppg_a * gp + allowed) / new_gp
        # Exponential moving average for recent form
        new_l10 = 0.9 * l10 + 0.1 * (1.0 if won else 0.0)

        updates = {
            "games_played": new_gp,
            "win_pct": new_win_pct,
            "ppg_scored": new_ppg_s,
            "ppg_allowed": new_ppg_a,
            "scoring_margin": new_ppg_s - new_ppg_a,
            "win_pct_last10": new_l10,
        }
        for col, val in updates.items():
            if col in self._store.columns:
                self._store.loc[mask, col] = val
