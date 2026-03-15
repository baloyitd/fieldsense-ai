"""
ncaa_live.simulator
===================
Stage 08 — Deterministic/stochastic tournament simulator for integration tests.

TournamentSimulator replays an N-team (power-of-2) single-elimination
bracket round by round, generating GameResult objects for each game.
Lower-seeded (better) teams win in deterministic mode; stochastic mode
samples outcomes from a logistic seed-difference model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from ncaa_live.ingester import GameResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BracketTeam
# ---------------------------------------------------------------------------

@dataclass
class BracketTeam:
    """A team participating in the simulated bracket."""

    team_id: int
    seed: int
    name: str = ""


# ---------------------------------------------------------------------------
# TournamentSimulator
# ---------------------------------------------------------------------------

class TournamentSimulator:
    """
    Single-elimination tournament simulator.

    Parameters
    ----------
    teams : list of BracketTeam
        Must be a power of 2 in count.  Sorted by seed ascending internally.
    feature_store : pd.DataFrame
        Feature data for all teams (used by downstream components).
    season : int
    deterministic : bool
        If True (default), the lower-seeded team always wins.
    rng_seed : int
        Random seed for stochastic mode.
    """

    def __init__(
        self,
        teams: List[BracketTeam],
        feature_store: pd.DataFrame,
        season: int = 2025,
        deterministic: bool = True,
        rng_seed: int = 42,
    ) -> None:
        n = len(teams)
        if n < 2 or (n & (n - 1)) != 0:
            raise ValueError(f"Number of teams must be a power of 2, got {n}")
        self.teams = sorted(teams, key=lambda t: t.seed)
        self.feature_store = feature_store
        self.season = season
        self.deterministic = deterministic
        self.rng = np.random.RandomState(rng_seed)
        self._seed_map = {t.team_id: t.seed for t in self.teams}
        self._completed_rounds: List[List[GameResult]] = []
        self._active_ids: List[int] = [t.team_id for t in self.teams]
        self._round_num: int = 0
        self._game_counter: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def simulate_round(self) -> List[GameResult]:
        """
        Advance the bracket by one round.

        Returns
        -------
        list of GameResult
        """
        if len(self._active_ids) < 2:
            raise RuntimeError("Tournament is already complete.")

        self._round_num += 1
        results: List[GameResult] = []
        winners: List[int] = []

        active = list(self._active_ids)
        n = len(active)
        # Seed the bracket: best vs worst, 2nd vs 2nd-worst, etc.
        for i in range(n // 2):
            t1 = active[i]
            t2 = active[n - 1 - i]
            result = self._play_game(t1, t2, self._round_num)
            results.append(result)
            winners.append(result.winner_id)

        self._active_ids = winners
        self._completed_rounds.append(results)
        logger.info(
            "Round %d: %d games, winners=%s", self._round_num, len(results), winners
        )
        return results

    def simulate_all_rounds(self) -> List[List[GameResult]]:
        """Run every round until the bracket is complete."""
        while len(self._active_ids) > 1:
            self.simulate_round()
        return list(self._completed_rounds)

    def simulate_round_upset(self) -> List[GameResult]:
        """
        Simulate one round where the HIGHER seed (underdog) always wins.

        Used to inject bad data for rollback testing.
        """
        if len(self._active_ids) < 2:
            raise RuntimeError("Tournament is already complete.")

        self._round_num += 1
        results: List[GameResult] = []
        winners: List[int] = []

        active = list(self._active_ids)
        n = len(active)
        for i in range(n // 2):
            t1 = active[i]
            t2 = active[n - 1 - i]
            # Force the higher-seeded (worse) team to win
            s1 = self._get_seed(t1)
            s2 = self._get_seed(t2)
            # Higher seed number = worse team; that team wins here
            if s1 > s2:
                winner, loser = t1, t2
            else:
                winner, loser = t2, t1
            self._game_counter += 1
            game_id = f"{self.season}_R{self._round_num}_G{self._game_counter}_upset"
            score_w = int(self.rng.randint(65, 85))
            score_l = int(self.rng.randint(50, score_w))
            result = GameResult(
                game_id=game_id,
                season=self.season,
                round_num=self._round_num,
                team1_id=t1,
                team2_id=t2,
                team1_score=score_w if winner == t1 else score_l,
                team2_score=score_w if winner == t2 else score_l,
            )
            results.append(result)
            winners.append(winner)

        self._active_ids = winners
        self._completed_rounds.append(results)
        logger.info(
            "Round %d (upset): %d games, winners=%s",
            self._round_num, len(results), winners,
        )
        return results

    def get_completed_rounds(self) -> List[List[GameResult]]:
        return list(self._completed_rounds)

    def get_all_results_flat(self) -> List[GameResult]:
        return [g for rnd in self._completed_rounds for g in rnd]

    def is_complete(self) -> bool:
        return len(self._active_ids) == 1

    def champion_id(self) -> Optional[int]:
        return self._active_ids[0] if self.is_complete() else None

    def n_rounds_completed(self) -> int:
        return len(self._completed_rounds)

    def active_team_ids(self) -> List[int]:
        return list(self._active_ids)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _play_game(self, t1_id: int, t2_id: int, round_num: int) -> GameResult:
        self._game_counter += 1
        game_id = f"{self.season}_R{round_num}_G{self._game_counter}"
        s1 = self._get_seed(t1_id)
        s2 = self._get_seed(t2_id)

        if self.deterministic:
            team1_wins = s1 <= s2
        else:
            seed_diff = s2 - s1
            prob_t1 = 1.0 / (1.0 + np.exp(-0.3 * seed_diff))
            team1_wins = bool(self.rng.random() < prob_t1)

        if team1_wins:
            score1 = int(self.rng.randint(65, 90))
            score2 = int(self.rng.randint(50, score1))
        else:
            score2 = int(self.rng.randint(65, 90))
            score1 = int(self.rng.randint(50, score2))

        return GameResult(
            game_id=game_id,
            season=self.season,
            round_num=round_num,
            team1_id=t1_id,
            team2_id=t2_id,
            team1_score=score1,
            team2_score=score2,
        )

    def _get_seed(self, team_id: int) -> int:
        return self._seed_map.get(team_id, 8)


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def make_simulator_from_feature_df(
    feature_df: pd.DataFrame,
    team_ids: Sequence[int],
    season: int = 2025,
    seed_col: str = "seed",
    deterministic: bool = True,
    rng_seed: int = 42,
) -> TournamentSimulator:
    """
    Build a TournamentSimulator from an existing feature DataFrame.

    Parameters
    ----------
    feature_df : pd.DataFrame
    team_ids : sequence of int
    season : int
    seed_col : str
    deterministic : bool

    Returns
    -------
    TournamentSimulator
    """
    season_df = feature_df[feature_df["season"] == season]
    teams: List[BracketTeam] = []
    for tid in team_ids:
        row = season_df[season_df["team_id"] == tid]
        if row.empty:
            seed = 8
        else:
            seed = max(1, int(round(float(row[seed_col].iloc[0]))))
        teams.append(BracketTeam(team_id=tid, seed=seed, name=str(tid)))

    return TournamentSimulator(
        teams=teams,
        feature_store=feature_df,
        season=season,
        deterministic=deterministic,
        rng_seed=rng_seed,
    )
