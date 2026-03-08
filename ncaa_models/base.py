"""
ncaa_models.base
================
Abstract base class for all NCAA tournament matchup predictors.

All future models (stages 03-10) must subclass MatchupPredictor and
implement both abstract methods.

Kaggle convention
-----------------
team1 = lower TeamId, team2 = higher TeamId.
y = 1 if team1 (lower ID) wins, y = 0 if team2 (higher ID) wins.
"""

from abc import ABC, abstractmethod

import numpy as np


class MatchupPredictor(ABC):
    """
    Standard interface for NCAA tournament matchup prediction models.

    Subclasses receive raw per-team feature vectors for both teams and
    must internally compute whatever representation is needed (e.g.
    differential, concatenation, attention).

    The interface deliberately keeps raw vectors separate so implementors
    can choose their own feature interaction strategy.
    """

    @abstractmethod
    def fit(
        self,
        X_team1: np.ndarray,
        X_team2: np.ndarray,
        y: np.ndarray,
    ) -> None:
        """
        Train on historical matchup data.

        Parameters
        ----------
        X_team1 : np.ndarray, shape (n_matchups, n_features)
            Feature matrix for the lower-TeamId team in each matchup.
        X_team2 : np.ndarray, shape (n_matchups, n_features)
            Feature matrix for the higher-TeamId team in each matchup.
        y : np.ndarray, shape (n_matchups,)
            Binary labels: 1 if team1 (lower ID) wins, 0 if team2 wins.
        """

    @abstractmethod
    def predict_proba(
        self,
        X_team1: np.ndarray,
        X_team2: np.ndarray,
    ) -> np.ndarray:
        """
        Return P(team1 wins) for each matchup.

        Parameters
        ----------
        X_team1 : np.ndarray, shape (n_matchups, n_features)
        X_team2 : np.ndarray, shape (n_matchups, n_features)

        Returns
        -------
        np.ndarray, shape (n_matchups,), dtype float, values in [0, 1].
        """
