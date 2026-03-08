"""
ncaa_models.cv
==============
Temporal cross-validation for NCAA tournament predictors.

Design
------
Expanding-window leave-one-season-out CV:
  Fold 1 : train 2021-2022 → validate 2023
  Fold 2 : train 2021-2023 → validate 2024
  Fold 3 : train 2021-2024 → validate 2025

CRITICAL: No data leakage — training set max season < validation season
is enforced by assertion in every fold.

Usage
-----
::

    def factory(X1, X2, y):
        m = LogisticBaseline()
        m.fit(X1, X2, y)
        return m

    matchup_df = build_matchup_df(feat_df, results_df, FEATURE_COLS)
    result = temporal_cross_validate(factory, matchup_df, val_seasons=[2023, 2024, 2025])
    print(result)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .base import MatchupPredictor
from .evaluate import compute_brier_score

logger = logging.getLogger(__name__)

# Type alias — factory receives (X_team1, X_team2, y) and returns a fitted model.
ModelFactory = Callable[
    [np.ndarray, np.ndarray, np.ndarray],
    MatchupPredictor,
]


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CVFold:
    """Result for a single temporal CV fold."""

    train_seasons: List[int]
    val_season: int
    brier_score: float
    n_val_games: int
    predictions: np.ndarray = field(repr=False)
    actuals: np.ndarray = field(repr=False)

    def __str__(self) -> str:
        return (
            f"CVFold(train={self.train_seasons} → val={self.val_season}): "
            f"Brier={self.brier_score:.4f}  n={self.n_val_games}"
        )


@dataclass
class CVResult:
    """Aggregate results across all temporal CV folds."""

    folds: List[CVFold]
    mean_brier: float
    std_brier: float

    def summary_df(self) -> pd.DataFrame:
        """Return per-fold + aggregate summary as a tidy DataFrame."""
        rows = [
            {
                "train_seasons": str(f.train_seasons),
                "val_season": f.val_season,
                "brier_score": f.brier_score,
                "n_val_games": f.n_val_games,
            }
            for f in self.folds
        ]
        df = pd.DataFrame(rows)
        if not df.empty:
            agg = pd.DataFrame([{
                "train_seasons": "ALL",
                "val_season": "mean",
                "brier_score": self.mean_brier,
                "n_val_games": int(df["n_val_games"].sum()),
            }])
            df = pd.concat([df, agg], ignore_index=True)
        return df

    def __str__(self) -> str:
        lines = [str(f) for f in self.folds]
        lines.append(
            f"Mean Brier: {self.mean_brier:.4f} ± {self.std_brier:.4f}"
        )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main CV function
# ---------------------------------------------------------------------------

def temporal_cross_validate(
    model_factory: ModelFactory,
    matchup_df: pd.DataFrame,
    val_seasons: Optional[List[int]] = None,
    min_train_seasons: int = 2,
) -> CVResult:
    """
    Leave-one-season-out temporal cross-validation with expanding training window.

    Parameters
    ----------
    model_factory : callable
        ``(X_team1, X_team2, y) -> fitted MatchupPredictor``
        Called fresh for each fold.
    matchup_df : pd.DataFrame
        One row per unique matchup (de-duplicated to winner perspective).
        Required columns: season, X_team1 (np.ndarray), X_team2 (np.ndarray), y (int).
    val_seasons : list of int
        Seasons to use as validation holdouts.
        Defaults to [2023, 2024, 2025] (matches specification).
    min_train_seasons : int
        Minimum number of distinct training seasons required to run a fold.

    Returns
    -------
    CVResult

    Raises
    ------
    ValueError
        If no folds complete (e.g. insufficient training data for all val_seasons).
    """
    if val_seasons is None:
        val_seasons = [2023, 2024, 2025]

    all_seasons = sorted(matchup_df["season"].unique())
    folds: List[CVFold] = []

    for val_season in sorted(val_seasons):
        # Expanding window: all available seasons strictly before val_season
        train_seasons = [s for s in all_seasons if s < val_season]

        if len(train_seasons) < min_train_seasons:
            logger.warning(
                "Skipping val=%d: only %d training season(s) available (min=%d).",
                val_season, len(train_seasons), min_train_seasons,
            )
            continue

        # CRITICAL: enforce no data leakage
        assert max(train_seasons) < val_season, (
            f"DATA LEAKAGE DETECTED: max training season {max(train_seasons)} "
            f">= validation season {val_season}"
        )

        train_df = matchup_df[matchup_df["season"].isin(train_seasons)]
        val_df = matchup_df[matchup_df["season"] == val_season]

        if val_df.empty:
            logger.warning("No validation data for season %d; skipping.", val_season)
            continue

        X1_train, X2_train, y_train = _extract_arrays(train_df)
        X1_val, X2_val, y_val = _extract_arrays(val_df)

        logger.info(
            "CV fold: train=%s (n=%d) → val=%d (n=%d)",
            train_seasons, len(y_train), val_season, len(y_val),
        )

        model = model_factory(X1_train, X2_train, y_train)
        preds = model.predict_proba(X1_val, X2_val)
        brier = compute_brier_score(preds, y_val)

        fold = CVFold(
            train_seasons=train_seasons,
            val_season=val_season,
            brier_score=brier,
            n_val_games=len(y_val),
            predictions=preds,
            actuals=y_val,
        )
        folds.append(fold)
        logger.info(str(fold))

    if not folds:
        raise ValueError(
            "No CV folds completed. Check val_seasons and data availability."
        )

    brier_scores = [f.brier_score for f in folds]
    return CVResult(
        folds=folds,
        mean_brier=float(np.mean(brier_scores)),
        std_brier=float(np.std(brier_scores)),
    )


# ---------------------------------------------------------------------------
# matchup_df builder
# ---------------------------------------------------------------------------

def build_matchup_df(
    feat_df: pd.DataFrame,
    results_df: pd.DataFrame,
    feature_cols: List[str],
    is_tourney: bool = True,
) -> pd.DataFrame:
    """
    Build the matchup DataFrame from a feature matrix and game results.

    Each row represents one unique tournament game, with:
    - team1 = lower TeamId (Kaggle convention)
    - team2 = higher TeamId
    - y = 1 if team1 (lower ID) won, 0 if team2 won

    Parameters
    ----------
    feat_df      : team-season features; columns: ['season', 'team_id'] + feature_cols.
    results_df   : game results; columns: ['season', 'team_id', 'opp_team_id', 'won'].
                   If 'is_tourney' column is present and is_tourney=True, only
                   tournament games are used.
    feature_cols : ordered list of feature column names.
    is_tourney   : if True and 'is_tourney' column exists, filter to tourney games only.

    Returns
    -------
    pd.DataFrame with columns:
        season, team_id_low, team_id_high, X_team1 (np.ndarray), X_team2 (np.ndarray), y (int)
    """
    # Filter to tournament games if requested
    if is_tourney and "is_tourney" in results_df.columns:
        games = results_df[results_df["is_tourney"]].copy()
    else:
        games = results_df.copy()

    # De-duplicate: keep only winner rows
    if "won" in games.columns:
        games = games[games["won"]].copy()

    feat_index = feat_df.set_index(["season", "team_id"])[feature_cols]

    rows = []
    for _, game in games.iterrows():
        season = int(game["season"])
        tid_w = int(game["team_id"])
        tid_l = int(game["opp_team_id"])

        key_w = (season, tid_w)
        key_l = (season, tid_l)

        if key_w not in feat_index.index or key_l not in feat_index.index:
            logger.debug(
                "Missing features for %d or %d in season %d; skipping.",
                tid_w, tid_l, season,
            )
            continue

        # Lower-ID = team1 (Kaggle convention)
        if tid_w <= tid_l:
            tid_low, tid_high = tid_w, tid_l
            X1 = feat_index.loc[key_w].values.astype(float)
            X2 = feat_index.loc[key_l].values.astype(float)
            y = 1  # lower-ID team won
        else:
            tid_low, tid_high = tid_l, tid_w
            X1 = feat_index.loc[key_l].values.astype(float)
            X2 = feat_index.loc[key_w].values.astype(float)
            y = 0  # lower-ID team lost

        rows.append({
            "season": season,
            "team_id_low": tid_low,
            "team_id_high": tid_high,
            "X_team1": X1,
            "X_team2": X2,
            "y": y,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_arrays(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract (X_team1, X_team2, y) numpy arrays from a matchup DataFrame."""
    X1 = np.stack(df["X_team1"].values)
    X2 = np.stack(df["X_team2"].values)
    y = df["y"].values.astype(int)
    return X1, X2, y
