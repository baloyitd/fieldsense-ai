"""
ncaa_model.matchup
==================
Build matchup feature matrices from team-season aggregate features.

Matchup representation
----------------------
A game between team A and team B is encoded as the element-wise *difference*
of their normalised aggregate feature vectors::

    x_matchup = features_A - features_B

The binary label is::

    y = 1  if team A wins
    y = 0  if team B wins

**Sign convention / Kaggle compatibility**

Kaggle submission IDs are ``"YYYY_TeamIdLow_TeamIdHigh"`` where the first
team (lower raw integer TeamID) is the one whose win probability we predict.
We therefore always define team A as the lower-ID team, so the model output
``p = predict(x_matchup)`` is the probability that the lower-ID team wins.

**Symmetry guarantee**

Because x_matchup flips sign when A and B are swapped::

    x(A, B) = -x(B, A)

any logistic model satisfies::

    P(A beats B) = 1 - P(B beats A)

automatically.

Features used
-------------
We use 14 of the 28 Stage-01 features — the subset with strongest predictive
signal for head-to-head matchups.  Compact-only pipelines (without detailed
box-score stats) will have NaN for shooting/efficiency columns; those columns
are median-imputed before subtraction.

``DEFAULT_FEATURE_COLS`` lists those 14 features.  Callers may override with
their own list.
"""

from __future__ import annotations

import logging
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default feature set for matchup model
# ---------------------------------------------------------------------------

#: 14 features chosen for maximum matchup predictive power.
#: All are signed so that higher = better for the team holding that value.
DEFAULT_FEATURE_COLS: List[str] = [
    # Efficiency
    "net_rtg",          # ORTG - DRTG: best single predictor of quality
    "ortg",             # offensive rating
    "drtg",             # defensive rating (lower is better → subtract is fine)
    "scoring_margin",   # PPG scored - PPG allowed
    # Win rate
    "win_pct",          # overall win percentage
    "win_pct_last10",   # form / recency
    "sos",              # strength of schedule (penalises easy-schedule teams)
    # Shooting
    "efg_pct",          # effective FG% — best single shooting metric
    "ft_pct",           # free-throw rate
    # Rebounding / ball-control
    "orb_rate",         # offensive rebound rate
    "drb_rate",         # defensive rebound rate
    "tov_rate",         # turnover rate (lower = better; sign flipped below)
    # Context
    "seed",             # tournament seed (1=best; sign flipped below)
    "games_played",     # proxy for experience / depth of schedule
]

# Features where a *lower* value is better for the team.
# When we compute feat_A - feat_B for these, we negate so the sign
# convention stays consistent (positive diff = team A is better).
_FLIP_SIGN_COLS = {"tov_rate", "drtg", "seed"}


# ---------------------------------------------------------------------------
# Feature helpers
# ---------------------------------------------------------------------------

def impute_features(
    feat_df: pd.DataFrame,
    feature_cols: List[str],
) -> pd.DataFrame:
    """
    Median-impute NaN values in *feature_cols* of *feat_df*.

    Imputation statistics are computed per (gender, season) group to avoid
    data leakage across seasons.

    Returns a copy of the DataFrame with NaNs filled.
    """
    df = feat_df.copy()
    for (gender, season), grp_idx in df.groupby(["gender", "season"]).groups.items():
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0.0
                continue
            col_data = df.loc[grp_idx, col]
            median_val = col_data.median()
            if pd.isna(median_val):
                median_val = 0.0
            df.loc[grp_idx, col] = col_data.fillna(median_val)
    return df


def matchup_vector(
    feat_a: pd.Series,
    feat_b: pd.Series,
    feature_cols: List[str],
) -> np.ndarray:
    """
    Compute the matchup feature vector ``x = feat_A - feat_B``.

    For features in ``_FLIP_SIGN_COLS`` (where lower is better) the sign
    is negated so the resulting vector is always "positive = team A is better".

    Parameters
    ----------
    feat_a, feat_b : pd.Series
        Feature vectors for each team, indexed by feature name.
    feature_cols : list[str]
        Ordered list of features to include.

    Returns
    -------
    np.ndarray  shape (len(feature_cols),)
    """
    diff = np.zeros(len(feature_cols), dtype=np.float64)
    for i, col in enumerate(feature_cols):
        a_val = float(feat_a.get(col, 0.0) or 0.0)
        b_val = float(feat_b.get(col, 0.0) or 0.0)
        delta = a_val - b_val
        if col in _FLIP_SIGN_COLS:
            delta = -delta
        diff[i] = delta
    return diff


# ---------------------------------------------------------------------------
# Training data builder
# ---------------------------------------------------------------------------

def build_training_data(
    feat_df: pd.DataFrame,
    tourney_df: pd.DataFrame,
    seasons: Iterable[int],
    feature_cols: List[str] = DEFAULT_FEATURE_COLS,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Build matchup feature matrix (X) and label vector (y) from tournament games.

    Parameters
    ----------
    feat_df : pd.DataFrame
        Team-season feature matrix from Stage 01 ``compute_features``.
        Must contain columns: gender, season, canonical_id, team_id + feature_cols.
    tourney_df : pd.DataFrame
        Normalised tournament game rows (``is_tourney=True``).
        Must contain: season, gender, canonical_id, opp_canonical_id,
        team_id, opp_team_id, won.
        Each game appears TWICE (winner row + loser row); we de-duplicate.
    seasons : iterable of int
        Seasons to include in training.
    feature_cols : list[str]
        Feature columns to use (default: DEFAULT_FEATURE_COLS).

    Returns
    -------
    X : np.ndarray  shape (n_matchups, len(feature_cols))
    y : np.ndarray  shape (n_matchups,)   dtype int  (1=lower_id_team wins)
    meta : pd.DataFrame
        One row per matchup with columns: season, gender,
        team_id_low, team_id_high, canonical_id_low, canonical_id_high,
        y (label).
    """
    season_list = list(seasons)
    feat_imp = impute_features(feat_df, feature_cols)

    # Index: (gender, season, canonical_id) → feature Series
    feat_index = (
        feat_imp.set_index(["gender", "season", "canonical_id"])
        [feature_cols]
    )

    # De-duplicate tournament games: keep only won=True rows
    tourn = tourney_df[
        tourney_df["season"].isin(season_list) & tourney_df["won"]
    ].copy()

    rows_X, rows_y, meta_rows = [], [], []

    for _, game in tourn.iterrows():
        season  = int(game["season"])
        gender  = str(game["gender"])
        cid_win = str(game["canonical_id"])
        cid_los = str(game["opp_canonical_id"])
        tid_win = int(game["team_id"])
        tid_los = int(game["opp_team_id"])

        key_win = (gender, season, cid_win)
        key_los = (gender, season, cid_los)

        if key_win not in feat_index.index or key_los not in feat_index.index:
            logger.debug("Missing features for %s vs %s in %s %s; skipping.",
                         cid_win, cid_los, gender, season)
            continue

        feat_win = feat_index.loc[key_win]
        feat_los = feat_index.loc[key_los]

        # Assign team A = lower raw TeamID (Kaggle convention)
        if tid_win <= tid_los:
            tid_low, tid_high = tid_win, tid_los
            cid_low, cid_high = cid_win, cid_los
            feat_a, feat_b = feat_win, feat_los
            label = 1   # lower-ID team won
        else:
            tid_low, tid_high = tid_los, tid_win
            cid_low, cid_high = cid_los, cid_win
            feat_a, feat_b = feat_los, feat_win
            label = 0   # higher-ID team won (lower-ID team lost)

        x = matchup_vector(feat_a, feat_b, feature_cols)
        rows_X.append(x)
        rows_y.append(label)
        meta_rows.append({
            "season": season, "gender": gender,
            "team_id_low": tid_low, "team_id_high": tid_high,
            "canonical_id_low": cid_low, "canonical_id_high": cid_high,
            "y": label,
        })

    if not rows_X:
        raise ValueError(
            f"No matchups found for seasons {season_list}. "
            "Check that tourney_df contains tournament game rows."
        )

    X = np.array(rows_X, dtype=np.float64)
    y = np.array(rows_y, dtype=int)
    meta = pd.DataFrame(meta_rows)

    logger.info("Built %d matchup training rows from seasons %s",
                len(X), season_list)
    return X, y, meta


# ---------------------------------------------------------------------------
# Prediction pairs builder
# ---------------------------------------------------------------------------

def build_prediction_pairs(
    feat_df: pd.DataFrame,
    team_ids: Iterable[int],
    gender: str,
    season: int,
    feature_cols: List[str] = DEFAULT_FEATURE_COLS,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Build all possible matchup pairs for a prediction season.

    Generates C(n, 2) pairs from *team_ids* — every possible combination
    of two teams, in lower-ID-first order.  This matches the Kaggle
    submission format where every pair must be predicted.

    Parameters
    ----------
    feat_df   : team-season features (output of compute_features).
    team_ids  : raw integer TeamIDs to include (tournament-eligible teams).
    gender    : ``'M'`` or ``'W'``.
    season    : prediction season (e.g. 2026).
    feature_cols : feature columns to use.

    Returns
    -------
    X    : np.ndarray  shape (n_pairs, len(feature_cols))
    meta : pd.DataFrame  columns: team_id_low, team_id_high,
                                  canonical_id_low, canonical_id_high,
                                  season, gender
    """
    team_ids_sorted = sorted(set(team_ids))
    feat_imp = impute_features(feat_df, feature_cols)

    # Season filter for the *most recent available season* (may be season-1
    # if the prediction season hasn't been played yet)
    avail_seasons = sorted(
        feat_imp[feat_imp["gender"] == gender]["season"].unique()
    )
    if season in avail_seasons:
        feature_season = season
    elif avail_seasons:
        feature_season = avail_seasons[-1]
        logger.info(
            "Season %d not in features; using %d instead.", season, feature_season
        )
    else:
        raise ValueError(f"No feature data available for gender={gender}")

    feat_index = (
        feat_imp[
            (feat_imp["gender"] == gender) &
            (feat_imp["season"] == feature_season)
        ]
        .set_index("team_id")[feature_cols]
    )

    rows_X, meta_rows = [], []

    for i, tid_low in enumerate(team_ids_sorted):
        for tid_high in team_ids_sorted[i + 1:]:
            if tid_low not in feat_index.index or tid_high not in feat_index.index:
                continue
            feat_a = feat_index.loc[tid_low]
            feat_b = feat_index.loc[tid_high]
            x = matchup_vector(feat_a, feat_b, feature_cols)
            rows_X.append(x)
            meta_rows.append({
                "season": season,
                "gender": gender,
                "team_id_low": tid_low,
                "team_id_high": tid_high,
                "canonical_id_low": f"{gender}_{tid_low}",
                "canonical_id_high": f"{gender}_{tid_high}",
            })

    if not rows_X:
        raise ValueError(
            f"No pairs built for gender={gender} season={season}. "
            "Check that team_ids are present in feat_df."
        )

    X = np.array(rows_X, dtype=np.float64)
    meta = pd.DataFrame(meta_rows)
    logger.info("Built %d prediction pairs for %s %s", len(X), gender, season)
    return X, meta
