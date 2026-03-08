"""
conftest.py — shared fixtures for ncaa_models tests
====================================================

Synthetic dataset design
------------------------
8 teams per gender, organised into 4 quality tiers (2 teams each):

  Tier | Seeds | net_rtg | win_pct | Description
  -----|-------|---------|---------|------------
    0  |  1-2  |  +20/+15|  0.88/0.82 | Elite
    1  |  3-4  |   +9/+4 |  0.68/0.58 | Good
    2  |  5-6  |  -4/-9  |  0.43/0.35 | Average
    3  |  7-8  | -15/-20 |  0.25/0.18 | Weak

Men's  team IDs : 1001-1008 (tier 0 = 1001, 1002; tier 3 = 1007, 1008)
Women's team IDs: 3001-3008 (same tier structure)

Tournament structure (8-team single-elimination, 7 games/season):
  Round 1 (day 134): seed1 v seed8, seed2 v seed7, seed3 v seed6, seed4 v seed5
  Round 2 (day 136): seed1 v seed4, seed2 v seed3
  Final   (day 138): seed1 v seed2

Outcomes are DETERMINISTIC — lower seed always wins.  This guarantees
the integration test (Brier < 0.20) passes reliably with any seed-aware model.

Seasons: 2021-2025  |  Train: 2021-2024  |  Val: 2025
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
import pytest

from ncaa_models.baseline import FEATURE_COLS, LogisticBaseline
from ncaa_models.cv import build_matchup_df

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# IDs are scrambled so lower-rank (better) teams don't always have lower IDs.
# This ensures both y=0 and y=1 labels appear in training data (sklearn needs both).
# ranks:   [0,    1,    2,    3,    4,    5,    6,    7   ]
# IDs:     [1001, 1003, 1005, 1007, 1002, 1004, 1006, 1008]
# In R1 game (2,5): rank2(1005) beats rank5(1004); lower-ID=1004 loses → y=0
# In R1 game (3,4): rank3(1007) beats rank4(1002); lower-ID=1002 loses → y=0
M_TEAM_IDS: List[int] = [1001, 1003, 1005, 1007, 1002, 1004, 1006, 1008]
W_TEAM_IDS: List[int] = [3001, 3003, 3005, 3007, 3002, 3004, 3006, 3008]
ALL_SEASONS: List[int] = [2021, 2022, 2023, 2024, 2025]
TRAIN_SEASONS: List[int] = [2021, 2022, 2023, 2024]
VAL_SEASON: int = 2025

# Quality tier index for each team (index within M_TEAM_IDS / W_TEAM_IDS)
_TIER: List[int] = [0, 0, 1, 1, 2, 2, 3, 3]   # 8 teams, 4 tiers

# Seed assigned by quality tier position
_SEEDS: List[int] = [1, 2, 3, 4, 5, 6, 7, 8]

# Feature values by quality tier (finer-grained)
_NET_RTG   = [20.0, 15.0, 9.0, 4.0, -4.0, -9.0, -15.0, -20.0]
_ORTG      = [118.0, 115.0, 111.0, 107.0, 102.0,  98.0,  93.0,  88.0]
_DRTG      = [ 98.0,  99.5, 102.0, 103.0, 106.0, 107.0, 108.0, 110.0]
_WIN_PCT   = [0.88, 0.82, 0.68, 0.58, 0.43, 0.35, 0.25, 0.18]
_SOS       = [0.60, 0.58, 0.55, 0.52, 0.48, 0.45, 0.42, 0.40]
_EFG_PCT   = [0.56, 0.54, 0.52, 0.50, 0.48, 0.46, 0.44, 0.42]
_TOV_RATE  = [0.14, 0.15, 0.16, 0.17, 0.19, 0.20, 0.21, 0.22]

# Tournament bracket (index into team_ids list): (winner_idx, loser_idx)
_BRACKET = [
    (0, 7), (1, 6), (2, 5), (3, 4),   # Round 1
    (0, 3), (1, 2),                    # Round 2
    (0, 1),                            # Final
]
_ROUND_DAY = {0: 134, 1: 134, 2: 134, 3: 134, 4: 136, 5: 136, 6: 138}


# ---------------------------------------------------------------------------
# Feature generation
# ---------------------------------------------------------------------------

def _team_features(team_id: int, rank: int, season: int) -> dict:
    """
    Return a feature dict for a team at a given quality rank (0=best, 7=worst).
    All 28 FEATURE_COLS are populated; key discriminative features vary by rank.
    """
    ortg = _ORTG[rank]
    drtg = _DRTG[rank]
    net = _NET_RTG[rank]
    wp = _WIN_PCT[rank]
    tempo = 70.0 + rank * 0.5          # weaker teams play slightly slower
    ppg_s = ortg * tempo / 100.0
    ppg_a = drtg * tempo / 100.0

    return {
        "team_id": team_id,
        "season": season,
        # Core stats
        "games_played": 30,
        "win_pct": wp,
        "ppg_scored": ppg_s,
        "ppg_allowed": ppg_a,
        "scoring_margin": ppg_s - ppg_a,
        "home_win_pct": min(0.97, wp + 0.10),
        "away_win_pct": max(0.03, wp - 0.15),
        "neutral_win_pct": wp,
        "win_pct_last10": wp,
        # Shooting
        "fg_pct":  0.45 - rank * 0.005,
        "fg3_pct": 0.36 - rank * 0.005,
        "ft_pct":  0.72,
        "efg_pct": _EFG_PCT[rank],
        "ts_pct":  _EFG_PCT[rank] + 0.02,
        # Rebounding
        "orb_rate": 0.32 - rank * 0.01,
        "drb_rate": 0.72 - rank * 0.01,
        # Efficiency
        "possessions_pg": tempo,
        "ortg": ortg,
        "drtg": drtg,
        "net_rtg": net,
        "tempo": tempo,
        # Ball security / creation
        "tov_rate": _TOV_RATE[rank],
        "ast_rate": 0.58 - rank * 0.01,
        "blk_rate": 0.10 - rank * 0.005,
        "stl_rate": 0.08,
        "ot_rate":  0.05,
        # Tournament context
        "sos":  _SOS[rank],
        "seed": float(_SEEDS[rank]),
    }


def make_feat_df(team_ids: List[int], seasons: List[int]) -> pd.DataFrame:
    """Build feature DataFrame for a set of teams across multiple seasons."""
    rows = []
    for season in seasons:
        for rank, tid in enumerate(team_ids):
            rows.append(_team_features(tid, rank, season))
    df = pd.DataFrame(rows)
    # Ensure column order matches FEATURE_COLS
    return df[["team_id", "season"] + FEATURE_COLS]


def make_tourney_df(team_ids: List[int], seasons: List[int]) -> pd.DataFrame:
    """Build tournament game results for an 8-team bracket (deterministic outcomes)."""
    rows = []
    for season in seasons:
        for game_idx, (w_rank, l_rank) in enumerate(_BRACKET):
            w_id = team_ids[w_rank]
            l_id = team_ids[l_rank]
            day = _ROUND_DAY[game_idx]

            for winner, loser, won in [(w_id, l_id, True), (l_id, w_id, False)]:
                rows.append({
                    "season": season,
                    "team_id": winner,
                    "opp_team_id": loser,
                    "won": won,
                    "is_tourney": True,
                    "day_num": day,
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def feat_df() -> pd.DataFrame:
    """Feature matrix for M_TEAM_IDS across ALL_SEASONS."""
    return make_feat_df(M_TEAM_IDS, ALL_SEASONS)


@pytest.fixture(scope="session")
def tourney_df() -> pd.DataFrame:
    """Tournament game results for M_TEAM_IDS across ALL_SEASONS."""
    return make_tourney_df(M_TEAM_IDS, ALL_SEASONS)


@pytest.fixture(scope="session")
def matchup_df(feat_df, tourney_df) -> pd.DataFrame:
    """Pre-built matchup DataFrame for use with temporal_cross_validate."""
    return build_matchup_df(feat_df, tourney_df, FEATURE_COLS, is_tourney=True)


@pytest.fixture(scope="session")
def train_arrays(matchup_df):
    """(X1_train, X2_train, y_train) arrays for TRAIN_SEASONS."""
    train = matchup_df[matchup_df["season"].isin(TRAIN_SEASONS)]
    X1 = np.stack(train["X_team1"].values)
    X2 = np.stack(train["X_team2"].values)
    y = train["y"].values.astype(int)
    return X1, X2, y


@pytest.fixture(scope="session")
def val_arrays(matchup_df):
    """(X1_val, X2_val, y_val) arrays for VAL_SEASON."""
    val = matchup_df[matchup_df["season"] == VAL_SEASON]
    X1 = np.stack(val["X_team1"].values)
    X2 = np.stack(val["X_team2"].values)
    y = val["y"].values.astype(int)
    return X1, X2, y


@pytest.fixture(scope="session")
def fitted_model(train_arrays) -> LogisticBaseline:
    """LogisticBaseline trained on TRAIN_SEASONS data."""
    X1, X2, y = train_arrays
    model = LogisticBaseline()
    model.fit(X1, X2, y)
    return model


@pytest.fixture(scope="session")
def w_feat_df() -> pd.DataFrame:
    """Feature matrix for W_TEAM_IDS across ALL_SEASONS."""
    return make_feat_df(W_TEAM_IDS, ALL_SEASONS)
