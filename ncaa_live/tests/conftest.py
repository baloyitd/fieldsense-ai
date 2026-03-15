"""
conftest.py — shared fixtures for ncaa_live tests
===================================================

Uses the synthetic 8-team dataset from ncaa_models/tests/conftest.py
to provide feature DataFrames, trained models, and tournament simulators.
"""

from __future__ import annotations

import sys
import os
from typing import List

import numpy as np
import pandas as pd
import pytest

sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "../../ncaa_models/tests"),
)
from conftest import (  # noqa: E402
    make_feat_df,
    make_tourney_df,
    M_TEAM_IDS,
    TRAIN_SEASONS,
    VAL_SEASON,
    ALL_SEASONS,
    build_matchup_df,
)

from ncaa_models.baseline import FEATURE_COLS, LogisticBaseline
from ncaa_live.ingester import BoxScoreIngester, GameResult
from ncaa_live.simulator import BracketTeam, TournamentSimulator
from ncaa_live.recalibrator import RoundRecalibrator

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEASON = VAL_SEASON  # 2025
TEAM_IDS = M_TEAM_IDS  # 8 teams


# ---------------------------------------------------------------------------
# Data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def feat_df() -> pd.DataFrame:
    return make_feat_df(TEAM_IDS, ALL_SEASONS)


@pytest.fixture(scope="session")
def val_feat_df(feat_df) -> pd.DataFrame:
    return feat_df[feat_df["season"] == SEASON].reset_index(drop=True)


@pytest.fixture(scope="session")
def matchup_df(feat_df) -> pd.DataFrame:
    tourney_df = make_tourney_df(TEAM_IDS, ALL_SEASONS)
    return build_matchup_df(feat_df, tourney_df, FEATURE_COLS, is_tourney=True)


@pytest.fixture(scope="session")
def train_arrays(matchup_df):
    train = matchup_df[matchup_df["season"].isin(TRAIN_SEASONS)]
    X1 = np.stack(train["X_team1"].values)
    X2 = np.stack(train["X_team2"].values)
    y = train["y"].values.astype(int)
    return X1, X2, y


@pytest.fixture(scope="session")
def val_arrays(matchup_df):
    val = matchup_df[matchup_df["season"] == SEASON]
    X1 = np.stack(val["X_team1"].values)
    X2 = np.stack(val["X_team2"].values)
    y = val["y"].values.astype(int)
    return X1, X2, y


@pytest.fixture(scope="session")
def fitted_model(train_arrays) -> LogisticBaseline:
    X1, X2, y = train_arrays
    m = LogisticBaseline()
    m.fit(X1, X2, y)
    return m


# ---------------------------------------------------------------------------
# Ingester fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ingester(feat_df) -> BoxScoreIngester:
    return BoxScoreIngester(feat_df, feature_cols=FEATURE_COLS)


@pytest.fixture
def sample_result() -> GameResult:
    """Seed-1 team beats seed-8 team (expected outcome)."""
    return GameResult(
        game_id="2025_R1_G1",
        season=SEASON,
        round_num=1,
        team1_id=TEAM_IDS[0],  # seed-1
        team2_id=TEAM_IDS[7],  # seed-8
        team1_score=78,
        team2_score=55,
    )


@pytest.fixture
def upset_result() -> GameResult:
    """Seed-8 team beats seed-1 team (upset)."""
    return GameResult(
        game_id="2025_R1_G1_upset",
        season=SEASON,
        round_num=1,
        team1_id=TEAM_IDS[7],  # seed-8
        team2_id=TEAM_IDS[0],  # seed-1
        team1_score=72,
        team2_score=68,
    )


# ---------------------------------------------------------------------------
# Simulator fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def simulator(feat_df) -> TournamentSimulator:
    """8-team deterministic simulator for VAL_SEASON."""
    season_df = feat_df[feat_df["season"] == SEASON]
    teams = []
    for rank, tid in enumerate(TEAM_IDS):
        seed = rank + 1
        teams.append(BracketTeam(team_id=tid, seed=seed, name=str(tid)))
    return TournamentSimulator(
        teams=teams,
        feature_store=feat_df,
        season=SEASON,
        deterministic=True,
    )


@pytest.fixture
def fresh_simulator(feat_df) -> TournamentSimulator:
    """New 8-team simulator each test (function scope)."""
    season_df = feat_df[feat_df["season"] == SEASON]
    teams = [
        BracketTeam(team_id=tid, seed=rank + 1, name=str(tid))
        for rank, tid in enumerate(TEAM_IDS)
    ]
    return TournamentSimulator(
        teams=teams,
        feature_store=feat_df,
        season=SEASON,
        deterministic=True,
    )


# ---------------------------------------------------------------------------
# Recalibrator fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def recalibrator(fitted_model) -> RoundRecalibrator:
    return RoundRecalibrator(base_model=fitted_model, feature_cols=FEATURE_COLS)
