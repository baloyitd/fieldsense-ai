"""
conftest.py — shared fixtures for ncaa_submission_2026 tests
=============================================================
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make the root importable
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ncaa_models" / "tests"))

from conftest import (  # noqa: E402 — ncaa_models test conftest
    make_feat_df,
    make_tourney_df,
    M_TEAM_IDS,
    TRAIN_SEASONS,
    VAL_SEASON,
    ALL_SEASONS,
    build_matchup_df,
)

from ncaa_models.baseline import FEATURE_COLS, LogisticBaseline

SEASON = VAL_SEASON  # 2025
TEST_SEASON = 2026
TEAM_IDS = M_TEAM_IDS


@pytest.fixture(scope="session")
def feat_df() -> pd.DataFrame:
    return make_feat_df(TEAM_IDS, ALL_SEASONS)


@pytest.fixture(scope="session")
def train_df(feat_df) -> pd.DataFrame:
    """Training-only features (seasons 2021-2024)."""
    return feat_df[feat_df["season"].isin(TRAIN_SEASONS)].reset_index(drop=True)


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
