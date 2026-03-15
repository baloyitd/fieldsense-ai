"""
conftest.py — shared fixtures for ncaa_deployment tests
========================================================
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pytest

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

SEASON = VAL_SEASON          # 2025
TEAM_IDS: List[int] = M_TEAM_IDS


# ---------------------------------------------------------------------------
# Constant-prediction model (always 0.5) — for optimizer contrast
# ---------------------------------------------------------------------------

class ConstantModel:
    """Always predicts 0.5 — used to create a weak component for ensemble tests."""

    def fit(self, X1, X2, y):
        return self

    def predict_proba(self, X1, X2):
        return np.full(len(X1), 0.5)


# ---------------------------------------------------------------------------
# Data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def feat_df() -> pd.DataFrame:
    return make_feat_df(TEAM_IDS, ALL_SEASONS)


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
def strong_model(train_arrays) -> LogisticBaseline:
    X1, X2, y = train_arrays
    m = LogisticBaseline(C=1.0)
    m.fit(X1, X2, y)
    return m


@pytest.fixture(scope="session")
def weak_model(train_arrays) -> LogisticBaseline:
    """Heavily regularized — predictions near 0.5."""
    X1, X2, y = train_arrays
    m = LogisticBaseline(C=0.0001)
    m.fit(X1, X2, y)
    return m


@pytest.fixture(scope="session")
def component_models() -> list:
    """Three *unfitted* component models for optimizer tests."""
    return [
        LogisticBaseline(C=1.0),      # strong
        LogisticBaseline(C=0.0001),   # weak (heavy regularization → ~0.5 preds)
        ConstantModel(),              # always 0.5 (worst)
    ]
