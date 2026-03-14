"""
conftest.py — shared fixtures for ncaa_agent tests
====================================================

Re-uses the synthetic 8-team dataset from ncaa_models/tests/conftest.py
to build feature vectors and ReasoningTraces suitable for testing the
NCAAReasoner, AnomalyDetector, BracketNarrativeGenerator, and ResultCache.
"""

from __future__ import annotations

import sys
import os
from typing import List

import numpy as np
import pytest

from ncaa_models.baseline import FEATURE_COLS

# Import helpers from ncaa_models test conftest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../ncaa_models/tests"))
from conftest import (  # noqa: E402
    make_feat_df,
    make_tourney_df,
    M_TEAM_IDS,
    TRAIN_SEASONS,
    VAL_SEASON,
    _team_features,
    _SEEDS,
)

from ncaa_agent.reasoner import NCAAReasoner, ReasoningTrace, ReasoningStep
from ncaa_agent.anomaly import AnomalyDetector, AnomalyFlag
from ncaa_agent.cache import ResultCache


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# 8 teams, ranks 0 (best) to 7 (worst)
N_TEAMS = 8
SEED_PAIRS = [(1, 16), (2, 15), (3, 14), (4, 13), (5, 12), (6, 11), (7, 10), (8, 9)]


# ---------------------------------------------------------------------------
# Feature vector helpers
# ---------------------------------------------------------------------------

def make_feature_vector(rank: int, season: int = 2025) -> np.ndarray:
    """Return a (28,) feature vector for a team of the given rank."""
    # Use team_id = 1000 + rank (arbitrary; not used in reasoner)
    feats = _team_features(1000 + rank, rank, season)
    return np.array([feats[c] for c in FEATURE_COLS], dtype=float)


def make_trace(
    seed1: int = 1,
    seed2: int = 8,
    ensemble_prob: float = 0.75,
    team1_name: str = "TeamA",
    team2_name: str = "TeamB",
    matchup_id: str = "matchup_001",
) -> ReasoningTrace:
    """Build a minimal ReasoningTrace for testing (no reasoner invocation)."""
    rank1 = seed1 - 1
    rank2 = min(seed2 - 1, N_TEAMS - 1)
    X1 = make_feature_vector(rank1)
    X2 = make_feature_vector(rank2)
    reasoner = NCAAReasoner()
    return reasoner.reason(
        X1, X2, ensemble_prob,
        meta={
            "team1_name": team1_name,
            "team2_name": team2_name,
            "seed1": seed1,
            "seed2": seed2,
            "matchup_id": matchup_id,
        },
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def reasoner() -> NCAAReasoner:
    return NCAAReasoner()


@pytest.fixture(scope="session")
def sample_X1() -> np.ndarray:
    """Feature vector for rank-0 (seed-1) team."""
    return make_feature_vector(0)


@pytest.fixture(scope="session")
def sample_X2() -> np.ndarray:
    """Feature vector for rank-7 (seed-8) team."""
    return make_feature_vector(7)


@pytest.fixture(scope="session")
def sample_trace(sample_X1, sample_X2) -> ReasoningTrace:
    """Pre-computed trace for seed 1 vs seed 8 (p=0.90)."""
    r = NCAAReasoner()
    return r.reason(
        sample_X1, sample_X2, 0.90,
        meta={
            "team1_name": "Kansas",
            "team2_name": "Howard",
            "seed1": 1,
            "seed2": 8,
            "matchup_id": "KU_HU_2025",
            "season": 2025,
        },
    )


@pytest.fixture(scope="session")
def traces_63() -> List[ReasoningTrace]:
    """
    63 synthetic traces — 8 first-round games × 4 seeds + extras.
    Seeds are taken from 8 first-round pairings; remaining are filler.
    """
    seed_pairs_all = [
        (1, 16), (2, 15), (3, 14), (4, 13),
        (5, 12), (6, 11), (7, 10), (8, 9),
    ]
    reasoner = NCAAReasoner()
    traces = []
    for i in range(63):
        s1, s2 = seed_pairs_all[i % len(seed_pairs_all)]
        rank1 = min(s1 - 1, N_TEAMS - 1)
        rank2 = min(s2 - 1, N_TEAMS - 1)
        X1 = make_feature_vector(rank1)
        X2 = make_feature_vector(rank2)
        prob = 0.5 + 0.02 * (s2 - s1)  # higher seed diff → higher p
        prob = float(np.clip(prob, 0.35, 0.95))
        t = reasoner.reason(
            X1, X2, prob,
            meta={
                "team1_name": f"Team{s1}_{i}",
                "team2_name": f"Team{s2}_{i}",
                "seed1": s1,
                "seed2": s2,
                "matchup_id": f"matchup_{i:03d}",
                "season": 2025,
                "round": (i // 16) + 1,
            },
        )
        traces.append(t)
    return traces


@pytest.fixture()
def tmp_cache(tmp_path) -> ResultCache:
    return ResultCache(cache_dir=tmp_path / "cache")
