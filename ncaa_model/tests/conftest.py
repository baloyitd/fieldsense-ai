"""
conftest.py — shared pytest fixtures for ncaa_model test suite.

Synthetic data design
---------------------
We simulate 16 teams (quality tier 1-4, 4 teams each) across 5 seasons
(2021-2025).  Higher-quality teams have better feature values so logistic
regression has genuine signal to learn.

Feature values per tier (used for synthetic regular-season aggregates):
  Tier 1 (elite):      net_rtg=+18, win_pct=0.82, seed=2,  sos=0.58
  Tier 2 (good):       net_rtg=+8,  win_pct=0.65, seed=5,  sos=0.52
  Tier 3 (average):    net_rtg=-3,  win_pct=0.48, seed=10, sos=0.46
  Tier 4 (weak):       net_rtg=-15, win_pct=0.30, seed=14, sos=0.40

Tournament results: the better-seeded (lower tier number) team wins with
probability ~0.75 (realistic upset rate), with a deterministic seed set via
team_id so results are reproducible.
"""

from __future__ import annotations

import random
from typing import Dict, Generator, List

import numpy as np
import pandas as pd
import pytest

from ncaa_data.features import FEATURE_COLS, KEY_COLS
from ncaa_model.matchup import DEFAULT_FEATURE_COLS

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEASONS = list(range(2021, 2026))
TRAIN_SEASONS = list(range(2021, 2025))  # 2021-2024
VAL_SEASON = 2025

# 16 teams: IDs 1001-1016 (men's) and 3001-3016 (women's)
N_TEAMS = 16
M_TEAM_IDS = list(range(1001, 1001 + N_TEAMS))
W_TEAM_IDS = list(range(3001, 3001 + N_TEAMS))

# Quality tiers (4 teams each, ordered by tier 1=best)
TIER_SIZES = [4, 4, 4, 4]
TIER_STATS = [
    # net_rtg, win_pct, seed, sos, ortg, drtg, scoring_margin, efg_pct, ft_pct, orb_rate, drb_rate, tov_rate, win_pct_last10, games_played
    dict(net_rtg=18.0, win_pct=0.82, seed=2,  sos=0.58, ortg=112.0, drtg=94.0,
         scoring_margin=12.0, efg_pct=0.53, ft_pct=0.76, orb_rate=0.33,
         drb_rate=0.72, tov_rate=0.135, win_pct_last10=0.85, games_played=30),
    dict(net_rtg=8.0,  win_pct=0.65, seed=5,  sos=0.52, ortg=108.0, drtg=100.0,
         scoring_margin=5.0,  efg_pct=0.50, ft_pct=0.72, orb_rate=0.29,
         drb_rate=0.68, tov_rate=0.155, win_pct_last10=0.65, games_played=30),
    dict(net_rtg=-3.0, win_pct=0.48, seed=10, sos=0.46, ortg=103.0, drtg=106.0,
         scoring_margin=-2.0, efg_pct=0.47, ft_pct=0.68, orb_rate=0.26,
         drb_rate=0.64, tov_rate=0.170, win_pct_last10=0.45, games_played=30),
    dict(net_rtg=-15.0, win_pct=0.30, seed=14, sos=0.40, ortg=97.0,  drtg=112.0,
         scoring_margin=-10.0, efg_pct=0.43, ft_pct=0.64, orb_rate=0.22,
         drb_rate=0.58, tov_rate=0.190, win_pct_last10=0.30, games_played=30),
]

# All feature columns needed (superset of DEFAULT_FEATURE_COLS)
_ALL_FEAT = [
    "net_rtg", "win_pct", "seed", "sos", "ortg", "drtg", "scoring_margin",
    "efg_pct", "ft_pct", "orb_rate", "drb_rate", "tov_rate",
    "win_pct_last10", "games_played",
    # Additional Stage-01 columns (set to sensible defaults)
    "ppg_scored", "ppg_allowed", "home_win_pct", "away_win_pct",
    "neutral_win_pct", "fg_pct", "fg3_pct", "ts_pct", "possessions_pg",
    "tempo", "ast_rate", "blk_rate", "stl_rate", "ot_rate", "drb_rate",
    "conf_id",
]


def _team_tier(team_id: int, team_ids: List[int]) -> int:
    """Return the 0-based tier index for a team."""
    idx = team_ids.index(team_id)
    cumulative = 0
    for tier, size in enumerate(TIER_SIZES):
        cumulative += size
        if idx < cumulative:
            return tier
    return len(TIER_SIZES) - 1


def _make_feat_row(team_id: int, season: int, gender: str, team_ids: List[int]) -> dict:
    tier = _team_tier(team_id, team_ids)
    stats = TIER_STATS[tier].copy()
    # Add small per-season noise for realism
    rng = random.Random(team_id * 1000 + season)
    for key in ("net_rtg", "win_pct", "sos", "scoring_margin"):
        stats[key] += rng.gauss(0, 0.3)
    # Derived
    stats["ppg_scored"]  = stats["ortg"] * stats["possessions_pg"] / 100 if "possessions_pg" in stats else 72.0
    stats["ppg_allowed"] = stats["drtg"] * stats["possessions_pg"] / 100 if "possessions_pg" in stats else 66.0
    stats.setdefault("ppg_scored",  72.0)
    stats.setdefault("ppg_allowed", 66.0)
    stats.setdefault("home_win_pct",    stats["win_pct"] + 0.05)
    stats.setdefault("away_win_pct",    stats["win_pct"] - 0.05)
    stats.setdefault("neutral_win_pct", stats["win_pct"])
    stats.setdefault("fg_pct",  stats["efg_pct"] - 0.03)
    stats.setdefault("fg3_pct", stats["efg_pct"] - 0.08)
    stats.setdefault("ts_pct",  stats["efg_pct"] + 0.02)
    stats.setdefault("possessions_pg", 70.0)
    stats.setdefault("tempo",    70.0)
    stats.setdefault("ast_rate", 0.55)
    stats.setdefault("blk_rate", 0.08)
    stats.setdefault("stl_rate", 0.08)
    stats.setdefault("ot_rate",  0.05)
    stats.setdefault("conf_id",  tier + 1)
    return {
        "gender": gender,
        "season": season,
        "team_id": team_id,
        "canonical_id": f"{gender}_{team_id}",
        **stats,
    }


def make_feat_df() -> pd.DataFrame:
    """Create a synthetic team-season feature DataFrame."""
    rows = []
    for gender, team_ids in [("M", M_TEAM_IDS), ("W", W_TEAM_IDS)]:
        for season in SEASONS:
            for tid in team_ids:
                rows.append(_make_feat_row(tid, season, gender, team_ids))
    df = pd.DataFrame(rows)
    # Ensure all DEFAULT_FEATURE_COLS are present
    for col in DEFAULT_FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0
    return df.reset_index(drop=True)


def _better_team_wins(tid_a: int, tid_b: int, team_ids: List[int], seed: int) -> int:
    """Return the winning team_id; better tier wins 75% of the time."""
    tier_a = _team_tier(tid_a, team_ids)
    tier_b = _team_tier(tid_b, team_ids)
    rng = random.Random(seed)
    if tier_a == tier_b:
        return tid_a if rng.random() < 0.5 else tid_b
    favoured = tid_a if tier_a < tier_b else tid_b
    underdog = tid_b if tier_a < tier_b else tid_a
    return favoured if rng.random() < 0.75 else underdog


def make_tourney_df() -> pd.DataFrame:
    """
    Create synthetic tournament game rows.

    Simulates an 8-team single-elimination bracket per gender per season
    (real NCAA has 64/68, but 8 gives enough games for unit tests).

    Each game is stored as two rows (winner + loser perspective).
    """
    rows = []
    day = 145

    for season in SEASONS:
        for gender, team_ids in [("M", M_TEAM_IDS), ("W", W_TEAM_IDS)]:
            # Seed the top 8 teams (first 8 in our list = tiers 1+2)
            bracket = team_ids[:8]
            # Round 1: 4 games  (1v8, 2v7, 3v6, 4v5)
            winners = []
            for i in range(4):
                t_high = bracket[i]      # better team (lower tier index)
                t_low  = bracket[7 - i]  # worse team
                w = _better_team_wins(t_high, t_low, team_ids,
                                       seed=season * 1000 + i)
                l = t_low if w == t_high else t_high
                rows += _game_rows(season, day + i, gender, w, l, team_ids)
                winners.append(w)
            # Round 2: 2 games
            semi_winners = []
            for i in range(2):
                w = _better_team_wins(winners[2 * i], winners[2 * i + 1],
                                       team_ids, seed=season * 2000 + i)
                l = winners[2 * i + 1] if w == winners[2 * i] else winners[2 * i]
                rows += _game_rows(season, day + 10 + i, gender, w, l, team_ids)
                semi_winners.append(w)
            # Final: 1 game
            w = _better_team_wins(semi_winners[0], semi_winners[1],
                                   team_ids, seed=season * 3000)
            l = semi_winners[1] if w == semi_winners[0] else semi_winners[0]
            rows += _game_rows(season, day + 20, gender, w, l, team_ids)

    return pd.DataFrame(rows)


def _game_rows(
    season: int, day: int, gender: str,
    w_tid: int, l_tid: int, team_ids: List[int]
) -> List[dict]:
    """Return two rows (winner + loser perspective) for a tournament game."""
    w_tier = _team_tier(w_tid, team_ids)
    l_tier = _team_tier(l_tid, team_ids)
    w_score = 70 + (4 - w_tier) * 5 + random.randint(-3, 3)
    l_score = w_score - random.randint(2, 18)
    l_score = max(l_score, w_score - 25)

    base = dict(
        season=season, day_num=day, gender=gender,
        is_tourney=True,
        num_ot=0, loc="N",
    )
    winner_row = {**base,
        "team_id": w_tid, "opp_team_id": l_tid,
        "canonical_id": f"{gender}_{w_tid}",
        "opp_canonical_id": f"{gender}_{l_tid}",
        "score": w_score, "opp_score": l_score,
        "margin": w_score - l_score, "won": True,
        "seed": _tier_seed(w_tier), "opp_seed": _tier_seed(l_tier),
    }
    loser_row = {**base,
        "team_id": l_tid, "opp_team_id": w_tid,
        "canonical_id": f"{gender}_{l_tid}",
        "opp_canonical_id": f"{gender}_{w_tid}",
        "score": l_score, "opp_score": w_score,
        "margin": l_score - w_score, "won": False,
        "seed": _tier_seed(l_tier), "opp_seed": _tier_seed(w_tier),
    }
    return [winner_row, loser_row]


def _tier_seed(tier: int) -> int:
    return [2, 5, 10, 14][tier]


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def feat_df() -> pd.DataFrame:
    """Session-scoped team-season feature DataFrame."""
    return make_feat_df()


@pytest.fixture(scope="session")
def tourney_df() -> pd.DataFrame:
    """Session-scoped normalised tournament game rows."""
    return make_tourney_df()


@pytest.fixture(scope="session")
def fitted_model(feat_df, tourney_df):
    """Session-scoped pre-fitted LogisticBaseline for reuse."""
    from ncaa_model.baseline import LogisticBaseline
    model = LogisticBaseline()
    model.fit(feat_df, tourney_df, train_seasons=TRAIN_SEASONS)
    return model
