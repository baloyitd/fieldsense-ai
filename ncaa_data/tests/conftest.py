"""
conftest.py – shared pytest fixtures for the NCAA data pipeline test suite.

Provides realistic synthetic Kaggle-format CSV files in a temporary directory.
All box-score stats are tuned to produce feature values within the expected
domain-valid ranges documented in test_features.py.

Stat design (per game, winning team):
  FGM=25, FGA=58, FGM3=7, FGA3=21, FTM=14, FTA=18
  OR=9, DR=20, AST=13, TO=13, STL=6, BLK=3, PF=18

Stat design (per game, losing team):
  FGM=22, FGA=56, FGM3=6, FGA3=19, FTM=11, FTA=15
  OR=10, DR=17, AST=9,  TO=15, STL=5, BLK=2, PF=20

Derived values (approximate):
  team_poss = 58 - 9 + 13 + 0.475*18 ≈ 70.6
  opp_poss  = 56 - 10 + 15 + 0.475*15 ≈ 68.1
  ORTG      ≈ 100 * 75 / 70.6 ≈ 106  ✓ [60,140]
  DRTG      ≈ 100 * 66 / 68.1 ≈  97  ✓ [60,140]
  FG%       = 25/58  ≈ 0.431           ✓ [0.25,0.70]
  3P%       = 7/21   ≈ 0.333           ✓ [0.15,0.55]
  FT%       = 14/18  ≈ 0.778           ✓ [0.40,0.95]
  ORB rate  = 9/(9+17) ≈ 0.346        ✓ [0.10,0.55]
  DRB rate  = 20/(20+10) ≈ 0.667      ✓ [0.40,0.90]
  Tempo     ≈ 70.6                     ✓ [55,85]
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Generator

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Low-level game builders
# ---------------------------------------------------------------------------

def _compact(season, day, wid, ws, lid, ls, wloc="N", num_ot=0):
    return dict(
        Season=season, DayNum=day,
        WTeamID=wid, WScore=ws, LTeamID=lid, LScore=ls,
        WLoc=wloc, NumOT=num_ot,
    )


def _detailed(season, day, wid, ws, lid, ls, wloc="N", num_ot=0):
    base = _compact(season, day, wid, ws, lid, ls, wloc, num_ot)
    base.update(dict(
        WFGM=25, WFGA=58, WFGM3=7, WFGA3=21, WFTM=14, WFTA=18,
        WOR=9,  WDR=20, WAST=13, WTO=13, WSTL=6,  WBLK=3,  WPF=18,
        LFGM=22, LFGA=56, LFGM3=6, LFGA3=19, LFTM=11, LFTA=15,
        LOR=10, LDR=17, LAST=9,  LTO=15, LSTL=5,  LBLK=2,  LPF=20,
    ))
    return base


# ---------------------------------------------------------------------------
# Data directory writer
# ---------------------------------------------------------------------------

TEAM_IDS = [1001, 1002, 1003, 1004]
SEASONS   = list(range(2021, 2026))

# Scores chosen so all teams win at least some games (varied SOS outcomes)
_SCHEDULE = [
    # (winner_idx, loser_idx, wloc, wscore, lscore)
    (0, 1, "H", 75, 66),
    (2, 3, "A", 80, 71),
    (0, 2, "N", 70, 64),
    (1, 3, "N", 78, 69),
    (3, 0, "H", 82, 75),   # team 3 beats team 0 occasionally
    (1, 2, "A", 68, 60),
]


def write_synthetic_data(data_dir: Path, gender: str = "M") -> None:
    """Write all required Kaggle-format CSV files for *gender* into *data_dir*."""
    g = gender

    # ---- Teams -------------------------------------------------------
    teams = pd.DataFrame([
        {"TeamID": tid, "TeamName": f"Team_{tid}"}
        for tid in TEAM_IDS
    ])
    teams.to_csv(data_dir / f"{g}Teams.csv", index=False)

    # ---- Seasons -----------------------------------------------------
    seasons_df = pd.DataFrame([
        {"Season": y, "DayzeroDate": f"{y-1}-10-30",
         "RegionW": "East", "RegionX": "West",
         "RegionY": "South", "RegionZ": "Midwest"}
        for y in SEASONS
    ])
    seasons_df.to_csv(data_dir / f"{g}Seasons.csv", index=False)

    # ---- Regular season results --------------------------------------
    compact_rows, detailed_rows = [], []
    for season in SEASONS:
        day = 10
        for wix, lix, wloc, ws, ls in _SCHEDULE:
            wid, lid = TEAM_IDS[wix], TEAM_IDS[lix]
            # Play each matchup 5 times per season (different days)
            for rep in range(5):
                d = day + rep * 4
                compact_rows.append(_compact(season, d, wid, ws, lid, ls, wloc))
                detailed_rows.append(_detailed(season, d, wid, ws, lid, ls, wloc))
            day += 25

    pd.DataFrame(compact_rows).to_csv(
        data_dir / f"{g}RegularSeasonCompactResults.csv", index=False
    )
    pd.DataFrame(detailed_rows).to_csv(
        data_dir / f"{g}RegularSeasonDetailedResults.csv", index=False
    )

    # ---- Tournament results ------------------------------------------
    tourn_compact, tourn_detailed = [], []
    for season in SEASONS:
        # Two tournament games per season
        tourn_compact.append(_compact(season, 145, 1001, 85, 1002, 72, "N"))
        tourn_compact.append(_compact(season, 148, 1003, 78, 1004, 68, "N"))
        tourn_detailed.append(_detailed(season, 145, 1001, 85, 1002, 72, "N"))
        tourn_detailed.append(_detailed(season, 148, 1003, 78, 1004, 68, "N"))

    pd.DataFrame(tourn_compact).to_csv(
        data_dir / f"{g}NCAATourneyCompactResults.csv", index=False
    )
    pd.DataFrame(tourn_detailed).to_csv(
        data_dir / f"{g}NCAATourneyDetailedResults.csv", index=False
    )

    # ---- Seeds -------------------------------------------------------
    seed_rows = []
    for s in SEASONS:
        seed_rows += [
            {"Season": s, "Seed": "W01", "TeamID": 1001},
            {"Season": s, "Seed": "X04", "TeamID": 1002},
            {"Season": s, "Seed": "Y02", "TeamID": 1003},
            {"Season": s, "Seed": "Z08a", "TeamID": 1004},
        ]
    pd.DataFrame(seed_rows).to_csv(
        data_dir / f"{g}NCAATourneySeeds.csv", index=False
    )

    # ---- Conferences -------------------------------------------------
    conf_map = {1001: "ACC", 1002: "ACC", 1003: "B10", 1004: "B10"}
    conf_rows = [
        {"Season": s, "TeamID": tid, "ConfAbbrev": conf}
        for s in SEASONS
        for tid, conf in conf_map.items()
    ]
    pd.DataFrame(conf_rows).to_csv(
        data_dir / f"{g}TeamConferences.csv", index=False
    )

    # ---- Game Cities (men's supplementary, optional) -----------------
    if gender == "M":
        cities = pd.DataFrame([
            {"Season": s, "DayNum": 10, "WTeamID": 1001, "LTeamID": 1002,
             "CRType": "A", "CityID": 101}
            for s in SEASONS
        ])
        cities.to_csv(data_dir / "MGameCities.csv", index=False)


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def data_dir() -> Generator[Path, None, None]:
    """Shared temporary directory with synthetic M + W Kaggle CSVs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        d = Path(tmpdir)
        write_synthetic_data(d, "M")
        write_synthetic_data(d, "W")
        yield d


@pytest.fixture(scope="session")
def raw_data(data_dir):
    """Loaded raw data dict (session-scoped for speed)."""
    from ncaa_data.ingest import load_raw_data
    return load_raw_data(data_dir)


@pytest.fixture(scope="session")
def games_df(raw_data):
    """Normalized game-level DataFrame (session-scoped)."""
    from ncaa_data.normalize import normalize_all
    return normalize_all(raw_data)


@pytest.fixture(scope="session")
def feat_df(games_df):
    """Full feature matrix for all seasons (session-scoped)."""
    from ncaa_data.features import compute_features
    return compute_features(games_df, seasons=SEASONS)


@pytest.fixture()
def out_dir(tmp_path):
    """Fresh temporary output directory per test."""
    return tmp_path / "features"
