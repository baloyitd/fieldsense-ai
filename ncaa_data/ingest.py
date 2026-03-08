"""
ncaa_data.ingest
================
Load and validate raw Kaggle NCAA competition CSV files.

Supported files (both M/Men's and W/Women's variants):
  - [MW]RegularSeasonCompactResults.csv
  - [MW]RegularSeasonDetailedResults.csv
  - [MW]NCAATourneyCompactResults.csv
  - [MW]NCAATourneyDetailedResults.csv
  - [MW]Seasons.csv
  - [MW]Teams.csv
  - [MW]NCAATourneySeeds.csv
  - MGameCities.csv  (men's only supplementary)
  - MTeamConferences.csv  (men's only; WTeamConferences.csv if present)

All functions return plain pandas DataFrames with original Kaggle column names.
A 'Gender' column ('M' or 'W') is added where applicable so downstream
code can distinguish men's from women's data in unified DataFrames.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Kaggle file catalogue
# ---------------------------------------------------------------------------

#: Files that exist in both M (men's) and W (women's) variants.
_GENDERED_FILES: Dict[str, str] = {
    "regular_compact": "{g}RegularSeasonCompactResults.csv",
    "regular_detailed": "{g}RegularSeasonDetailedResults.csv",
    "tourney_compact": "{g}NCAATourneyCompactResults.csv",
    "tourney_detailed": "{g}NCAATourneyDetailedResults.csv",
    "seasons": "{g}Seasons.csv",
    "teams": "{g}Teams.csv",
    "seeds": "{g}NCAATourneySeeds.csv",
    "team_conferences": "{g}TeamConferences.csv",
}

#: Files that exist only in the men's dataset (or have a known single name).
_SUPPLEMENTARY_FILES: Dict[str, str] = {
    "game_cities": "MGameCities.csv",
}

# Compact results columns expected
_COMPACT_COLS = [
    "Season", "DayNum", "WTeamID", "WScore", "LTeamID", "LScore",
    "WLoc", "NumOT",
]

# Detailed results extra columns (W-prefix = winning team, L-prefix = losing)
_DETAILED_EXTRA_COLS = [
    "WFGM", "WFGA", "WFGM3", "WFGA3", "WFTM", "WFTA",
    "WOR", "WDR", "WAST", "WTO", "WSTL", "WBLK", "WPF",
    "LFGM", "LFGA", "LFGM3", "LFGA3", "LFTM", "LFTA",
    "LOR", "LDR", "LAST", "LTO", "LSTL", "LBLK", "LPF",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_csv(path: Path, required_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """Load a CSV, validate required columns, and return the DataFrame."""
    if not path.exists():
        raise FileNotFoundError(f"Kaggle data file not found: {path}")
    df = pd.read_csv(path)
    if required_cols:
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"{path.name} is missing expected columns: {missing}"
            )
    logger.debug("Loaded %s  shape=%s", path.name, df.shape)
    return df


def _try_load(path: Path) -> Optional[pd.DataFrame]:
    """Load CSV if it exists; return None otherwise (non-fatal)."""
    if not path.exists():
        logger.warning("Optional file not found, skipping: %s", path)
        return None
    return _load_csv(path)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_raw_data(data_dir: str | Path) -> Dict[str, pd.DataFrame]:
    """
    Load all available Kaggle NCAA data files from *data_dir*.

    Parameters
    ----------
    data_dir:
        Directory containing the Kaggle CSV files (e.g. ``data/ncaa/raw``).

    Returns
    -------
    dict
        Keys are logical names such as ``"M_regular_compact"``,
        ``"W_regular_detailed"``, ``"M_teams"``, etc.
        Values are DataFrames with a ``Gender`` column added
        (``'M'`` or ``'W'``) for the gendered files.

    Raises
    ------
    FileNotFoundError
        If a *required* file (regular season compact results for either
        gender) is missing.
    """
    data_dir = Path(data_dir)
    result: Dict[str, pd.DataFrame] = {}

    for gender in ("M", "W"):
        g = gender  # single-char prefix used in filenames

        for key, template in _GENDERED_FILES.items():
            filename = template.format(g=g)
            path = data_dir / filename
            full_key = f"{gender}_{key}"

            # Regular-season compact results are required; everything else is
            # optional (detailed stats may not ship with every competition).
            is_required = key == "regular_compact"

            if is_required:
                df = _load_csv(path, required_cols=_COMPACT_COLS)
            else:
                df = _try_load(path)
                if df is None:
                    continue

            df = df.copy()
            df["Gender"] = gender
            result[full_key] = df

    # Supplementary (men's-only) files
    for key, filename in _SUPPLEMENTARY_FILES.items():
        path = data_dir / filename
        df = _try_load(path)
        if df is not None:
            result[key] = df

    loaded = list(result.keys())
    logger.info("Loaded %d data files: %s", len(loaded), loaded)
    return result


def load_results(
    data_dir: str | Path,
    gender: str = "M",
    detailed: bool = True,
    tourney: bool = False,
) -> pd.DataFrame:
    """
    Convenience loader for a single result type.

    Parameters
    ----------
    data_dir : str | Path
    gender    : ``'M'`` or ``'W'``
    detailed  : Load detailed box-score results (True) or compact (False).
    tourney   : Load tournament games (True) or regular-season (False).

    Returns
    -------
    pd.DataFrame with ``Gender`` column added.
    """
    g = gender.upper()
    phase = "NCAATourn" if tourney else "RegularSeason"
    kind = "Detailed" if detailed else "Compact"
    filename = f"{g}{phase}{kind}Results.csv"
    path = Path(data_dir) / filename

    required = _COMPACT_COLS if not detailed else _COMPACT_COLS + _DETAILED_EXTRA_COLS[:1]
    df = _load_csv(path, required_cols=required)
    df = df.copy()
    df["Gender"] = g
    return df


def load_teams(data_dir: str | Path, gender: str = "M") -> pd.DataFrame:
    """Load teams reference table for one gender."""
    g = gender.upper()
    path = Path(data_dir) / f"{g}Teams.csv"
    return _load_csv(path, required_cols=["TeamID", "TeamName"])


def load_seeds(data_dir: str | Path, gender: str = "M") -> pd.DataFrame:
    """Load tournament seeds for one gender."""
    g = gender.upper()
    path = Path(data_dir) / f"{g}NCAATourneySeeds.csv"
    return _load_csv(path, required_cols=["Season", "Seed", "TeamID"])


def load_conferences(data_dir: str | Path, gender: str = "M") -> Optional[pd.DataFrame]:
    """Load team-conference mapping (returns None if file absent)."""
    g = gender.upper()
    path = Path(data_dir) / f"{g}TeamConferences.csv"
    return _try_load(path)
