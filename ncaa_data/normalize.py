"""
ncaa_data.normalize
===================
Canonical team ID schema and data normalization for the NCAA pipeline.

Design
------
Kaggle assigns integer TeamIDs independently for men's and women's datasets,
so the same integer can represent different teams.  To produce a unified,
collision-free identifier we prefix every TeamID with its gender character:

    canonical_id = f"{gender}_{team_id}"   e.g. "M_1234" or "W_5678"

This module also:
  * Flattens *game-level* win/loss rows into *team-level* rows (one row per
    participating team per game), making downstream aggregation straightforward.
  * Attaches conference and seed metadata when available.
  * Encodes conference strings as stable integers for ML consumption.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Team-ID helpers
# ---------------------------------------------------------------------------

def make_canonical_id(gender: str, team_id: int | str) -> str:
    """Return a collision-free team identifier: ``"M_1234"`` or ``"W_5678"``."""
    return f"{gender.upper()}_{int(team_id)}"


def parse_seed_number(seed_str: str) -> int:
    """
    Extract the numeric seed from a Kaggle seed string such as ``'W01a'``.

    The Kaggle format is ``<region><seed>[<play-in suffix>]``:
      * Region letter (W / X / Y / Z)
      * Two-digit seed (01-16)
      * Optional play-in suffix (a / b)

    Returns seed as int (1-16).  Returns 17 for any parse error (treated as
    "not seeded / unknown").
    """
    try:
        # Strip leading region letter and trailing play-in suffix
        numeric = "".join(c for c in seed_str if c.isdigit())
        return int(numeric) if numeric else 17
    except (ValueError, TypeError):
        return 17


# ---------------------------------------------------------------------------
# Conference registry
# ---------------------------------------------------------------------------

class ConferenceRegistry:
    """
    Maps conference abbreviation strings to stable integer codes.

    The registry is built lazily from whatever conferences appear in the data.
    Code 0 is reserved for "Unknown".
    """

    def __init__(self) -> None:
        self._map: Dict[str, int] = {"Unknown": 0}
        self._next_id = 1

    def encode(self, conf_abbrev: str) -> int:
        """Return (and register if new) the integer code for *conf_abbrev*."""
        key = str(conf_abbrev).strip() if conf_abbrev else "Unknown"
        if key not in self._map:
            self._map[key] = self._next_id
            self._next_id += 1
        return self._map[key]

    def encode_series(self, series: pd.Series) -> pd.Series:
        """Vectorised encode for a whole column."""
        return series.fillna("Unknown").map(lambda x: self.encode(str(x)))

    @property
    def mapping(self) -> Dict[str, int]:
        """Return a copy of the current abbrev → int mapping."""
        return dict(self._map)


# ---------------------------------------------------------------------------
# Game flattening
# ---------------------------------------------------------------------------

def flatten_games(
    results_df: pd.DataFrame,
    gender: str,
    is_tourney: bool = False,
) -> pd.DataFrame:
    """
    Convert a Kaggle results DataFrame (one row = one game, W-prefix / L-prefix
    columns) into a *team-perspective* DataFrame (one row = one team's view of
    one game).

    Parameters
    ----------
    results_df : pd.DataFrame
        Output of :func:`~ncaa_data.ingest.load_results` or equivalent.
        Must contain at minimum: ``Season``, ``DayNum``, ``WTeamID``,
        ``WScore``, ``LTeamID``, ``LScore``, ``WLoc``, ``NumOT``.
    gender : str
        ``'M'`` or ``'W'`` – used to build canonical IDs.
    is_tourney : bool
        Tag rows as tournament or regular-season games.

    Returns
    -------
    pd.DataFrame
        Columns:
          season, day_num, gender, team_id, canonical_id, opp_id,
          score, opp_score, margin, won, loc,
          num_ot, is_tourney,
          [optional detailed box-score columns with team_ / opp_ prefix]
    """
    df = results_df.copy()
    g = gender.upper()

    # Detect whether detailed stats columns are present
    has_detailed = "WFGM" in df.columns

    # ---- Build winning-team rows ----------------------------------------
    w_rows = pd.DataFrame()
    w_rows["season"] = df["Season"]
    w_rows["day_num"] = df["DayNum"]
    w_rows["gender"] = g
    w_rows["team_id"] = df["WTeamID"].astype(int)
    w_rows["opp_team_id"] = df["LTeamID"].astype(int)
    w_rows["score"] = df["WScore"]
    w_rows["opp_score"] = df["LScore"]
    w_rows["margin"] = df["WScore"] - df["LScore"]
    w_rows["won"] = True
    # WLoc is from winning team's perspective: H / A / N
    w_rows["loc"] = df["WLoc"].fillna("N")
    w_rows["num_ot"] = df["NumOT"].fillna(0).astype(int)
    w_rows["is_tourney"] = is_tourney

    if has_detailed:
        _attach_detailed(w_rows, df, winner=True)

    # ---- Build losing-team rows -----------------------------------------
    l_rows = pd.DataFrame()
    l_rows["season"] = df["Season"]
    l_rows["day_num"] = df["DayNum"]
    l_rows["gender"] = g
    l_rows["team_id"] = df["LTeamID"].astype(int)
    l_rows["opp_team_id"] = df["WTeamID"].astype(int)
    l_rows["score"] = df["LScore"]
    l_rows["opp_score"] = df["WScore"]
    l_rows["margin"] = df["LScore"] - df["WScore"]
    l_rows["won"] = False
    # Flip location from losing team's perspective
    loc_flip = {"H": "A", "A": "H", "N": "N"}
    l_rows["loc"] = df["WLoc"].fillna("N").map(loc_flip)
    l_rows["num_ot"] = df["NumOT"].fillna(0).astype(int)
    l_rows["is_tourney"] = is_tourney

    if has_detailed:
        _attach_detailed(l_rows, df, winner=False)

    combined = pd.concat([w_rows, l_rows], ignore_index=True)

    # Add canonical IDs
    combined["canonical_id"] = combined.apply(
        lambda r: make_canonical_id(r["gender"], r["team_id"]), axis=1
    )
    combined["opp_canonical_id"] = combined.apply(
        lambda r: make_canonical_id(r["gender"], r["opp_team_id"]), axis=1
    )

    return combined


def _attach_detailed(
    target: pd.DataFrame,
    src: pd.DataFrame,
    winner: bool,
) -> None:
    """
    Attach detailed box-score stats to *target* (in-place).

    Maps W-prefix columns → team stats and L-prefix columns → opp stats
    (or vice-versa when *winner=False*).
    """
    prefix_team = "W" if winner else "L"
    prefix_opp = "L" if winner else "W"

    detailed_stats = ["FGM", "FGA", "FGM3", "FGA3", "FTM", "FTA",
                      "OR", "DR", "AST", "TO", "STL", "BLK", "PF"]

    for stat in detailed_stats:
        col_team = f"{prefix_team}{stat}"
        col_opp = f"{prefix_opp}{stat}"
        if col_team in src.columns:
            target[f"team_{stat.lower()}"] = src[col_team].values
        if col_opp in src.columns:
            target[f"opp_{stat.lower()}"] = src[col_opp].values


# ---------------------------------------------------------------------------
# Metadata attachment
# ---------------------------------------------------------------------------

def attach_seeds(
    games_df: pd.DataFrame,
    seeds_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge tournament seed information onto *games_df*.

    Adds column ``seed`` (int, 1-16; 17 = not seeded / not in tournament).

    Parameters
    ----------
    games_df  : output of :func:`flatten_games`
    seeds_df  : output of :func:`~ncaa_data.ingest.load_seeds`
    """
    seeds = seeds_df[["Season", "TeamID", "Seed"]].copy()
    seeds = seeds.rename(columns={"TeamID": "team_id", "Season": "season"})
    seeds["seed_num"] = seeds["Seed"].apply(parse_seed_number)
    seeds = seeds.drop(columns=["Seed"])

    merged = games_df.merge(
        seeds[["season", "team_id", "seed_num"]],
        on=["season", "team_id"],
        how="left",
    )
    merged["seed"] = merged["seed_num"].fillna(17).astype(int)
    merged = merged.drop(columns=["seed_num"])
    return merged


def attach_conferences(
    games_df: pd.DataFrame,
    conf_df: pd.DataFrame,
    registry: ConferenceRegistry,
) -> pd.DataFrame:
    """
    Merge conference information and encode as integer.

    Adds column ``conf_id`` (int, 0 = unknown).

    Parameters
    ----------
    games_df  : output of :func:`flatten_games`
    conf_df   : output of :func:`~ncaa_data.ingest.load_conferences`
    registry  : shared :class:`ConferenceRegistry` instance
    """
    conf = conf_df[["Season", "TeamID", "ConfAbbrev"]].copy()
    conf = conf.rename(columns={"TeamID": "team_id", "Season": "season"})
    conf["conf_id"] = registry.encode_series(conf["ConfAbbrev"])

    merged = games_df.merge(
        conf[["season", "team_id", "conf_id"]],
        on=["season", "team_id"],
        how="left",
    )
    merged["conf_id"] = merged["conf_id"].fillna(0).astype(int)
    return merged


# ---------------------------------------------------------------------------
# Master normalizer
# ---------------------------------------------------------------------------

def normalize_all(
    raw_data: Dict[str, pd.DataFrame],
    registry: Optional[ConferenceRegistry] = None,
) -> pd.DataFrame:
    """
    Full normalization pass over the raw data dict from
    :func:`~ncaa_data.ingest.load_raw_data`.

    Returns a single unified game-level DataFrame with canonical IDs,
    optional conference and seed columns, ready for feature computation.

    Parameters
    ----------
    raw_data  : dict returned by :func:`~ncaa_data.ingest.load_raw_data`
    registry  : optional shared :class:`ConferenceRegistry`; created if None

    Returns
    -------
    pd.DataFrame
        All regular-season and tournament games for both genders, deduplicated,
        sorted by (gender, season, day_num).
    """
    if registry is None:
        registry = ConferenceRegistry()

    all_frames: List[pd.DataFrame] = []

    for gender in ("M", "W"):
        g = gender

        # Prefer detailed results; fall back to compact
        for is_tourney, phase_key_detailed, phase_key_compact in [
            (False, f"{g}_regular_detailed", f"{g}_regular_compact"),
            (True,  f"{g}_tourney_detailed",  f"{g}_tourney_compact"),
        ]:
            if phase_key_detailed in raw_data:
                results = raw_data[phase_key_detailed]
            elif phase_key_compact in raw_data:
                results = raw_data[phase_key_compact]
                logger.warning(
                    "Detailed results not found for %s %s; using compact.",
                    gender, "tourney" if is_tourney else "regular",
                )
            else:
                logger.warning(
                    "No results found for %s %s – skipping.",
                    gender, "tourney" if is_tourney else "regular",
                )
                continue

            flat = flatten_games(results, gender=g, is_tourney=is_tourney)

            # Attach seeds if available
            seed_key = f"{g}_seeds"
            if seed_key in raw_data:
                flat = attach_seeds(flat, raw_data[seed_key])

            # Attach conferences if available
            conf_key = f"{g}_team_conferences"
            if conf_key in raw_data:
                flat = attach_conferences(flat, raw_data[conf_key], registry)

            all_frames.append(flat)

    if not all_frames:
        raise ValueError("No result data found in raw_data dict.")

    unified = pd.concat(all_frames, ignore_index=True)
    unified = unified.sort_values(
        ["gender", "season", "day_num"], ignore_index=True
    )

    logger.info(
        "Normalized %d game-team rows across %d seasons",
        len(unified),
        unified["season"].nunique(),
    )
    return unified
