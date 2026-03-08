"""
ncaa_data.features
==================
Compute 28 basketball-specific aggregate features per (gender, team, season).

Feature catalogue (28 total)
-----------------------------
Basic performance
  1  games_played          total games played
  2  win_pct               overall win percentage
  3  ppg_scored            points per game (scored)
  4  ppg_allowed           points per game (allowed)
  5  scoring_margin        ppg_scored - ppg_allowed

Situational win %
  6  home_win_pct          win % in home games
  7  away_win_pct          win % in away games
  8  neutral_win_pct       win % at neutral sites
  9  win_pct_last10        win % across final 10 regular-season games

Shooting (requires detailed stats; NaN if unavailable)
  10 fg_pct                FGM / FGA
  11 fg3_pct               FGM3 / FGA3
  12 ft_pct                FTM / FTA
  13 efg_pct               (FGM + 0.5·FGM3) / FGA    (effective FG%)
  14 ts_pct                pts / (2·(FGA + 0.44·FTA)) (true shooting%)

Rebounding
  15 orb_rate              ORB / (ORB + opp_DR)
  16 drb_rate              DRB / (DRB + opp_OR)

Efficiency ratings (Oliver formula for possessions)
  17 possessions_pg        estimated possessions per game
  18 ortg                  points scored per 100 possessions
  19 drtg                  points allowed per 100 possessions
  20 net_rtg               ORTG - DRTG
  21 tempo                 possessions per game (proxy for pace)

Miscellaneous per-possession rates
  22 tov_rate              turnovers / possessions
  23 ast_rate              assists / FGM
  24 blk_rate              blocks / opp FGA
  25 stl_rate              steals / opp possessions

Schedule / context
  26 ot_rate               fraction of games going to overtime
  27 sos                   strength of schedule (avg opp win%)
  28 seed                  lowest tournament seed in that season (17 = unseeded)

Conference metadata
  29 conf_id               integer-encoded conference (0 = unknown)
     [not counted towards 28; attached from normalizer]

Usage
-----
    from ncaa_data.features import compute_features
    feat_df = compute_features(games_df, seasons=range(2021, 2026))
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Possessions formula (Oliver 2004)
# ---------------------------------------------------------------------------
# Poss ≈ FGA - OR + TO + 0.44·FTA
# When box-score stats are unavailable we fall back to a score-based estimate.

_EPSILON = 1e-6  # avoid division by zero


def _possessions(fga, or_, to, fta) -> float:
    """Estimate possessions via Oliver's formula (0.475 FTA coefficient)."""
    return fga - or_ + to + 0.475 * fta


# ---------------------------------------------------------------------------
# Core feature computation
# ---------------------------------------------------------------------------

def _safe_div(num, den, default=np.nan):
    """Divide, returning *default* where denominator is zero."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(den > _EPSILON, num / den, default)
    return result


def compute_features(
    games_df: pd.DataFrame,
    seasons: Optional[Iterable[int]] = None,
) -> pd.DataFrame:
    """
    Aggregate game-level rows from :func:`~ncaa_data.normalize.normalize_all`
    into one feature row per (gender, season, team).

    Parameters
    ----------
    games_df : pd.DataFrame
        Output of :func:`~ncaa_data.normalize.normalize_all`.
        Must contain at minimum: ``gender``, ``season``, ``canonical_id``,
        ``team_id``, ``opp_canonical_id``, ``score``, ``opp_score``,
        ``won``, ``loc``, ``num_ot``, ``day_num``, ``is_tourney``.
        Detailed box-score columns (``team_fgm``, etc.) are used when present.
    seasons : iterable of int, optional
        Filter to specific seasons (e.g. ``range(2021, 2026)``).
        If None, all seasons in the data are used.

    Returns
    -------
    pd.DataFrame
        One row per (gender, season, canonical_id).
        Index is reset (integer 0-based).
        Feature columns are documented in the module docstring.
    """
    df = games_df.copy()

    if seasons is not None:
        season_list = list(seasons)
        df = df[df["season"].isin(season_list)].copy()
        if df.empty:
            raise ValueError(f"No games found for seasons {season_list}")

    # Only regular-season games for most features
    reg = df[~df["is_tourney"]].copy()
    # Tournament games used for seed extraction
    tourn = df[df["is_tourney"]].copy()

    has_detailed = "team_fgm" in reg.columns

    # Pre-compute per-game possessions for the detailed path
    if has_detailed:
        for part, fga_col, or_col, to_col, fta_col in [
            ("team", "team_fga", "team_or", "team_to", "team_fta"),
            ("opp",  "opp_fga",  "opp_or",  "opp_to",  "opp_fta"),
        ]:
            reg[f"{part}_poss"] = (
                reg[fga_col].fillna(0)
                - reg[or_col].fillna(0)
                + reg[to_col].fillna(0)
                + 0.475 * reg[fta_col].fillna(0)
            ).clip(lower=1)

    # -----------------------------------------------------------------------
    # Aggregate per (gender, season, canonical_id)
    # -----------------------------------------------------------------------
    group_keys = ["gender", "season", "canonical_id"]

    records = []

    for (gender, season, cid), grp in reg.groupby(group_keys):
        r: dict = {}
        r["gender"] = gender
        r["season"] = season
        r["canonical_id"] = cid
        r["team_id"] = int(grp["team_id"].iloc[0])

        n = len(grp)
        r["games_played"] = n

        # ---- Basic win stats -------------------------------------------
        wins = grp["won"].sum()
        r["win_pct"] = wins / n if n else np.nan
        r["ppg_scored"] = grp["score"].mean()
        r["ppg_allowed"] = grp["opp_score"].mean()
        r["scoring_margin"] = r["ppg_scored"] - r["ppg_allowed"]

        # ---- Situational win % -----------------------------------------
        for loc_val, col in [("H", "home_win_pct"), ("A", "away_win_pct"), ("N", "neutral_win_pct")]:
            sub = grp[grp["loc"] == loc_val]
            r[col] = sub["won"].mean() if len(sub) else np.nan

        # Last-10 games (sorted by day_num)
        last10 = grp.nlargest(10, "day_num")
        r["win_pct_last10"] = last10["won"].mean() if len(last10) else np.nan

        # ---- OT rate ---------------------------------------------------
        r["ot_rate"] = (grp["num_ot"] > 0).mean()

        # ---- Detailed shooting / rebounding / efficiency ---------------
        if has_detailed:
            g = grp  # alias

            tot_fgm  = g["team_fgm"].sum()
            tot_fga  = g["team_fga"].sum()
            tot_fgm3 = g["team_fgm3"].sum()
            tot_fga3 = g["team_fga3"].sum()
            tot_ftm  = g["team_ftm"].sum()
            tot_fta  = g["team_fta"].sum()
            tot_or   = g["team_or"].sum()
            tot_dr   = g["team_dr"].sum()
            tot_ast  = g["team_ast"].sum()
            tot_to   = g["team_to"].sum()
            tot_stl  = g["team_stl"].sum()
            tot_blk  = g["team_blk"].sum()

            tot_opp_fga = g["opp_fga"].sum()
            tot_opp_or  = g["opp_or"].sum()
            tot_opp_dr  = g["opp_dr"].sum()
            tot_opp_to  = g["opp_to"].sum()
            tot_opp_fgm = g["opp_fgm"].sum()

            team_poss = g["team_poss"].sum()
            opp_poss  = g["opp_poss"].sum()

            # Shooting
            r["fg_pct"]  = tot_fgm / tot_fga if tot_fga > 0 else np.nan
            r["fg3_pct"] = tot_fgm3 / tot_fga3 if tot_fga3 > 0 else np.nan
            r["ft_pct"]  = tot_ftm / tot_fta if tot_fta > 0 else np.nan
            r["efg_pct"] = (tot_fgm + 0.5 * tot_fgm3) / tot_fga if tot_fga > 0 else np.nan
            denom_ts = 2 * (tot_fga + 0.475 * tot_fta)
            r["ts_pct"] = g["score"].sum() / denom_ts if denom_ts > 0 else np.nan

            # Rebounding rates
            orb_denom = tot_or + tot_opp_dr
            drb_denom = tot_dr + tot_opp_or
            r["orb_rate"] = tot_or / orb_denom if orb_denom > 0 else np.nan
            r["drb_rate"] = tot_dr / drb_denom if drb_denom > 0 else np.nan

            # Efficiency ratings
            avg_poss = team_poss / n if n else 1
            r["possessions_pg"] = avg_poss
            r["tempo"] = avg_poss  # pace proxy

            pts_scored  = g["score"].sum()
            pts_allowed = g["opp_score"].sum()
            r["ortg"] = 100 * pts_scored  / team_poss if team_poss > 0 else np.nan
            r["drtg"] = 100 * pts_allowed / opp_poss  if opp_poss  > 0 else np.nan
            r["net_rtg"] = (r["ortg"] - r["drtg"]) if (
                not np.isnan(r["ortg"]) and not np.isnan(r["drtg"])
            ) else np.nan

            # Per-possession misc rates
            r["tov_rate"] = tot_to  / team_poss if team_poss > 0 else np.nan
            r["ast_rate"] = tot_ast / tot_fgm   if tot_fgm  > 0 else np.nan
            r["blk_rate"] = tot_blk / tot_opp_fga if tot_opp_fga > 0 else np.nan
            r["stl_rate"] = tot_stl / opp_poss  if opp_poss  > 0 else np.nan
        else:
            # Compact-only fallback: NaN for stats requiring box scores
            for col in [
                "fg_pct", "fg3_pct", "ft_pct", "efg_pct", "ts_pct",
                "orb_rate", "drb_rate",
                "possessions_pg", "ortg", "drtg", "net_rtg", "tempo",
                "tov_rate", "ast_rate", "blk_rate", "stl_rate",
            ]:
                r[col] = np.nan

        records.append(r)

    feat_df = pd.DataFrame(records)

    # -----------------------------------------------------------------------
    # Strength of Schedule (two-pass: need win% first)
    # -----------------------------------------------------------------------
    feat_df = _compute_sos(feat_df, reg)

    # -----------------------------------------------------------------------
    # Attach best tournament seed per team-season
    # -----------------------------------------------------------------------
    feat_df = _attach_best_seed(feat_df, tourn)

    # -----------------------------------------------------------------------
    # Attach conf_id if present in games_df
    # -----------------------------------------------------------------------
    if "conf_id" in reg.columns:
        conf_map = (
            reg.groupby(["gender", "season", "canonical_id"])["conf_id"]
            .first()
            .reset_index()
        )
        feat_df = feat_df.merge(conf_map, on=["gender", "season", "canonical_id"], how="left")
        feat_df["conf_id"] = feat_df["conf_id"].fillna(0).astype(int)
    else:
        feat_df["conf_id"] = 0

    feat_df = feat_df.reset_index(drop=True)

    logger.info(
        "Computed %d features for %d team-seasons",
        len(feat_df.columns) - 3,  # exclude gender, season, canonical_id
        len(feat_df),
    )
    return feat_df


# ---------------------------------------------------------------------------
# SOS helper
# ---------------------------------------------------------------------------

def _compute_sos(feat_df: pd.DataFrame, reg_games: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Strength of Schedule as average opponent win percentage.

    Merges onto *feat_df* as column ``sos``.
    """
    # Build (gender, season, canonical_id) → win_pct lookup
    win_pct_lookup = feat_df.set_index(
        ["gender", "season", "canonical_id"]
    )["win_pct"].to_dict()

    def _opp_win_pct(row) -> float:
        key = (row["gender"], row["season"], row["opp_canonical_id"])
        return win_pct_lookup.get(key, np.nan)

    reg = reg_games.copy()
    reg["opp_win_pct"] = reg.apply(_opp_win_pct, axis=1)

    sos = (
        reg.groupby(["gender", "season", "canonical_id"])["opp_win_pct"]
        .mean()
        .reset_index()
        .rename(columns={"opp_win_pct": "sos"})
    )

    feat_df = feat_df.merge(sos, on=["gender", "season", "canonical_id"], how="left")
    return feat_df


# ---------------------------------------------------------------------------
# Seed helper
# ---------------------------------------------------------------------------

def _attach_best_seed(feat_df: pd.DataFrame, tourn_games: pd.DataFrame) -> pd.DataFrame:
    """
    Extract the best (lowest numeric) seed per team-season from tournament rows.
    Teams that did not reach the tournament get seed = 17.
    """
    if tourn_games.empty or "seed" not in tourn_games.columns:
        feat_df["seed"] = 17
        return feat_df

    best_seed = (
        tourn_games.groupby(["gender", "season", "canonical_id"])["seed"]
        .min()
        .reset_index()
    )
    feat_df = feat_df.merge(
        best_seed, on=["gender", "season", "canonical_id"], how="left"
    )
    feat_df["seed"] = feat_df["seed"].fillna(17).astype(int)
    return feat_df


# ---------------------------------------------------------------------------
# Utility: list all 28 feature columns (excluding keys)
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "games_played", "win_pct", "ppg_scored", "ppg_allowed", "scoring_margin",
    "home_win_pct", "away_win_pct", "neutral_win_pct", "win_pct_last10",
    "fg_pct", "fg3_pct", "ft_pct", "efg_pct", "ts_pct",
    "orb_rate", "drb_rate",
    "possessions_pg", "ortg", "drtg", "net_rtg", "tempo",
    "tov_rate", "ast_rate", "blk_rate", "stl_rate",
    "ot_rate", "sos", "seed",
]

KEY_COLS = ["gender", "season", "canonical_id", "team_id"]
