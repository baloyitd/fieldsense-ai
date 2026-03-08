"""
ncaa_model.submit
=================
Kaggle submission CSV generation.

Submission format
-----------------
The competition requires a CSV with exactly two columns::

    ID,Pred

Where:
  * ``ID   = "YYYY_TeamIdLow_TeamIdHigh"``   (raw integer TeamIDs)
  * ``Pred = P(lower TeamID team wins)``     float ∈ [0, 1]

One row must be present for **every possible ordered pair** of teams from
the pool of tournament-eligible teams, for the prediction season.

Men's and Women's submissions may be separate files or concatenated into one
(the competition has varied this by year; this module supports both).

Usage
-----
::

    from ncaa_model.submit import build_submission
    sub_df = build_submission(
        model, feat_df,
        team_ids_m=m_team_ids,
        team_ids_w=w_team_ids,
        season=2026,
        output_path="submission.csv",
    )
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

from .matchup import build_prediction_pairs

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ID helpers
# ---------------------------------------------------------------------------

def submission_id(season: int, team_id_low: int, team_id_high: int) -> str:
    """
    Format the Kaggle submission ID string.

    Parameters
    ----------
    season : int  e.g. 2026
    team_id_low, team_id_high : int
        Raw Kaggle TeamIDs; team_id_low < team_id_high.

    Returns
    -------
    str  e.g. ``"2026_1101_1201"``
    """
    return f"{season}_{team_id_low}_{team_id_high}"


def parse_submission_id(sid: str):
    """
    Parse a submission ID back into (season, team_id_low, team_id_high).

    Parameters
    ----------
    sid : str  e.g. ``"2026_1101_1201"``

    Returns
    -------
    tuple (int, int, int)

    Raises
    ------
    ValueError  if the string is malformed
    """
    parts = sid.split("_")
    if len(parts) != 3:
        raise ValueError(f"Invalid submission ID: {sid!r} (expected 'YYYY_LowId_HighId')")
    return int(parts[0]), int(parts[1]), int(parts[2])


# ---------------------------------------------------------------------------
# Per-gender prediction
# ---------------------------------------------------------------------------

def _predict_gender(
    model,
    feat_df: pd.DataFrame,
    team_ids: Iterable[int],
    gender: str,
    season: int,
) -> pd.DataFrame:
    """
    Build predictions for all pairs of *team_ids* for a single gender.

    Returns a DataFrame with columns: ID, Pred.
    """
    team_ids_list = sorted(set(team_ids))
    if len(team_ids_list) < 2:
        logger.warning("Fewer than 2 team_ids for gender=%s; no pairs possible.", gender)
        return pd.DataFrame(columns=["ID", "Pred"])

    X, meta = build_prediction_pairs(
        feat_df, team_ids_list, gender, season, model.feature_cols
    )
    probs: np.ndarray = model.predict_proba_batch(X)

    rows = []
    for i, (_, row) in enumerate(meta.iterrows()):
        sid = submission_id(season, int(row["team_id_low"]), int(row["team_id_high"]))
        rows.append({"ID": sid, "Pred": float(probs[i])})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_submission(
    model,
    feat_df: pd.DataFrame,
    team_ids_m: Optional[Iterable[int]] = None,
    team_ids_w: Optional[Iterable[int]] = None,
    season: int = 2026,
    clip_probs: bool = True,
    output_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """
    Generate the Kaggle submission CSV.

    Parameters
    ----------
    model : NCAAPredictor
        Fitted model with ``predict_proba_matchup`` and
        ``predict_proba_batch`` methods.
    feat_df : pd.DataFrame
        Team-season features (all genders + seasons).
    team_ids_m : iterable of int, optional
        Raw men's TeamIDs to include.  If None, derived from feat_df.
    team_ids_w : iterable of int, optional
        Raw women's TeamIDs to include.  If None, derived from feat_df.
    season : int
        Prediction season (e.g. 2026).
    clip_probs : bool
        If True, clip predictions to [0.025, 0.975] to avoid extreme
        log-loss penalties.  Consistent with Kaggle best practices.
    output_path : str | Path, optional
        If provided, write the submission CSV to this path.

    Returns
    -------
    pd.DataFrame  with columns: ID, Pred  (sorted by ID ascending)
    """
    parts: List[pd.DataFrame] = []

    def _ids_for_gender(ids_arg, gender):
        if ids_arg is not None:
            return list(ids_arg)
        # Fall back: use all teams with features for this gender
        sub = feat_df[feat_df["gender"] == gender]
        if sub.empty:
            return []
        return sorted(sub["team_id"].unique().tolist())

    for gender, ids_arg in [("M", team_ids_m), ("W", team_ids_w)]:
        ids = _ids_for_gender(ids_arg, gender)
        if not ids:
            logger.warning("No team_ids for gender=%s; skipping.", gender)
            continue
        part = _predict_gender(model, feat_df, ids, gender, season)
        if not part.empty:
            parts.append(part)

    if not parts:
        raise ValueError("No predictions generated — check team_ids and feat_df.")

    sub_df = pd.concat(parts, ignore_index=True)
    sub_df = sub_df.sort_values("ID").reset_index(drop=True)

    if clip_probs:
        sub_df["Pred"] = sub_df["Pred"].clip(0.025, 0.975)

    logger.info("Submission: %d rows, Pred range [%.4f, %.4f]",
                len(sub_df), sub_df["Pred"].min(), sub_df["Pred"].max())

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sub_df.to_csv(output_path, index=False)
        logger.info("Submission saved to %s", output_path)

    return sub_df


def validate_submission(
    sub_df: pd.DataFrame,
    expected_season: Optional[int] = None,
) -> dict:
    """
    Validate a submission DataFrame against Kaggle format requirements.

    Parameters
    ----------
    sub_df : pd.DataFrame  with columns ID, Pred
    expected_season : int, optional
        If provided, all IDs must start with this year.

    Returns
    -------
    dict with keys: valid (bool), n_rows, issues (list[str])
    """
    issues: List[str] = []

    # Required columns
    for col in ("ID", "Pred"):
        if col not in sub_df.columns:
            issues.append(f"Missing required column: {col}")

    if issues:
        return {"valid": False, "n_rows": len(sub_df), "issues": issues}

    # Pred range
    out_of_range = ((sub_df["Pred"] < 0) | (sub_df["Pred"] > 1)).sum()
    if out_of_range:
        issues.append(f"{out_of_range} Pred values outside [0, 1]")

    # Duplicate IDs
    dupes = sub_df["ID"].duplicated().sum()
    if dupes:
        issues.append(f"{dupes} duplicate IDs")

    # ID format
    malformed = 0
    for sid in sub_df["ID"]:
        try:
            season, lo, hi = parse_submission_id(sid)
            if lo >= hi:
                issues.append(f"team_id_low >= team_id_high in ID: {sid}")
                break
            if expected_season and season != expected_season:
                issues.append(f"Season mismatch in ID {sid}: expected {expected_season}")
                break
        except ValueError as exc:
            malformed += 1
            if malformed == 1:
                issues.append(f"Malformed ID(s): {exc}")

    return {
        "valid": len(issues) == 0,
        "n_rows": len(sub_df),
        "issues": issues,
    }
