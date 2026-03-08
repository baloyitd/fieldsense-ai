"""
ncaa_models.submit
==================
Kaggle submission generator for NCAA tournament predictions.

Generates all C(N, 2) possible matchups for N tournament-eligible teams.

ID format
---------
"YYYY_TeamIdLow_TeamIdHigh"  — lower TeamId always first (Kaggle convention).

Usage
-----
::

    df = build_submission(
        model=fitted_model,
        team_features=feat_df,
        team_ids=[1001, 1002, ..., 1068],
        season=2026,
        feature_cols=FEATURE_COLS,
        output_path="submission_2026.csv",
    )
    report = validate_submission(df, expected_season=2026)
"""

from __future__ import annotations

import logging
from itertools import combinations
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .base import MatchupPredictor

logger = logging.getLogger(__name__)

PROB_CLIP_LOW: float = 0.01
PROB_CLIP_HIGH: float = 0.99


# ---------------------------------------------------------------------------
# ID helpers
# ---------------------------------------------------------------------------

def make_submission_id(season: int, team_id_low: int, team_id_high: int) -> str:
    """Return Kaggle-format submission ID string."""
    return f"{season}_{team_id_low}_{team_id_high}"


def parse_submission_id(sid: str) -> Tuple[int, int, int]:
    """
    Parse a submission ID string into (season, team_id_low, team_id_high).

    Raises ValueError on malformed input.
    """
    parts = sid.split("_")
    if len(parts) != 3:
        raise ValueError(
            f"Malformed submission ID '{sid}': expected format YYYY_LO_HI"
        )
    try:
        season, lo, hi = int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError:
        raise ValueError(
            f"Non-integer component in submission ID '{sid}'"
        )
    return season, lo, hi


# ---------------------------------------------------------------------------
# Submission builder
# ---------------------------------------------------------------------------

def build_submission(
    model: MatchupPredictor,
    team_features: pd.DataFrame,
    team_ids: Sequence[int],
    season: int,
    feature_cols: Optional[List[str]] = None,
    clip_probs: bool = True,
    output_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """
    Generate all C(N, 2) possible tournament matchup predictions.

    Parameters
    ----------
    model         : Fitted MatchupPredictor.
    team_features : DataFrame with columns ['team_id', 'season'] + feature_cols.
                    If 'season' column is absent, all rows are used.
    team_ids      : Sequence of tournament-eligible team IDs (any order).
    season        : Tournament season (e.g. 2026).
    feature_cols  : Ordered list of feature column names.
                    Falls back to model.feature_cols if the attribute exists.
    clip_probs    : Clip predictions to [0.01, 0.99] (default True).
    output_path   : Optional path to write CSV file.

    Returns
    -------
    pd.DataFrame with columns ['ID', 'Pred'], sorted by ID.
    """
    if feature_cols is None:
        feature_cols = getattr(model, "feature_cols", None)
        if feature_cols is None:
            raise ValueError(
                "feature_cols must be provided or available as model.feature_cols."
            )

    # Filter to requested season if column exists
    if "season" in team_features.columns:
        season_df = team_features[team_features["season"] == season]
    else:
        season_df = team_features

    feat_index = season_df.set_index("team_id")[feature_cols]

    team_ids_sorted = sorted(team_ids)
    all_pairs = list(combinations(team_ids_sorted, 2))

    if not all_pairs:
        return pd.DataFrame(columns=["ID", "Pred"])

    # Filter pairs where both teams have features; batch the rest
    valid_pairs: List[Tuple[int, int]] = []
    X1_list: List[np.ndarray] = []
    X2_list: List[np.ndarray] = []

    for lo, hi in all_pairs:
        if lo not in feat_index.index or hi not in feat_index.index:
            logger.debug("Missing features for team %d or %d; skipping.", lo, hi)
            continue
        valid_pairs.append((lo, hi))
        X1_list.append(feat_index.loc[lo].values.astype(float))
        X2_list.append(feat_index.loc[hi].values.astype(float))

    if not valid_pairs:
        logger.warning("No valid pairs found for season %d.", season)
        return pd.DataFrame(columns=["ID", "Pred"])

    X1 = np.array(X1_list, dtype=float)
    X2 = np.array(X2_list, dtype=float)
    preds = model.predict_proba(X1, X2)

    if clip_probs:
        preds = np.clip(preds, PROB_CLIP_LOW, PROB_CLIP_HIGH)

    rows = [
        {"ID": make_submission_id(season, lo, hi), "Pred": float(pred)}
        for (lo, hi), pred in zip(valid_pairs, preds)
    ]

    df = (
        pd.DataFrame(rows, columns=["ID", "Pred"])
        .sort_values("ID")
        .reset_index(drop=True)
    )

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logger.info("Submission CSV written to %s (%d rows).", path, len(df))

    return df


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_submission(
    sub_df: pd.DataFrame,
    expected_season: Optional[int] = None,
    expected_n_rows: Optional[int] = None,
) -> dict:
    """
    Validate a submission DataFrame against Kaggle schema requirements.

    Parameters
    ----------
    sub_df          : DataFrame to validate.
    expected_season : If provided, all IDs must belong to this season.
    expected_n_rows : If provided, row count must match exactly.

    Returns
    -------
    dict with keys:
        valid    : bool   — True iff no issues found.
        n_rows   : int    — number of rows.
        issues   : list[str] — human-readable problem descriptions.
    """
    issues: List[str] = []

    # Required columns
    for col in ("ID", "Pred"):
        if col not in sub_df.columns:
            issues.append(f"Missing required column: '{col}'")

    if issues:
        return {"valid": False, "n_rows": len(sub_df), "issues": issues}

    # Pred range
    out_of_range = int(((sub_df["Pred"] < 0.0) | (sub_df["Pred"] > 1.0)).sum())
    if out_of_range > 0:
        issues.append(f"{out_of_range} Pred value(s) outside [0, 1]")

    # Duplicate IDs
    n_dupes = int(sub_df["ID"].duplicated().sum())
    if n_dupes > 0:
        issues.append(f"{n_dupes} duplicate ID(s)")

    # ID format, ordering, and season
    malformed = 0
    wrong_season = 0
    id_order_errors = 0

    for sid in sub_df["ID"]:
        try:
            s, lo, hi = parse_submission_id(str(sid))
            if lo >= hi:
                id_order_errors += 1
            if expected_season is not None and s != expected_season:
                wrong_season += 1
        except ValueError:
            malformed += 1

    if malformed > 0:
        issues.append(f"{malformed} malformed ID(s)")
    if id_order_errors > 0:
        issues.append(f"{id_order_errors} ID(s) where LowId >= HighId")
    if wrong_season > 0:
        issues.append(
            f"{wrong_season} ID(s) with wrong season (expected {expected_season})"
        )

    if expected_n_rows is not None and len(sub_df) != expected_n_rows:
        issues.append(
            f"Row count mismatch: got {len(sub_df)}, expected {expected_n_rows}"
        )

    return {
        "valid": len(issues) == 0,
        "n_rows": len(sub_df),
        "issues": issues,
    }
