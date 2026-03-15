"""
ncaa_live.regenerator
=====================
Stage 08 — Submission regeneration after each recalibration cycle.

SubmissionRegenerator produces a Kaggle-compliant CSV that:
  - Excludes matchups already completed (results are known).
  - Predicts probabilities for all remaining possible matchups.
  - Clips probabilities to [0.01, 0.99].
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Collection, List, Optional, Set, Union

import pandas as pd

from ncaa_models.baseline import FEATURE_COLS
from ncaa_models.submit import build_submission, validate_submission

logger = logging.getLogger(__name__)


class SubmissionRegenerator:
    """
    Regenerates a Kaggle-compliant submission after live recalibration.

    Parameters
    ----------
    model : MatchupPredictor
        Current fitted model.
    feature_store : pd.DataFrame
        Up-to-date team features (from BoxScoreIngester).
    feature_cols : list of str, optional
    season : int
    """

    def __init__(
        self,
        model,
        feature_store: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        season: int = 2025,
    ) -> None:
        self.model = model
        self.feature_store = feature_store.copy()
        self.feature_cols = list(feature_cols or FEATURE_COLS)
        self.season = season

    def update_model(self, new_model) -> None:
        """Swap in a newly recalibrated model."""
        self.model = new_model

    def update_feature_store(self, new_store: pd.DataFrame) -> None:
        """Swap in updated feature data from BoxScoreIngester."""
        self.feature_store = new_store.copy()

    def regenerate(
        self,
        completed_game_ids: Collection[str],
        eligible_team_ids: List[int],
        output_path: Optional[Union[str, Path]] = None,
    ) -> pd.DataFrame:
        """
        Build a fresh submission CSV for remaining matchups.

        Parameters
        ----------
        completed_game_ids : collection of str
            Kaggle submission IDs (``"YYYY_LO_HI"``) for finished games.
        eligible_team_ids : list of int
            Team IDs still eligible for future matchups.
        output_path : path, optional
            If given, write CSV to this path.

        Returns
        -------
        pd.DataFrame
            Columns ``['ID', 'Pred']``, sorted by ID.
        """
        completed_set: Set[str] = set(completed_game_ids)

        full_sub = build_submission(
            model=self.model,
            team_features=self.feature_store,
            team_ids=eligible_team_ids,
            season=self.season,
            feature_cols=self.feature_cols,
            clip_probs=True,
        )

        sub_df = full_sub[~full_sub["ID"].isin(completed_set)].reset_index(drop=True)

        logger.info(
            "Regenerated submission: %d total matchups, %d completed excluded, "
            "%d remaining",
            len(full_sub),
            len(completed_set),
            len(sub_df),
        )

        if output_path is not None:
            sub_df.to_csv(output_path, index=False)
            logger.info("Submission written to %s", output_path)

        return sub_df

    def validate(
        self,
        sub_df: pd.DataFrame,
        expected_n_rows: Optional[int] = None,
    ) -> dict:
        """Validate a submission DataFrame using ncaa_models.submit."""
        return validate_submission(
            sub_df,
            expected_season=self.season,
            expected_n_rows=expected_n_rows,
        )
