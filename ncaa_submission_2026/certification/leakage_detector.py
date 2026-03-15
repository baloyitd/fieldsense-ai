"""
ncaa_submission_2026.certification.leakage_detector
====================================================
Stage 09 — Data leakage detection.

Verifies that no test-set (season 2026) data appears in any training
or feature engineering DataFrame.  Can inject deliberate leakage for
certification testing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Seasons used in training (may not reference test season)
TRAIN_SEASONS: List[int] = [2021, 2022, 2023, 2024]
VAL_SEASON: int = 2025
TEST_SEASON: int = 2026


@dataclass
class LeakageReport:
    """Result of a data leakage check."""

    context: str
    leakage_found: bool
    leaking_seasons: List[int] = field(default_factory=list)
    leaking_row_count: int = 0
    details: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "context": self.context,
            "leakage_found": self.leakage_found,
            "leaking_seasons": self.leaking_seasons,
            "leaking_row_count": self.leaking_row_count,
            "details": self.details,
        }


class DataLeakageDetector:
    """
    Scans DataFrames to detect test-set leakage into training data.

    Leakage is defined as any row with ``season >= test_season`` appearing
    in a DataFrame that is used for model training or feature engineering.

    Parameters
    ----------
    test_season : int
        Season considered the test/holdout set.  Default 2026.
    train_seasons : list of int, optional
        Allowed training seasons.  Default [2021, 2022, 2023, 2024].
    """

    def __init__(
        self,
        test_season: int = TEST_SEASON,
        train_seasons: Optional[List[int]] = None,
    ) -> None:
        self.test_season = test_season
        self.train_seasons = list(train_seasons or TRAIN_SEASONS)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_dataframe(
        self,
        df: pd.DataFrame,
        context: str = "unnamed_df",
        season_col: str = "season",
    ) -> LeakageReport:
        """
        Check a DataFrame for test-season leakage.

        Parameters
        ----------
        df : pd.DataFrame
        context : str
            Label for the report.
        season_col : str
            Name of the season column.

        Returns
        -------
        LeakageReport
        """
        if season_col not in df.columns:
            return LeakageReport(
                context=context,
                leakage_found=False,
                details=f"Column '{season_col}' not present; skipping check.",
            )

        leaked = df[df[season_col] >= self.test_season]
        if leaked.empty:
            return LeakageReport(
                context=context,
                leakage_found=False,
                details=f"No season >= {self.test_season} found. Clean.",
            )

        leaking_seasons = sorted(leaked[season_col].unique().tolist())
        report = LeakageReport(
            context=context,
            leakage_found=True,
            leaking_seasons=leaking_seasons,
            leaking_row_count=len(leaked),
            details=(
                f"LEAKAGE DETECTED: {len(leaked)} rows with season(s) "
                f"{leaking_seasons} in training data."
            ),
        )
        logger.warning("Leakage detected in %s: %s", context, report.details)
        return report

    def check_many(
        self,
        dataframes: Dict[str, pd.DataFrame],
        season_col: str = "season",
    ) -> List[LeakageReport]:
        """Check multiple named DataFrames."""
        return [
            self.check_dataframe(df, context=name, season_col=season_col)
            for name, df in dataframes.items()
        ]

    def any_leakage(self, reports: List[LeakageReport]) -> bool:
        """Return True if any report indicates leakage."""
        return any(r.leakage_found for r in reports)

    # ------------------------------------------------------------------
    # Injection helpers (for certification testing)
    # ------------------------------------------------------------------

    def inject_leakage(
        self,
        df: pd.DataFrame,
        n_rows: int = 5,
        season: Optional[int] = None,
        season_col: str = "season",
    ) -> pd.DataFrame:
        """
        Return a copy of *df* with *n_rows* injected test-season rows.

        Used to verify the detector correctly identifies leakage.

        Parameters
        ----------
        df : pd.DataFrame
        n_rows : int
        season : int, optional
            Injected season value.  Defaults to ``self.test_season``.
        season_col : str

        Returns
        -------
        pd.DataFrame with leakage rows appended.
        """
        injected_season = season if season is not None else self.test_season
        sample = df.head(n_rows).copy()
        if season_col in sample.columns:
            sample[season_col] = injected_season
        else:
            sample[season_col] = injected_season
        result = pd.concat([df, sample], ignore_index=True)
        logger.info(
            "Injected %d rows of season %d leakage into DataFrame",
            n_rows, injected_season,
        )
        return result

    def inject_and_detect(
        self,
        df: pd.DataFrame,
        context: str = "injection_test",
        n_rows: int = 5,
    ) -> tuple:
        """
        Inject leakage and immediately run detection.

        Returns
        -------
        (dirty_df, report) : (pd.DataFrame, LeakageReport)
        """
        dirty = self.inject_leakage(df, n_rows=n_rows)
        report = self.check_dataframe(dirty, context=context)
        return dirty, report
