"""
test_cv.py
==========
Tests for ncaa_models.cv (temporal cross-validation).

Covers
------
1. UNIT TEST 4: No data leakage — training set max season < validation season
   for every fold.
2. Correct fold seasons (train 2021-2022 → val 2023, etc.).
3. Number of completed folds matches val_seasons list.
4. CVResult attributes (mean_brier, std_brier, folds).
5. summary_df shape and content.
6. build_matchup_df produces correct rows and labels.
7. Insufficient training seasons skipped gracefully.
"""

from __future__ import annotations

import numpy as np
import pytest

from ncaa_models.baseline import LogisticBaseline
from ncaa_models.cv import (
    CVFold,
    CVResult,
    build_matchup_df,
    temporal_cross_validate,
)
from ncaa_models.evaluate import compute_brier_score
from ncaa_models.tests.conftest import (
    ALL_SEASONS,
    FEATURE_COLS,
    M_TEAM_IDS,
    TRAIN_SEASONS,
    VAL_SEASON,
    make_feat_df,
    make_tourney_df,
)


# ---------------------------------------------------------------------------
# Fixture: model factory
# ---------------------------------------------------------------------------

def _factory(X1, X2, y):
    """Create and train a LogisticBaseline."""
    m = LogisticBaseline()
    m.fit(X1, X2, y)
    return m


# ---------------------------------------------------------------------------
# UNIT TEST 4: No data leakage
# ---------------------------------------------------------------------------

class TestNoDataLeakage:
    """
    UNIT TEST 4: verify training set max season < validation season for every fold.
    """

    def test_max_train_season_lt_val_season(self, matchup_df):
        result = temporal_cross_validate(
            _factory, matchup_df, val_seasons=[2023, 2024, 2025]
        )
        for fold in result.folds:
            assert max(fold.train_seasons) < fold.val_season, (
                f"DATA LEAKAGE in fold: max(train)={max(fold.train_seasons)} "
                f">= val={fold.val_season}"
            )

    def test_val_season_not_in_train_seasons(self, matchup_df):
        result = temporal_cross_validate(
            _factory, matchup_df, val_seasons=[2023, 2024, 2025]
        )
        for fold in result.folds:
            assert fold.val_season not in fold.train_seasons, (
                f"Validation season {fold.val_season} appears in training seasons "
                f"{fold.train_seasons}"
            )

    def test_train_seasons_strictly_before_val(self, matchup_df):
        result = temporal_cross_validate(
            _factory, matchup_df, val_seasons=[2023, 2024, 2025]
        )
        for fold in result.folds:
            for s in fold.train_seasons:
                assert s < fold.val_season, (
                    f"Training season {s} is not strictly before val {fold.val_season}"
                )


# ---------------------------------------------------------------------------
# Fold seasons
# ---------------------------------------------------------------------------

class TestFoldSeasons:

    def test_fold_2023_trains_on_2021_2022(self, matchup_df):
        """Default fold: train 2021-2022 → validate 2023."""
        result = temporal_cross_validate(
            _factory, matchup_df, val_seasons=[2023, 2024, 2025]
        )
        fold_2023 = next(f for f in result.folds if f.val_season == 2023)
        assert fold_2023.train_seasons == [2021, 2022]

    def test_fold_2024_trains_on_2021_2022_2023(self, matchup_df):
        """Default fold: train 2021-2023 → validate 2024."""
        result = temporal_cross_validate(
            _factory, matchup_df, val_seasons=[2023, 2024, 2025]
        )
        fold_2024 = next(f for f in result.folds if f.val_season == 2024)
        assert fold_2024.train_seasons == [2021, 2022, 2023]

    def test_fold_2025_trains_on_2021_through_2024(self, matchup_df):
        """Default fold: train 2021-2024 → validate 2025."""
        result = temporal_cross_validate(
            _factory, matchup_df, val_seasons=[2023, 2024, 2025]
        )
        fold_2025 = next(f for f in result.folds if f.val_season == 2025)
        assert fold_2025.train_seasons == [2021, 2022, 2023, 2024]

    def test_n_folds_matches_val_seasons(self, matchup_df):
        result = temporal_cross_validate(
            _factory, matchup_df, val_seasons=[2023, 2024, 2025]
        )
        assert len(result.folds) == 3

    def test_custom_val_seasons(self, matchup_df):
        result = temporal_cross_validate(
            _factory, matchup_df, val_seasons=[2024, 2025]
        )
        assert len(result.folds) == 2
        val_seasons = [f.val_season for f in result.folds]
        assert 2024 in val_seasons
        assert 2025 in val_seasons


# ---------------------------------------------------------------------------
# CVResult structure
# ---------------------------------------------------------------------------

class TestCVResultStructure:

    def test_cv_result_type(self, matchup_df):
        result = temporal_cross_validate(_factory, matchup_df)
        assert isinstance(result, CVResult)

    def test_mean_brier_is_float(self, matchup_df):
        result = temporal_cross_validate(_factory, matchup_df)
        assert isinstance(result.mean_brier, float)

    def test_std_brier_is_float(self, matchup_df):
        result = temporal_cross_validate(_factory, matchup_df)
        assert isinstance(result.std_brier, float)

    def test_mean_brier_equals_fold_mean(self, matchup_df):
        result = temporal_cross_validate(_factory, matchup_df)
        expected = np.mean([f.brier_score for f in result.folds])
        assert abs(result.mean_brier - expected) < 1e-9

    def test_std_brier_equals_fold_std(self, matchup_df):
        result = temporal_cross_validate(_factory, matchup_df)
        expected = np.std([f.brier_score for f in result.folds])
        assert abs(result.std_brier - expected) < 1e-9

    def test_each_fold_has_predictions(self, matchup_df):
        result = temporal_cross_validate(_factory, matchup_df)
        for fold in result.folds:
            assert len(fold.predictions) == fold.n_val_games
            assert len(fold.actuals) == fold.n_val_games

    def test_fold_brier_matches_manual(self, matchup_df):
        result = temporal_cross_validate(_factory, matchup_df)
        for fold in result.folds:
            manual = compute_brier_score(fold.predictions, fold.actuals)
            assert abs(fold.brier_score - manual) < 1e-9

    def test_brier_scores_non_negative(self, matchup_df):
        result = temporal_cross_validate(_factory, matchup_df)
        for fold in result.folds:
            assert fold.brier_score >= 0.0


# ---------------------------------------------------------------------------
# summary_df
# ---------------------------------------------------------------------------

class TestSummaryDf:

    def test_summary_df_returns_dataframe(self, matchup_df):
        result = temporal_cross_validate(_factory, matchup_df)
        df = result.summary_df()
        assert hasattr(df, "columns")

    def test_summary_df_has_required_columns(self, matchup_df):
        result = temporal_cross_validate(_factory, matchup_df)
        df = result.summary_df()
        for col in ("train_seasons", "val_season", "brier_score", "n_val_games"):
            assert col in df.columns

    def test_summary_df_has_n_folds_plus_1_rows(self, matchup_df):
        """summary_df has one row per fold + one aggregate row."""
        result = temporal_cross_validate(_factory, matchup_df)
        df = result.summary_df()
        assert len(df) == len(result.folds) + 1

    def test_summary_df_aggregate_row_mean_brier(self, matchup_df):
        result = temporal_cross_validate(_factory, matchup_df)
        df = result.summary_df()
        # Last row is aggregate
        agg_row = df.iloc[-1]
        assert abs(float(agg_row["brier_score"]) - result.mean_brier) < 1e-9


# ---------------------------------------------------------------------------
# build_matchup_df
# ---------------------------------------------------------------------------

class TestBuildMatchupDf:

    @pytest.fixture(scope="class")
    def feat(self):
        return make_feat_df(M_TEAM_IDS, ALL_SEASONS)

    @pytest.fixture(scope="class")
    def tourn(self):
        return make_tourney_df(M_TEAM_IDS, ALL_SEASONS)

    def test_returns_dataframe(self, feat, tourn):
        df = build_matchup_df(feat, tourn, FEATURE_COLS)
        assert hasattr(df, "columns")

    def test_required_columns(self, feat, tourn):
        df = build_matchup_df(feat, tourn, FEATURE_COLS)
        for col in ("season", "team_id_low", "team_id_high", "X_team1", "X_team2", "y"):
            assert col in df.columns

    def test_row_count_per_season(self, feat, tourn):
        """7 games per 8-team bracket per season."""
        df = build_matchup_df(feat, tourn, FEATURE_COLS)
        for season in ALL_SEASONS:
            n = len(df[df["season"] == season])
            assert n == 7, f"Season {season}: expected 7 games, got {n}"

    def test_labels_binary(self, feat, tourn):
        df = build_matchup_df(feat, tourn, FEATURE_COLS)
        assert set(df["y"].unique()).issubset({0, 1})

    def test_lower_id_always_team1(self, feat, tourn):
        df = build_matchup_df(feat, tourn, FEATURE_COLS)
        assert (df["team_id_low"] < df["team_id_high"]).all()

    def test_x_arrays_have_correct_shape(self, feat, tourn):
        df = build_matchup_df(feat, tourn, FEATURE_COLS)
        for row in df.itertuples():
            assert row.X_team1.shape == (len(FEATURE_COLS),)
            assert row.X_team2.shape == (len(FEATURE_COLS),)

    def test_insufficient_min_train_seasons_skipped(self, matchup_df):
        """If min_train_seasons=3, fold 2023 (only 2 train seasons) is skipped."""
        result = temporal_cross_validate(
            _factory, matchup_df,
            val_seasons=[2023, 2024, 2025],
            min_train_seasons=3,
        )
        val_seasons = [f.val_season for f in result.folds]
        assert 2023 not in val_seasons
        assert 2024 in val_seasons
        assert 2025 in val_seasons
