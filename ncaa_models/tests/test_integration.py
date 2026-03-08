"""
test_integration.py
===================
Integration tests for ncaa_models.

INTEGRATION TEST 5
------------------
Full train-evaluate cycle:
  - Train LogisticBaseline on seasons 2021-2024 tournament data
  - Predict 2025 tournament matchups
  - Assert Brier score < 0.20

INTEGRATION TEST 6
------------------
Generate mock 2026 submission CSV and validate format:
  - Correct columns: ['ID', 'Pred']
  - Correct row count: C(N, 2)
  - All probabilities in [0.01, 0.99]
  - validate_submission returns valid=True
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from ncaa_models.baseline import FEATURE_COLS, LogisticBaseline
from ncaa_models.cv import build_matchup_df, temporal_cross_validate
from ncaa_models.evaluate import compute_brier_score
from ncaa_models.submit import build_submission, validate_submission
from ncaa_models.tests.conftest import (
    ALL_SEASONS,
    M_TEAM_IDS,
    TRAIN_SEASONS,
    VAL_SEASON,
    W_TEAM_IDS,
    make_feat_df,
    make_tourney_df,
)


# ---------------------------------------------------------------------------
# INTEGRATION TEST 5: Full train-evaluate cycle — Brier < 0.20
# ---------------------------------------------------------------------------

class TestFullTrainEvaluateCycle:
    """
    INTEGRATION TEST 5:
    Train on seasons 2021-2024 tournament data, predict 2025 tournament,
    assert Brier score < 0.20.
    """

    def test_brier_below_threshold(self, fitted_model, val_arrays):
        """
        Core integration test: trained model must achieve Brier < 0.20
        on 2025 tournament holdout.
        """
        X1_val, X2_val, y_val = val_arrays
        probs = fitted_model.predict_proba(X1_val, X2_val)
        brier = compute_brier_score(probs, y_val)
        assert brier < 0.20, (
            f"Brier score {brier:.4f} exceeds target threshold of 0.20. "
            f"Model may not be learning from the data."
        )

    def test_brier_below_random_baseline(self, fitted_model, val_arrays):
        """Trained model must beat the random (always-0.5) baseline of 0.25."""
        X1_val, X2_val, y_val = val_arrays
        probs = fitted_model.predict_proba(X1_val, X2_val)
        brier = compute_brier_score(probs, y_val)
        assert brier < 0.25, (
            f"Brier score {brier:.4f} does not beat the 0.5-baseline (0.25). "
            f"Model is not learning."
        )

    def test_accuracy_above_50pct(self, fitted_model, val_arrays):
        """Model should correctly predict more than 50% of games."""
        X1_val, X2_val, y_val = val_arrays
        probs = fitted_model.predict_proba(X1_val, X2_val)
        predictions = (probs > 0.5).astype(int)
        accuracy = float(np.mean(predictions == y_val))
        assert accuracy > 0.5, (
            f"Accuracy {accuracy:.3f} is not better than coin flip."
        )

    def test_predictions_in_valid_range(self, fitted_model, val_arrays):
        """All predictions must be in [0.01, 0.99] (clipped)."""
        X1_val, X2_val, _ = val_arrays
        probs = fitted_model.predict_proba(X1_val, X2_val)
        assert (probs >= 0.01).all()
        assert (probs <= 0.99).all()

    def test_n_val_games_is_7(self, val_arrays):
        """8-team bracket has exactly 7 games."""
        X1_val, X2_val, y_val = val_arrays
        assert len(y_val) == 7

    def test_temporal_cv_all_folds_below_threshold(self, matchup_df):
        """
        Temporal CV across all 3 folds (2023, 2024, 2025) should each
        achieve Brier < 0.20.
        """
        def factory(X1, X2, y):
            m = LogisticBaseline()
            m.fit(X1, X2, y)
            return m

        result = temporal_cross_validate(
            factory, matchup_df, val_seasons=[2023, 2024, 2025]
        )
        for fold in result.folds:
            assert fold.brier_score < 0.20, (
                f"Fold val={fold.val_season}: Brier={fold.brier_score:.4f} >= 0.20"
            )

    def test_cv_mean_brier_below_threshold(self, matchup_df):
        def factory(X1, X2, y):
            m = LogisticBaseline()
            m.fit(X1, X2, y)
            return m

        result = temporal_cross_validate(
            factory, matchup_df, val_seasons=[2023, 2024, 2025]
        )
        assert result.mean_brier < 0.20, (
            f"Mean CV Brier {result.mean_brier:.4f} >= 0.20"
        )

    def test_women_brier_below_threshold(self):
        """Same test but using Women's team IDs."""
        w_feat = make_feat_df(W_TEAM_IDS, ALL_SEASONS)
        w_tourn = make_tourney_df(W_TEAM_IDS, ALL_SEASONS)
        w_matchup = build_matchup_df(w_feat, w_tourn, FEATURE_COLS, is_tourney=True)

        train_df = w_matchup[w_matchup["season"].isin(TRAIN_SEASONS)]
        val_df = w_matchup[w_matchup["season"] == VAL_SEASON]

        X1_tr = np.stack(train_df["X_team1"].values)
        X2_tr = np.stack(train_df["X_team2"].values)
        y_tr = train_df["y"].values.astype(int)

        X1_v = np.stack(val_df["X_team1"].values)
        X2_v = np.stack(val_df["X_team2"].values)
        y_v = val_df["y"].values.astype(int)

        model = LogisticBaseline()
        model.fit(X1_tr, X2_tr, y_tr)
        probs = model.predict_proba(X1_v, X2_v)
        brier = compute_brier_score(probs, y_v)
        assert brier < 0.20, f"Women's Brier {brier:.4f} >= 0.20"


# ---------------------------------------------------------------------------
# INTEGRATION TEST 6: Mock 2026 submission validation
# ---------------------------------------------------------------------------

SUBMISSION_SEASON = 2026


class TestMock2026Submission:
    """
    INTEGRATION TEST 6:
    Generate mock 2026 submission CSV; validate format, row count,
    column names, and probability range.
    """

    @pytest.fixture(scope="class")
    def feat_2026(self):
        """Synthetic 2026 team features (same quality structure as 2025)."""
        return make_feat_df(M_TEAM_IDS, [SUBMISSION_SEASON])

    @pytest.fixture(scope="class")
    def submission_df(self, fitted_model, feat_2026):
        """Full C(8,2)=28 matchup submission for 2026."""
        return build_submission(
            fitted_model,
            feat_2026,
            team_ids=M_TEAM_IDS,
            season=SUBMISSION_SEASON,
        )

    def test_submission_valid(self, fitted_model, feat_2026):
        """INTEGRATION TEST 6 — validate_submission must return valid=True."""
        df = build_submission(
            fitted_model,
            feat_2026,
            team_ids=M_TEAM_IDS,
            season=SUBMISSION_SEASON,
        )
        report = validate_submission(df, expected_season=SUBMISSION_SEASON)
        assert report["valid"] is True, (
            f"Submission validation failed: {report['issues']}"
        )

    def test_correct_columns(self, submission_df):
        assert list(submission_df.columns) == ["ID", "Pred"]

    def test_correct_row_count(self, submission_df):
        n = len(M_TEAM_IDS)
        expected = math.comb(n, 2)
        assert len(submission_df) == expected, (
            f"Expected C({n},2)={expected} rows, got {len(submission_df)}"
        )

    def test_all_probs_in_range(self, submission_df):
        """INTEGRATION TEST 6 — all probabilities in [0.01, 0.99]."""
        assert (submission_df["Pred"] >= 0.01).all(), (
            f"Some Pred values < 0.01: min={submission_df['Pred'].min():.4f}"
        )
        assert (submission_df["Pred"] <= 0.99).all(), (
            f"Some Pred values > 0.99: max={submission_df['Pred'].max():.4f}"
        )

    def test_all_ids_have_correct_season(self, submission_df):
        from ncaa_models.submit import parse_submission_id
        for sid in submission_df["ID"]:
            s, _, _ = parse_submission_id(sid)
            assert s == SUBMISSION_SEASON

    def test_lower_id_first_in_all_ids(self, submission_df):
        from ncaa_models.submit import parse_submission_id
        for sid in submission_df["ID"]:
            _, lo, hi = parse_submission_id(sid)
            assert lo < hi, f"ID '{sid}': lo >= hi"

    def test_no_duplicate_ids(self, submission_df):
        assert submission_df["ID"].duplicated().sum() == 0

    def test_sorted_by_id(self, submission_df):
        assert list(submission_df["ID"]) == sorted(submission_df["ID"].tolist())

    def test_csv_written_valid(self, fitted_model, feat_2026, tmp_path):
        """Write to CSV and re-read; validate loaded DataFrame."""
        csv_path = tmp_path / "submission_2026.csv"
        build_submission(
            fitted_model,
            feat_2026,
            team_ids=M_TEAM_IDS,
            season=SUBMISSION_SEASON,
            output_path=csv_path,
        )
        assert csv_path.exists()
        loaded = pd.read_csv(csv_path)
        report = validate_submission(loaded, expected_season=SUBMISSION_SEASON)
        assert report["valid"] is True, f"Loaded CSV invalid: {report['issues']}"

    def test_combined_men_women_submission(self, fitted_model, feat_2026, tmp_path):
        """Combine men's and women's predictions into one submission file."""
        w_feat_2026 = make_feat_df(W_TEAM_IDS, [SUBMISSION_SEASON])

        df_m = build_submission(
            fitted_model, feat_2026, M_TEAM_IDS, SUBMISSION_SEASON
        )
        df_w = build_submission(
            fitted_model, w_feat_2026, W_TEAM_IDS, SUBMISSION_SEASON
        )

        combined = (
            pd.concat([df_m, df_w], ignore_index=True)
            .sort_values("ID")
            .reset_index(drop=True)
        )

        n_m = math.comb(len(M_TEAM_IDS), 2)
        n_w = math.comb(len(W_TEAM_IDS), 2)
        assert len(combined) == n_m + n_w

        # Validate the combined submission
        report = validate_submission(combined, expected_season=SUBMISSION_SEASON)
        assert report["valid"] is True, f"Combined submission invalid: {report['issues']}"
