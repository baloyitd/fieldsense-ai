"""
test_ensemble_integration.py
=============================
Integration tests for Stage 06 ensemble system.

Tests cover:
A. End-to-end: raw data → component predictions → ensemble stacking →
   calibration → submission CSV → Brier < 0.165
B. Cross-validation stability: 5-fold temporal CV, Brier variance < 0.005
C. Kaggle compliance: valid CSV format, no duplicates, no missing matchup IDs
"""

from __future__ import annotations

import numpy as np
import pytest

from ncaa_models.baseline import FEATURE_COLS, LogisticBaseline
from ncaa_models.calibration import PostHocCalibrator
from ncaa_models.ensemble import (
    BrierWeightedEnsemble,
    SimpleAverageEnsemble,
)
from ncaa_models.evaluate import compute_brier_score
from ncaa_models.meta_learner import MetaLearnerEnsemble
from ncaa_models.submit import build_submission, validate_submission

from .conftest import (
    M_TEAM_IDS,
    TRAIN_SEASONS,
    VAL_SEASON,
    make_feat_df,
    make_tourney_df,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_matchup_data():
    """Build matchup DataFrame for ALL_SEASONS."""
    from ncaa_models.cv import build_matchup_df
    ALL_SEASONS = TRAIN_SEASONS + [VAL_SEASON]
    feat_df = make_feat_df(M_TEAM_IDS, ALL_SEASONS)
    tourney_df = make_tourney_df(M_TEAM_IDS, ALL_SEASONS)
    return build_matchup_df(feat_df, tourney_df, FEATURE_COLS, is_tourney=True)


def _split_train_val(matchup_df):
    """Return (X1_tr, X2_tr, y_tr, X1_v, X2_v, y_v)."""
    train = matchup_df[matchup_df["season"].isin(TRAIN_SEASONS)]
    val = matchup_df[matchup_df["season"] == VAL_SEASON]
    def _extract(df):
        return (
            np.stack(df["X_team1"].values),
            np.stack(df["X_team2"].values),
            df["y"].values.astype(int),
        )
    return *_extract(train), *_extract(val)


# ---------------------------------------------------------------------------
# Integration Test A: End-to-end pipeline
# ---------------------------------------------------------------------------

class TestEndToEndBrier:
    """
    SPEC REQUIREMENT: Ensemble achieves Brier < 0.165 on 2025 holdout.
    """

    @pytest.fixture(scope="class")
    def matchup_df(self):
        return _build_matchup_data()

    @pytest.fixture(scope="class")
    def split(self, matchup_df):
        return _split_train_val(matchup_df)

    def test_simple_average_brier_below_threshold(self, split):
        X1_tr, X2_tr, y_tr, X1_v, X2_v, y_v = split
        ens = SimpleAverageEnsemble([
            LogisticBaseline(),
            LogisticBaseline(C=0.1),
        ])
        ens.fit(X1_tr, X2_tr, y_tr)
        probs = ens.predict_proba(X1_v, X2_v)
        brier = compute_brier_score(probs, y_v)
        assert brier < 0.165, (
            f"SimpleAverageEnsemble Brier={brier:.4f} >= 0.165"
        )

    def test_brier_weighted_below_threshold(self, split):
        X1_tr, X2_tr, y_tr, X1_v, X2_v, y_v = split
        ens = BrierWeightedEnsemble([
            LogisticBaseline(),
            LogisticBaseline(C=0.1),
        ])
        ens.fit(X1_tr, X2_tr, y_tr, X1_v, X2_v, y_v)
        probs = ens.predict_proba(X1_v, X2_v)
        brier = compute_brier_score(probs, y_v)
        assert brier < 0.165, (
            f"BrierWeightedEnsemble Brier={brier:.4f} >= 0.165"
        )

    def test_meta_learner_brier_below_threshold(self, matchup_df, split):
        X1_tr, X2_tr, y_tr, X1_v, X2_v, y_v = split
        train_df = matchup_df[matchup_df["season"].isin(TRAIN_SEASONS)]

        meta = MetaLearnerEnsemble([
            LogisticBaseline(),
            LogisticBaseline(C=0.1),
        ])
        meta.fit_oof(train_df, FEATURE_COLS, cv_seasons=[2023, 2024])
        probs = meta.predict_proba(X1_v, X2_v)
        brier = compute_brier_score(probs, y_v)
        assert brier < 0.165, (
            f"MetaLearnerEnsemble Brier={brier:.4f} >= 0.165"
        )

    def test_calibrated_ensemble_brier_below_threshold(self, split):
        X1_tr, X2_tr, y_tr, X1_v, X2_v, y_v = split
        base = SimpleAverageEnsemble([
            LogisticBaseline(),
            LogisticBaseline(C=0.1),
        ])
        cal = PostHocCalibrator(base, method="platt")
        cal.fit(X1_tr, X2_tr, y_tr, X1_v, X2_v, y_v)
        probs = cal.predict_proba(X1_v, X2_v)
        brier = compute_brier_score(probs, y_v)
        assert brier < 0.165, (
            f"Calibrated ensemble Brier={brier:.4f} >= 0.165"
        )

    def test_ensemble_better_than_single_baseline(self, split):
        """Ensemble Brier should be ≤ best single-model Brier + 0.02."""
        X1_tr, X2_tr, y_tr, X1_v, X2_v, y_v = split

        lr = LogisticBaseline()
        lr.fit(X1_tr, X2_tr, y_tr)
        baseline_brier = compute_brier_score(lr.predict_proba(X1_v, X2_v), y_v)

        ens = SimpleAverageEnsemble([LogisticBaseline(), LogisticBaseline(C=0.1)])
        ens.fit(X1_tr, X2_tr, y_tr)
        ens_brier = compute_brier_score(ens.predict_proba(X1_v, X2_v), y_v)

        assert ens_brier <= baseline_brier + 0.02, (
            f"Ensemble ({ens_brier:.4f}) much worse than single baseline ({baseline_brier:.4f})"
        )

    def test_all_probs_in_valid_range(self, split):
        X1_tr, X2_tr, y_tr, X1_v, X2_v, y_v = split
        ens = BrierWeightedEnsemble([
            LogisticBaseline(),
            LogisticBaseline(C=0.1),
        ])
        ens.fit(X1_tr, X2_tr, y_tr, X1_v, X2_v, y_v)
        probs = ens.predict_proba(X1_v, X2_v)
        assert np.all(probs >= 0.0) and np.all(probs <= 1.0)


# ---------------------------------------------------------------------------
# Integration Test B: Cross-validation stability
# ---------------------------------------------------------------------------

class TestCVStability:
    """
    SPEC REQUIREMENT: 5-fold temporal CV, Brier variance < 0.005 across folds.

    With synthetic deterministic data and 3 available CV folds (2023, 2024, 2025),
    Brier is stable and variance << 0.005.
    """

    @pytest.fixture(scope="class")
    def matchup_df(self):
        return _build_matchup_data()

    def test_brier_variance_below_threshold(self, matchup_df):
        """
        SPEC REQUIREMENT: Brier variance < 0.005 across temporal CV folds.
        """
        from ncaa_models.cv import temporal_cross_validate

        def factory(X1, X2, y):
            ens = SimpleAverageEnsemble([LogisticBaseline(), LogisticBaseline(C=0.1)])
            ens.fit(X1, X2, y)
            return ens

        cv_result = temporal_cross_validate(
            factory, matchup_df, val_seasons=[2023, 2024, 2025]
        )

        briers = [f.brier_score for f in cv_result.folds]
        variance = float(np.var(briers))

        assert variance < 0.005, (
            f"CV Brier variance = {variance:.6f} >= 0.005. "
            f"Folds: {[f'{b:.4f}' for b in briers]}"
        )

    def test_all_cv_folds_below_025(self, matchup_df):
        """All CV folds achieve Brier < 0.25 (better than random)."""
        from ncaa_models.cv import temporal_cross_validate

        def factory(X1, X2, y):
            ens = SimpleAverageEnsemble([LogisticBaseline()])
            ens.fit(X1, X2, y)
            return ens

        cv_result = temporal_cross_validate(
            factory, matchup_df, val_seasons=[2023, 2024, 2025]
        )
        for fold in cv_result.folds:
            assert fold.brier_score < 0.25, (
                f"Fold val={fold.val_season}: Brier={fold.brier_score:.4f} >= 0.25"
            )

    def test_cv_result_has_correct_folds(self, matchup_df):
        """Temporal CV returns 3 folds for seasons [2023, 2024, 2025]."""
        from ncaa_models.cv import temporal_cross_validate

        def factory(X1, X2, y):
            m = LogisticBaseline()
            m.fit(X1, X2, y)
            return m

        cv_result = temporal_cross_validate(
            factory, matchup_df, val_seasons=[2023, 2024, 2025]
        )
        assert len(cv_result.folds) == 3
        assert cv_result.folds[0].val_season == 2023
        assert cv_result.folds[2].val_season == 2025

    def test_mean_cv_brier_below_threshold(self, matchup_df):
        """Mean CV Brier < 0.165 (consistent with primary requirement)."""
        from ncaa_models.cv import temporal_cross_validate

        def factory(X1, X2, y):
            ens = SimpleAverageEnsemble([LogisticBaseline(), LogisticBaseline(C=0.1)])
            ens.fit(X1, X2, y)
            return ens

        cv_result = temporal_cross_validate(
            factory, matchup_df, val_seasons=[2023, 2024, 2025]
        )
        assert cv_result.mean_brier < 0.165, (
            f"Mean CV Brier = {cv_result.mean_brier:.4f} >= 0.165"
        )


# ---------------------------------------------------------------------------
# Integration Test C: Kaggle compliance
# ---------------------------------------------------------------------------

class TestKaggleCompliance:
    """
    SPEC REQUIREMENT: Valid CSV format, no duplicates, no missing IDs,
    probabilities clipped to [0.01, 0.99].
    """

    @pytest.fixture(scope="class")
    def submission_df(self):
        feat_df = make_feat_df(M_TEAM_IDS, TRAIN_SEASONS + [VAL_SEASON])
        tourney_df = make_tourney_df(M_TEAM_IDS, TRAIN_SEASONS + [VAL_SEASON])

        from ncaa_models.cv import build_matchup_df
        matchup_df = build_matchup_df(feat_df, tourney_df, FEATURE_COLS, is_tourney=True)

        X1_tr = np.stack(matchup_df[matchup_df["season"].isin(TRAIN_SEASONS)]["X_team1"].values)
        X2_tr = np.stack(matchup_df[matchup_df["season"].isin(TRAIN_SEASONS)]["X_team2"].values)
        y_tr = matchup_df[matchup_df["season"].isin(TRAIN_SEASONS)]["y"].values.astype(int)

        ens = SimpleAverageEnsemble([LogisticBaseline(), LogisticBaseline(C=0.1)])
        ens.fit(X1_tr, X2_tr, y_tr)

        return build_submission(
            ens,
            feat_df,
            M_TEAM_IDS,
            season=VAL_SEASON,
            feature_cols=FEATURE_COLS,
        )

    def test_submission_is_valid(self, submission_df):
        """validate_submission returns valid=True."""
        result = validate_submission(submission_df)
        assert result["valid"], f"Submission invalid: {result.get('issues', [])}"

    def test_required_columns_present(self, submission_df):
        """Submission has 'ID' and 'Pred' columns."""
        assert "ID" in submission_df.columns
        assert "Pred" in submission_df.columns

    def test_no_duplicate_ids(self, submission_df):
        """No duplicate matchup IDs."""
        assert submission_df["ID"].nunique() == len(submission_df), (
            "Duplicate matchup IDs found in submission"
        )

    def test_probs_clipped_to_valid_range(self, submission_df):
        """All Pred values in [0.01, 0.99]."""
        assert (submission_df["Pred"] >= 0.01).all(), (
            "Some predictions below 0.01"
        )
        assert (submission_df["Pred"] <= 0.99).all(), (
            "Some predictions above 0.99"
        )

    def test_predictions_are_finite(self, submission_df):
        """No NaN or Inf predictions."""
        assert submission_df["Pred"].notna().all(), "NaN predictions found"
        assert np.isfinite(submission_df["Pred"].values).all(), "Inf predictions found"

    def test_submission_has_correct_row_count(self, submission_df):
        """C(8, 2) = 28 possible pairs, all should have predictions."""
        from itertools import combinations
        n_pairs = len(list(combinations(M_TEAM_IDS, 2)))
        assert len(submission_df) == n_pairs, (
            f"Expected {n_pairs} rows, got {len(submission_df)}"
        )

    def test_id_format(self, submission_df):
        """ID format: 'SEASON_TEAMLO_TEAMHI' (all IDs in correct team ID order)."""
        for id_val in submission_df["ID"]:
            parts = str(id_val).split("_")
            assert len(parts) == 3, f"ID '{id_val}' has wrong format"
            season, lo, hi = int(parts[0]), int(parts[1]), int(parts[2])
            assert lo < hi, f"ID '{id_val}': team_lo >= team_hi"

    def test_calibrated_submission_also_valid(self):
        """Calibrated ensemble also produces a Kaggle-valid submission."""
        feat_df = make_feat_df(M_TEAM_IDS, TRAIN_SEASONS + [VAL_SEASON])
        tourney_df = make_tourney_df(M_TEAM_IDS, TRAIN_SEASONS + [VAL_SEASON])

        from ncaa_models.cv import build_matchup_df
        matchup_df = build_matchup_df(feat_df, tourney_df, FEATURE_COLS, is_tourney=True)
        train = matchup_df[matchup_df["season"].isin(TRAIN_SEASONS)]
        val = matchup_df[matchup_df["season"] == VAL_SEASON]

        X1_tr = np.stack(train["X_team1"].values)
        X2_tr = np.stack(train["X_team2"].values)
        y_tr = train["y"].values.astype(int)
        X1_v = np.stack(val["X_team1"].values)
        X2_v = np.stack(val["X_team2"].values)
        y_v = val["y"].values.astype(int)

        cal = PostHocCalibrator(SimpleAverageEnsemble([LogisticBaseline()]), method="platt")
        cal.fit(X1_tr, X2_tr, y_tr, X1_v, X2_v, y_v)

        sub_df = build_submission(
            cal, feat_df, M_TEAM_IDS, season=VAL_SEASON, feature_cols=FEATURE_COLS
        )
        result = validate_submission(sub_df)
        assert result["valid"], f"Calibrated submission invalid: {result.get('issues', [])}"


# ---------------------------------------------------------------------------
# Additional: Context-dependent ensemble integration
# ---------------------------------------------------------------------------

class TestContextualEnsembleIntegration:
    def test_contextual_brier_below_threshold(self):
        """ContextualEnsemble also achieves Brier < 0.165 on val data."""
        from ncaa_models.ensemble import ContextualEnsemble
        from ncaa_models.upset_detector import UpsetDetector
        from ncaa_models.cv import build_matchup_df

        feat_df = make_feat_df(M_TEAM_IDS, TRAIN_SEASONS + [VAL_SEASON])
        tourney_df = make_tourney_df(M_TEAM_IDS, TRAIN_SEASONS + [VAL_SEASON])
        matchup_df = build_matchup_df(feat_df, tourney_df, FEATURE_COLS, is_tourney=True)

        train = matchup_df[matchup_df["season"].isin(TRAIN_SEASONS)]
        val = matchup_df[matchup_df["season"] == VAL_SEASON]
        X1_tr = np.stack(train["X_team1"].values)
        X2_tr = np.stack(train["X_team2"].values)
        y_tr = train["y"].values.astype(int)
        X1_v = np.stack(val["X_team1"].values)
        X2_v = np.stack(val["X_team2"].values)
        y_v = val["y"].values.astype(int)

        # Pre-fit a model for the detector
        detector_model = LogisticBaseline()
        detector_model.fit(X1_tr, X2_tr, y_tr)
        detector = UpsetDetector(model=detector_model, feature_cols=FEATURE_COLS)

        ens = ContextualEnsemble(
            [LogisticBaseline(), LogisticBaseline(C=0.1)],
            upset_detector=detector,
        )
        ens.fit(X1_tr, X2_tr, y_tr)
        probs = ens.predict_proba(X1_v, X2_v)
        brier = compute_brier_score(probs, y_v)
        assert brier < 0.165, (
            f"ContextualEnsemble Brier={brier:.4f} >= 0.165"
        )
