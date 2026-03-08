"""
test_evaluate.py
================
Tests for ncaa_model.evaluate:
  1. brier_score / log_loss correctness and edge cases.
  2. calibration_bins structure and boundary behaviour.
  3. EvaluationResult fields are correctly populated.
  4. evaluate_season produces a valid EvaluationResult with real data.
  5. cross_validate produces one result per validation season.
  6. summarise_cv includes a mean row.
  7. NCAAPredictor Protocol is runtime-checkable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ncaa_model.evaluate import (
    EvaluationResult,
    NCAAPredictor,
    brier_score,
    calibration_bins,
    cross_validate,
    evaluate_season,
    log_loss,
    summarise_cv,
)
from ncaa_model.baseline import LogisticBaseline
from ncaa_model.tests.conftest import (
    TRAIN_SEASONS, VAL_SEASON,
    make_feat_df, make_tourney_df,
)


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------

class TestBrierScore:

    def test_perfect_prediction_score_zero(self):
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1.0, 0.0, 1.0, 0.0])
        assert brier_score(y_true, y_pred) == pytest.approx(0.0)

    def test_worst_prediction_score_one(self):
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([0.0, 1.0, 0.0, 1.0])
        assert brier_score(y_true, y_pred) == pytest.approx(1.0)

    def test_random_prediction_near_025(self):
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 2, size=10000)
        y_pred = np.full(10000, 0.5)
        assert abs(brier_score(y_true, y_pred) - 0.25) < 0.01

    def test_brier_in_0_1(self):
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, size=200)
        y_pred = rng.uniform(0, 1, size=200)
        bs = brier_score(y_true, y_pred)
        assert 0.0 <= bs <= 1.0

    def test_single_game(self):
        assert brier_score(np.array([1]), np.array([0.8])) == pytest.approx(0.04)
        assert brier_score(np.array([0]), np.array([0.3])) == pytest.approx(0.09)

    def test_symmetry_in_predictions(self):
        y_true = np.array([1, 0])
        y_pred = np.array([0.7, 0.3])
        # BS is symmetric: same score regardless of ordering
        bs1 = brier_score(y_true, y_pred)
        bs2 = brier_score(y_true[::-1], y_pred[::-1])
        assert bs1 == pytest.approx(bs2)


class TestLogLoss:

    def test_perfect_prediction_near_zero(self):
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([0.9999, 0.0001, 0.9999, 0.0001])
        assert log_loss(y_true, y_pred) < 0.01

    def test_random_prediction_near_ln2(self):
        y_true = np.array([1, 0] * 5000)
        y_pred = np.full(10000, 0.5)
        assert abs(log_loss(y_true, y_pred) - np.log(2)) < 0.01

    def test_log_loss_positive(self):
        rng = np.random.default_rng(7)
        y_true = rng.integers(0, 2, size=100)
        y_pred = rng.uniform(0, 1, size=100)
        assert log_loss(y_true, y_pred) > 0


class TestCalibrationBins:

    def test_returns_list_of_dicts(self):
        y_true = np.array([1, 0, 1, 0, 1])
        y_pred = np.array([0.8, 0.2, 0.7, 0.3, 0.9])
        result = calibration_bins(y_true, y_pred)
        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, dict)

    def test_dict_keys(self):
        y_true = np.array([1, 0, 1, 0, 1])
        y_pred = np.array([0.8, 0.2, 0.7, 0.3, 0.9])
        result = calibration_bins(y_true, y_pred)
        for item in result:
            for key in ("bin_centre", "mean_pred", "fraction_pos", "n"):
                assert key in item, f"Missing key: {key}"

    def test_fraction_pos_between_0_1(self):
        rng = np.random.default_rng(1)
        y_true = rng.integers(0, 2, 200)
        y_pred = rng.uniform(0, 1, 200)
        for item in calibration_bins(y_true, y_pred):
            assert 0.0 <= item["fraction_pos"] <= 1.0

    def test_n_sums_to_total(self):
        n = 150
        y_true = np.zeros(n); y_true[:75] = 1
        y_pred  = np.linspace(0.01, 0.99, n)
        bins = calibration_bins(y_true, y_pred, n_bins=10)
        total_n = sum(b["n"] for b in bins)
        assert total_n == n

    def test_perfectly_calibrated_data(self):
        """Data where mean_pred ≈ fraction_pos in every bin."""
        rng = np.random.default_rng(9)
        y_pred = rng.uniform(0, 1, 2000)
        y_true = (rng.uniform(0, 1, 2000) < y_pred).astype(float)
        bins = calibration_bins(y_true, y_pred, n_bins=10)
        for b in bins:
            if b["n"] > 20:
                assert abs(b["mean_pred"] - b["fraction_pos"]) < 0.12, (
                    f"Calibration error: bin_centre={b['bin_centre']:.2f} "
                    f"mean_pred={b['mean_pred']:.3f} fraction_pos={b['fraction_pos']:.3f}"
                )


# ---------------------------------------------------------------------------
# evaluate_season
# ---------------------------------------------------------------------------

class TestEvaluateSeason:

    def test_returns_evaluation_result(self, fitted_model, feat_df, tourney_df):
        result = evaluate_season(fitted_model, feat_df, tourney_df, VAL_SEASON)
        assert isinstance(result, EvaluationResult)

    def test_brier_in_0_1(self, fitted_model, feat_df, tourney_df):
        result = evaluate_season(fitted_model, feat_df, tourney_df, VAL_SEASON)
        assert 0.0 <= result.brier_score <= 1.0

    def test_log_loss_positive(self, fitted_model, feat_df, tourney_df):
        result = evaluate_season(fitted_model, feat_df, tourney_df, VAL_SEASON)
        assert result.log_loss > 0

    def test_accuracy_in_0_1(self, fitted_model, feat_df, tourney_df):
        result = evaluate_season(fitted_model, feat_df, tourney_df, VAL_SEASON)
        assert 0.0 <= result.accuracy <= 1.0

    def test_n_games_matches_tourney(self, fitted_model, feat_df, tourney_df):
        result = evaluate_season(fitted_model, feat_df, tourney_df, VAL_SEASON)
        expected_games = tourney_df[
            (tourney_df["season"] == VAL_SEASON) & tourney_df["won"]
        ]["season"].count()
        assert result.n_games <= expected_games   # some may be skipped on missing feats

    def test_predictions_df_columns(self, fitted_model, feat_df, tourney_df):
        result = evaluate_season(fitted_model, feat_df, tourney_df, VAL_SEASON)
        for col in ("season", "gender", "y_true", "y_pred", "correct"):
            assert col in result.predictions_df.columns

    def test_predictions_df_length(self, fitted_model, feat_df, tourney_df):
        result = evaluate_season(fitted_model, feat_df, tourney_df, VAL_SEASON)
        assert len(result.predictions_df) == result.n_games

    def test_y_pred_in_0_1(self, fitted_model, feat_df, tourney_df):
        result = evaluate_season(fitted_model, feat_df, tourney_df, VAL_SEASON)
        y_pred = result.predictions_df["y_pred"].values
        assert (y_pred >= 0).all() and (y_pred <= 1).all()

    def test_y_true_binary(self, fitted_model, feat_df, tourney_df):
        result = evaluate_season(fitted_model, feat_df, tourney_df, VAL_SEASON)
        y_true = result.predictions_df["y_true"].values
        assert set(y_true).issubset({0, 1})

    def test_calibration_present(self, fitted_model, feat_df, tourney_df):
        result = evaluate_season(fitted_model, feat_df, tourney_df, VAL_SEASON)
        assert isinstance(result.calibration, list)

    def test_invalid_season_raises(self, fitted_model, feat_df, tourney_df):
        with pytest.raises(ValueError):
            evaluate_season(fitted_model, feat_df, tourney_df, val_season=1900)

    def test_gender_filter(self, fitted_model, feat_df, tourney_df):
        result_m = evaluate_season(
            fitted_model, feat_df, tourney_df, VAL_SEASON, gender="M"
        )
        result_w = evaluate_season(
            fitted_model, feat_df, tourney_df, VAL_SEASON, gender="W"
        )
        assert (result_m.predictions_df["gender"] == "M").all()
        assert (result_w.predictions_df["gender"] == "W").all()

    def test_brier_better_than_naive(self, fitted_model, feat_df, tourney_df):
        """Model should beat the always-0.5 naive baseline (Brier=0.25)."""
        result = evaluate_season(fitted_model, feat_df, tourney_df, VAL_SEASON)
        assert result.brier_score < 0.25


# ---------------------------------------------------------------------------
# cross_validate
# ---------------------------------------------------------------------------

class TestCrossValidate:

    @pytest.fixture(scope="class")
    def cv_results(self, feat_df, tourney_df):
        def factory(fd, td, seasons):
            m = LogisticBaseline()
            m.fit(fd, td, seasons)
            return m

        return cross_validate(
            factory, feat_df, tourney_df,
            val_seasons=[VAL_SEASON],
            train_lookback=4,
        )

    def test_returns_dict(self, cv_results):
        assert isinstance(cv_results, dict)

    def test_one_result_per_val_season(self, cv_results):
        assert VAL_SEASON in cv_results

    def test_result_type(self, cv_results):
        for result in cv_results.values():
            assert isinstance(result, EvaluationResult)

    def test_all_seasons_evaluated(self, feat_df, tourney_df):
        val_seasons = [2024, 2025]

        def factory(fd, td, seasons):
            m = LogisticBaseline()
            m.fit(fd, td, seasons)
            return m

        results = cross_validate(
            factory, feat_df, tourney_df,
            val_seasons=val_seasons, train_lookback=3,
        )
        for vs in val_seasons:
            assert vs in results, f"Missing CV result for season {vs}"

    def test_no_future_leakage(self, feat_df, tourney_df):
        """Training seasons must all precede the validation season."""
        trained_on: dict = {}

        def factory(fd, td, seasons):
            m = LogisticBaseline()
            m.fit(fd, td, seasons)
            trained_on[m._train_seasons[0]] = seasons
            return m

        cross_validate(
            factory, feat_df, tourney_df,
            val_seasons=[VAL_SEASON], train_lookback=4,
        )
        for val_s, train_ss in trained_on.items():
            assert all(ts < VAL_SEASON for ts in train_ss), (
                f"Training on season >= val_season={VAL_SEASON}: {train_ss}"
            )


# ---------------------------------------------------------------------------
# summarise_cv
# ---------------------------------------------------------------------------

class TestSummariseCV:

    @pytest.fixture(scope="class")
    def cv_results(self, feat_df, tourney_df):
        def factory(fd, td, seasons):
            m = LogisticBaseline()
            m.fit(fd, td, seasons)
            return m
        return cross_validate(
            factory, feat_df, tourney_df,
            val_seasons=[2024, 2025], train_lookback=3,
        )

    def test_returns_dataframe(self, cv_results):
        df = summarise_cv(cv_results)
        assert isinstance(df, pd.DataFrame)

    def test_has_mean_row(self, cv_results):
        df = summarise_cv(cv_results)
        assert "mean" in df["season"].values

    def test_columns_present(self, cv_results):
        df = summarise_cv(cv_results)
        for col in ("season", "brier_score", "log_loss", "accuracy", "n_games"):
            assert col in df.columns

    def test_mean_brier_in_0_1(self, cv_results):
        df = summarise_cv(cv_results)
        mean_row = df[df["season"] == "mean"].iloc[0]
        assert 0.0 <= mean_row["brier_score"] <= 1.0


# ---------------------------------------------------------------------------
# Protocol runtime check
# ---------------------------------------------------------------------------

class TestNCAAPredictor:

    def test_logistic_baseline_satisfies_protocol(self, fitted_model):
        assert isinstance(fitted_model, NCAAPredictor)

    def test_unfitted_model_also_satisfies_protocol(self):
        """Protocol check is structural (duck-typing), not state-based."""
        model = LogisticBaseline()
        assert isinstance(model, NCAAPredictor)

    def test_arbitrary_object_does_not_satisfy(self):
        class NotAModel:
            pass
        assert not isinstance(NotAModel(), NCAAPredictor)
