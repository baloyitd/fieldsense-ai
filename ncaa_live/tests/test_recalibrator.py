"""
test_recalibrator.py
====================
Unit tests for ncaa_live.recalibrator.RoundRecalibrator and AdapterVersion.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from ncaa_live.recalibrator import RoundRecalibrator, AdapterVersion, brier_score
from ncaa_models.baseline import LogisticBaseline, FEATURE_COLS

from .conftest import SEASON, TEAM_IDS


# ---------------------------------------------------------------------------
# brier_score helper
# ---------------------------------------------------------------------------

class TestBrierScore:
    def test_perfect_predictions(self):
        y = np.array([1, 0, 1, 0])
        p = np.array([0.99, 0.01, 0.99, 0.01])
        bs = brier_score(p, y)
        assert bs < 0.01

    def test_worst_case(self):
        y = np.array([1, 0])
        p = np.array([0.0, 1.0])
        bs = brier_score(p, y)
        assert abs(bs - 1.0) < 1e-9

    def test_random_guessing(self):
        rng = np.random.RandomState(42)
        y = rng.randint(0, 2, size=1000)
        p = np.full(1000, 0.5)
        bs = brier_score(p, y)
        assert abs(bs - 0.25) < 0.02

    def test_output_is_float(self):
        assert isinstance(brier_score(np.array([0.5]), np.array([1])), float)


# ---------------------------------------------------------------------------
# AdapterVersion
# ---------------------------------------------------------------------------

class TestAdapterVersion:
    def test_fields_accessible(self, fitted_model):
        v = AdapterVersion(round_num=1, model=fitted_model, brier=0.15)
        assert v.round_num == 1
        assert v.brier == 0.15
        assert v.model is fitted_model

    def test_default_elapsed_zero(self, fitted_model):
        v = AdapterVersion(round_num=0, model=fitted_model, brier=0.10)
        assert v.elapsed_seconds == 0.0

    def test_timestamp_is_string(self, fitted_model):
        v = AdapterVersion(round_num=0, model=fitted_model, brier=0.10)
        assert isinstance(v.timestamp, str)

    def test_metadata_default_empty(self, fitted_model):
        v = AdapterVersion(round_num=0, model=fitted_model, brier=0.10)
        assert v.metadata == {}


# ---------------------------------------------------------------------------
# RoundRecalibrator — set_baseline
# ---------------------------------------------------------------------------

class TestSetBaseline:
    def test_set_baseline_records_version(self, recalibrator, fitted_model, val_arrays):
        X1, X2, y = val_arrays
        v = recalibrator.set_baseline(fitted_model, X1, X2, y, round_num=0)
        assert recalibrator.current_version() is v
        assert v.round_num == 0

    def test_baseline_brier_is_positive(self, recalibrator, fitted_model, val_arrays):
        X1, X2, y = val_arrays
        v = recalibrator.set_baseline(fitted_model, X1, X2, y)
        assert v.brier >= 0.0

    def test_baseline_brier_is_finite(self, recalibrator, fitted_model, val_arrays):
        X1, X2, y = val_arrays
        v = recalibrator.set_baseline(fitted_model, X1, X2, y)
        assert np.isfinite(v.brier)

    def test_baseline_metadata_label(self, recalibrator, fitted_model, val_arrays):
        X1, X2, y = val_arrays
        v = recalibrator.set_baseline(fitted_model, X1, X2, y)
        assert v.metadata.get("label") == "baseline"


# ---------------------------------------------------------------------------
# RoundRecalibrator — recalibrate
# ---------------------------------------------------------------------------

class TestRecalibrate:
    def _run_recalibration(self, recalibrator, train_arrays, val_arrays, round_num=1):
        X1_tr, X2_tr, y_tr = train_arrays
        X1_v, X2_v, y_v = val_arrays
        # Generate "tournament" data: first 4 games from training set
        res_X1 = X1_tr[:4]
        res_X2 = X2_tr[:4]
        res_y = y_tr[:4]
        return recalibrator.recalibrate(
            results_X1=res_X1,
            results_X2=res_X2,
            results_y=res_y,
            X1_val=X1_v,
            X2_val=X2_v,
            y_val=y_v,
            X1_hist=X1_tr,
            X2_hist=X2_tr,
            y_hist=y_tr,
            round_num=round_num,
        )

    def test_recalibrate_returns_adapter_version(self, recalibrator, train_arrays, val_arrays):
        v = self._run_recalibration(recalibrator, train_arrays, val_arrays)
        assert isinstance(v, AdapterVersion)

    def test_recalibrate_brier_is_finite(self, recalibrator, train_arrays, val_arrays):
        v = self._run_recalibration(recalibrator, train_arrays, val_arrays)
        assert np.isfinite(v.brier)

    def test_recalibrate_appends_version(self, recalibrator, train_arrays, val_arrays):
        self._run_recalibration(recalibrator, train_arrays, val_arrays)
        assert len(recalibrator.version_history()) == 1

    def test_recalibrate_records_tournament_game_count(self, recalibrator, train_arrays, val_arrays):
        v = self._run_recalibration(recalibrator, train_arrays, val_arrays)
        assert v.metadata.get("n_tournament_games") == 4

    def test_recalibrate_model_can_predict(self, recalibrator, train_arrays, val_arrays):
        v = self._run_recalibration(recalibrator, train_arrays, val_arrays)
        X1_v, X2_v, y_v = val_arrays
        probs = v.model.predict_proba(X1_v, X2_v)
        assert len(probs) == len(y_v)
        assert np.all((probs >= 0.01) & (probs <= 0.99))

    def test_recalibrate_produces_different_weights(self, fitted_model, train_arrays, val_arrays):
        """
        Model retrained on historical + upset tournament data should have
        different coefficients from the pre-tournament baseline.
        """
        X1_tr, X2_tr, y_tr = train_arrays
        X1_v, X2_v, y_v = val_arrays

        # Tournament data: flip labels (all upsets) to maximally displace weights
        res_X1 = X1_tr[:7]
        res_X2 = X2_tr[:7]
        res_y = 1 - y_tr[:7]  # inverted outcomes

        recal = RoundRecalibrator(base_model=fitted_model)
        v = recal.recalibrate(
            results_X1=res_X1, results_X2=res_X2, results_y=res_y,
            X1_val=X1_v, X2_val=X2_v, y_val=y_v,
            X1_hist=X1_tr, X2_hist=X2_tr, y_hist=y_tr,
            round_num=1,
        )
        # Extract logistic regression coefficients
        orig_coef = fitted_model._pipeline.named_steps["lr"].coef_.ravel()
        new_coef = v.model._pipeline.named_steps["lr"].coef_.ravel()
        assert not np.allclose(orig_coef, new_coef), (
            "Recalibrated model should have different coefficients"
        )

    def test_recalibrate_without_history(self, fitted_model, train_arrays, val_arrays):
        """Recalibration should work with only tournament data (no hist)."""
        X1_tr, X2_tr, y_tr = train_arrays
        X1_v, X2_v, y_v = val_arrays
        recal = RoundRecalibrator(base_model=fitted_model)
        v = recal.recalibrate(
            results_X1=X1_tr[:7],
            results_X2=X2_tr[:7],
            results_y=y_tr[:7],
            X1_val=X1_v, X2_val=X2_v, y_val=y_v,
            round_num=1,
        )
        assert isinstance(v, AdapterVersion)
        assert np.isfinite(v.brier)

    def test_elapsed_seconds_recorded(self, recalibrator, train_arrays, val_arrays):
        v = self._run_recalibration(recalibrator, train_arrays, val_arrays)
        assert v.elapsed_seconds >= 0.0

    def test_recalibration_within_30_minutes(self, recalibrator, train_arrays, val_arrays):
        """Recalibration must complete within MAX_RECAL_SECONDS."""
        v = self._run_recalibration(recalibrator, train_arrays, val_arrays)
        assert v.elapsed_seconds < RoundRecalibrator.MAX_RECAL_SECONDS

    def test_multiple_rounds_version_history(self, recalibrator, train_arrays, val_arrays):
        for rnd in range(1, 4):
            self._run_recalibration(recalibrator, train_arrays, val_arrays, round_num=rnd)
        assert len(recalibrator.version_history()) == 3

    def test_compute_brier_matches_manual(self, recalibrator, fitted_model, val_arrays):
        X1, X2, y = val_arrays
        b = recalibrator.compute_brier(fitted_model, X1, X2, y)
        manual = brier_score(fitted_model.predict_proba(X1, X2), y)
        assert abs(b - manual) < 1e-12
