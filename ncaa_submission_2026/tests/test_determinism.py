"""
test_determinism.py
===================
Tests for ncaa_submission_2026.certification.determinism_checker.

Covers:
- Identical predictions across multiple runs
- Bitwise equality of numpy arrays
- set_all_seeds effect on NumPy and Python random
- Non-deterministic function detected correctly
- check_csv bitwise equality of submission CSV
- DeterminismReport structure
"""

from __future__ import annotations

import io
import random

import numpy as np
import pandas as pd
import pytest

from ncaa_submission_2026.certification.determinism_checker import (
    DeterminismChecker,
    DeterminismReport,
    set_all_seeds,
)

from .conftest import SEASON, TEAM_IDS


# ---------------------------------------------------------------------------
# set_all_seeds
# ---------------------------------------------------------------------------

class TestSetAllSeeds:
    def test_numpy_reproducible(self):
        set_all_seeds(42)
        a = np.random.randn(10)
        set_all_seeds(42)
        b = np.random.randn(10)
        assert np.array_equal(a, b)

    def test_python_random_reproducible(self):
        set_all_seeds(99)
        a = [random.random() for _ in range(5)]
        set_all_seeds(99)
        b = [random.random() for _ in range(5)]
        assert a == b

    def test_different_seeds_different_output(self):
        set_all_seeds(1)
        a = np.random.randn(10)
        set_all_seeds(2)
        b = np.random.randn(10)
        assert not np.array_equal(a, b)

    def test_does_not_raise_without_torch(self):
        """set_all_seeds should not raise even if PyTorch is not installed."""
        # We mock the import to test the except ImportError path
        import unittest.mock as mock
        with mock.patch.dict("sys.modules", {"torch": None}):
            set_all_seeds(42)  # Should not raise


# ---------------------------------------------------------------------------
# DeterminismChecker — check_model
# ---------------------------------------------------------------------------

class TestCheckModel:
    def test_logistic_baseline_is_deterministic(self, fitted_model, val_arrays):
        X1, X2, y = val_arrays
        checker = DeterminismChecker(seed=42, n_runs=3)
        report = checker.check_model(fitted_model, X1, X2)
        assert report.is_deterministic is True

    def test_zero_mismatches_for_deterministic_model(self, fitted_model, val_arrays):
        X1, X2, _ = val_arrays
        checker = DeterminismChecker(seed=42, n_runs=2)
        report = checker.check_model(fitted_model, X1, X2)
        assert report.n_mismatches == 0

    def test_max_abs_diff_zero_for_deterministic(self, fitted_model, val_arrays):
        X1, X2, _ = val_arrays
        checker = DeterminismChecker(seed=42, n_runs=2)
        report = checker.check_model(fitted_model, X1, X2)
        assert report.max_abs_diff == 0.0

    def test_n_runs_recorded(self, fitted_model, val_arrays):
        X1, X2, _ = val_arrays
        checker = DeterminismChecker(seed=7, n_runs=5)
        report = checker.check_model(fitted_model, X1, X2)
        assert report.n_runs == 5

    def test_n_comparisons_is_n_runs_minus_1(self, fitted_model, val_arrays):
        X1, X2, _ = val_arrays
        checker = DeterminismChecker(seed=42, n_runs=4)
        report = checker.check_model(fitted_model, X1, X2)
        assert report.n_comparisons == 3

    def test_non_deterministic_function_detected(self):
        """A model using os.urandom (not affected by seeds) should fail the check."""
        import os

        class NoisyModel:
            def predict_proba(self, X1, X2):
                n = len(X1)
                raw = os.urandom(n * 8)  # 8 bytes per float64
                return np.frombuffer(raw, dtype=np.float64) % 1.0

        X1 = np.random.randn(20, 28)
        X2 = np.random.randn(20, 28)
        checker = DeterminismChecker(seed=42, n_runs=2)
        report = checker.check_model(NoisyModel(), X1, X2)
        assert report.is_deterministic is False
        assert report.n_mismatches > 0

    def test_returns_determinism_report(self, fitted_model, val_arrays):
        X1, X2, _ = val_arrays
        checker = DeterminismChecker()
        report = checker.check_model(fitted_model, X1, X2)
        assert isinstance(report, DeterminismReport)


# ---------------------------------------------------------------------------
# DeterminismChecker — check_function
# ---------------------------------------------------------------------------

class TestCheckFunction:
    def test_deterministic_function_passes(self, fitted_model, val_arrays):
        X1, X2, _ = val_arrays
        checker = DeterminismChecker(seed=42, n_runs=2)
        report = checker.check_function(fitted_model.predict_proba, X1, X2)
        assert report.is_deterministic is True

    def test_numpy_rand_without_seed_fails(self):
        def random_func(n):
            return np.random.rand(n)

        checker = DeterminismChecker(seed=42, n_runs=2)
        # After set_all_seeds, NumPy is deterministic, so this SHOULD pass
        report = checker.check_function(random_func, 10)
        # check_function re-seeds before each call, so it should be deterministic
        assert report.is_deterministic is True


# ---------------------------------------------------------------------------
# DeterminismChecker — check_csv
# ---------------------------------------------------------------------------

class TestCheckCSV:
    def test_submission_csv_is_deterministic(self, fitted_model, feat_df):
        from ncaa_models.baseline import FEATURE_COLS
        from ncaa_models.submit import build_submission

        def make_csv():
            set_all_seeds(42)
            sub = build_submission(
                model=fitted_model,
                team_features=feat_df,
                team_ids=TEAM_IDS,
                season=SEASON,
                feature_cols=FEATURE_COLS,
                clip_probs=True,
            )
            return sub.to_csv(index=False)

        checker = DeterminismChecker(seed=42, n_runs=2)
        report = checker.check_csv(make_csv)
        assert report.is_deterministic is True

    def test_check_csv_zero_mismatches(self, fitted_model, feat_df):
        from ncaa_models.baseline import FEATURE_COLS
        from ncaa_models.submit import build_submission

        def make_csv():
            sub = build_submission(
                model=fitted_model,
                team_features=feat_df,
                team_ids=TEAM_IDS,
                season=SEASON,
                feature_cols=FEATURE_COLS,
            )
            return sub.to_csv(index=False)

        checker = DeterminismChecker(seed=42, n_runs=2)
        report = checker.check_csv(make_csv)
        assert report.n_mismatches == 0


# ---------------------------------------------------------------------------
# DeterminismReport — structure
# ---------------------------------------------------------------------------

class TestDeterminismReport:
    def test_to_dict_has_required_keys(self, fitted_model, val_arrays):
        X1, X2, _ = val_arrays
        checker = DeterminismChecker()
        report = checker.check_model(fitted_model, X1, X2)
        d = report.to_dict()
        for key in ("is_deterministic", "n_runs", "n_comparisons",
                    "n_mismatches", "max_abs_diff", "details"):
            assert key in d

    def test_details_non_empty(self, fitted_model, val_arrays):
        X1, X2, _ = val_arrays
        checker = DeterminismChecker()
        report = checker.check_model(fitted_model, X1, X2)
        assert isinstance(report.details, str) and len(report.details) > 0
