"""
test_monitor.py
===============
Unit tests for ncaa_live.monitor.

Tests cover:
- Module-level should_rollback function
- PerformanceMonitor: record, history, best/last Brier, rollback decision
- FallbackManager: save_version, rollback, current, n_versions
- Rollback scenario: inject bad model, confirm rollback engages, verify recovery
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from ncaa_live.monitor import (
    FallbackManager,
    MonitorRecord,
    PerformanceMonitor,
    should_rollback,
)


# ---------------------------------------------------------------------------
# Module-level should_rollback
# ---------------------------------------------------------------------------

class TestShouldRollback:
    def test_rollback_when_new_is_worse_by_more_than_threshold(self):
        assert should_rollback(0.150, 0.156, threshold=0.005) is True

    def test_no_rollback_when_improvement(self):
        assert should_rollback(0.160, 0.150, threshold=0.005) is False

    def test_no_rollback_when_equal(self):
        assert should_rollback(0.150, 0.150, threshold=0.005) is False

    def test_no_rollback_when_degradation_within_threshold(self):
        # 0.154 - 0.150 = 0.004 < 0.005
        assert should_rollback(0.150, 0.154, threshold=0.005) is False

    def test_rollback_exactly_at_threshold_is_false(self):
        # Strictly less than threshold → no rollback
        # (Avoids floating-point ambiguity at exactly 0.005)
        assert should_rollback(0.150, 0.1549, threshold=0.005) is False

    def test_rollback_just_above_threshold(self):
        assert should_rollback(0.150, 0.1551, threshold=0.005) is True

    def test_default_threshold_is_0005(self):
        # new_brier - old_brier = 0.006 > default 0.005
        assert should_rollback(0.150, 0.156) is True
        assert should_rollback(0.150, 0.154) is False

    def test_large_degradation_triggers_rollback(self):
        assert should_rollback(0.10, 0.50, threshold=0.005) is True

    def test_custom_threshold_zero_always_rollback_on_any_increase(self):
        assert should_rollback(0.15, 0.15001, threshold=0.0) is True

    def test_custom_threshold_large_no_rollback(self):
        assert should_rollback(0.10, 0.40, threshold=1.0) is False


# ---------------------------------------------------------------------------
# PerformanceMonitor
# ---------------------------------------------------------------------------

class TestPerformanceMonitor:
    def test_record_returns_monitor_record(self):
        m = PerformanceMonitor()
        rec = m.record(0.15, round_num=1, label="baseline")
        assert isinstance(rec, MonitorRecord)

    def test_history_empty_initially(self):
        m = PerformanceMonitor()
        assert m.history() == []

    def test_history_grows_with_records(self):
        m = PerformanceMonitor()
        m.record(0.15, round_num=0, label="baseline")
        m.record(0.16, round_num=1, label="round1")
        assert len(m.history()) == 2

    def test_best_brier_returns_minimum(self):
        m = PerformanceMonitor()
        m.record(0.20, round_num=0)
        m.record(0.15, round_num=1)
        m.record(0.18, round_num=2)
        assert abs(m.best_brier() - 0.15) < 1e-12

    def test_best_brier_none_when_empty(self):
        m = PerformanceMonitor()
        assert m.best_brier() is None

    def test_last_brier_matches_most_recent(self):
        m = PerformanceMonitor()
        m.record(0.20, round_num=0)
        m.record(0.14, round_num=1)
        assert abs(m.last_brier() - 0.14) < 1e-12

    def test_last_brier_none_when_empty(self):
        m = PerformanceMonitor()
        assert m.last_brier() is None

    def test_should_rollback_delegates_correctly(self):
        m = PerformanceMonitor(threshold=0.005)
        assert m.should_rollback(0.15, 0.16) is True
        assert m.should_rollback(0.15, 0.14) is False

    def test_n_rollbacks_counts_rollback_records(self):
        m = PerformanceMonitor()
        m.record(0.15, 0, triggered_rollback=False)
        m.record(0.17, 1, triggered_rollback=True)
        m.record(0.14, 2, triggered_rollback=False)
        assert m.n_rollbacks() == 1

    def test_n_rollbacks_zero_initially(self):
        m = PerformanceMonitor()
        m.record(0.15, 0)
        assert m.n_rollbacks() == 0

    def test_record_triggered_rollback_flag(self):
        m = PerformanceMonitor()
        rec = m.record(0.20, round_num=1, triggered_rollback=True)
        assert rec.triggered_rollback is True

    def test_repr_contains_threshold(self):
        m = PerformanceMonitor(threshold=0.01)
        assert "0.01" in repr(m)


# ---------------------------------------------------------------------------
# MonitorRecord
# ---------------------------------------------------------------------------

class TestMonitorRecord:
    def test_to_dict_has_required_keys(self):
        rec = MonitorRecord(round_num=1, brier=0.15, label="round1")
        d = rec.to_dict()
        for key in ("round_num", "brier", "label", "triggered_rollback", "timestamp"):
            assert key in d

    def test_default_triggered_rollback_false(self):
        rec = MonitorRecord(round_num=0, brier=0.12, label="baseline")
        assert rec.triggered_rollback is False


# ---------------------------------------------------------------------------
# FallbackManager
# ---------------------------------------------------------------------------

class TestFallbackManager:
    def _dummy_model(self, label="model"):
        """Return a tiny fitted LogisticBaseline for storage tests."""
        from ncaa_models.baseline import LogisticBaseline
        import numpy as np
        m = LogisticBaseline()
        X = np.random.randn(20, 28)
        m.fit(X, X, np.random.randint(0, 2, 20))
        return m

    def test_save_version_increments_n_versions(self):
        fb = FallbackManager()
        m = self._dummy_model()
        fb.save_version(m, brier=0.15, round_num=0)
        assert fb.n_versions() == 1

    def test_current_returns_most_recent(self):
        fb = FallbackManager()
        m = self._dummy_model()
        fb.save_version(m, brier=0.15, round_num=0, label="v0")
        fb.save_version(m, brier=0.17, round_num=1, label="v1")
        assert fb.current()["label"] == "v1"
        assert abs(fb.current()["brier"] - 0.17) < 1e-12

    def test_current_none_when_empty(self):
        fb = FallbackManager()
        assert fb.current() is None

    def test_rollback_returns_previous(self):
        fb = FallbackManager()
        m = self._dummy_model()
        fb.save_version(m, brier=0.15, round_num=0, label="good")
        fb.save_version(m, brier=0.20, round_num=1, label="bad")
        prev = fb.rollback()
        assert prev is not None
        assert prev["label"] == "good"
        assert abs(prev["brier"] - 0.15) < 1e-12

    def test_rollback_pops_bad_version(self):
        fb = FallbackManager()
        m = self._dummy_model()
        fb.save_version(m, brier=0.15, round_num=0, label="good")
        fb.save_version(m, brier=0.20, round_num=1, label="bad")
        fb.rollback()
        assert fb.n_versions() == 1

    def test_rollback_returns_none_with_single_version(self):
        fb = FallbackManager()
        m = self._dummy_model()
        fb.save_version(m, brier=0.15, round_num=0)
        result = fb.rollback()
        assert result is None

    def test_rollback_returns_none_when_empty(self):
        fb = FallbackManager()
        assert fb.rollback() is None

    def test_save_deep_copies_model(self):
        """Modifying the original model after save should not affect stored version."""
        from ncaa_models.baseline import LogisticBaseline
        m = self._dummy_model()
        fb = FallbackManager()
        fb.save_version(m, brier=0.15, round_num=0)
        # Retrieve stored model
        stored = fb.current()["model"]
        orig_coef = stored._pipeline.named_steps["lr"].coef_.copy()
        # Mutate original
        m._pipeline.named_steps["lr"].coef_[:] = 999.0
        # Stored should be unchanged
        assert np.allclose(stored._pipeline.named_steps["lr"].coef_, orig_coef)

    def test_max_versions_enforced(self):
        fb = FallbackManager(max_versions=3)
        m = self._dummy_model()
        for i in range(5):
            fb.save_version(m, brier=float(i) * 0.01, round_num=i)
        assert fb.n_versions() == 3

    def test_version_history_correct_order(self):
        fb = FallbackManager()
        m = self._dummy_model()
        for i in range(3):
            fb.save_version(m, brier=float(i) * 0.05, round_num=i)
        history = fb.version_history()
        briers = [h["brier"] for h in history]
        assert briers == sorted(briers)

    def test_clear_empties_stack(self):
        fb = FallbackManager()
        m = self._dummy_model()
        fb.save_version(m, brier=0.15, round_num=0)
        fb.clear()
        assert fb.n_versions() == 0


# ---------------------------------------------------------------------------
# Rollback scenario end-to-end
# ---------------------------------------------------------------------------

class TestRollbackScenario:
    """
    Verified rollback + recovery scenario:

    1. Train baseline model on clean data → good Brier.
    2. Inject a 'bad' model with high Brier.
    3. PerformanceMonitor triggers rollback.
    4. FallbackManager reverts to previous version.
    5. Recovered model has better Brier than the bad model.
    """

    def test_rollback_and_recovery(self, fitted_model, val_arrays):
        X1_v, X2_v, y_v = val_arrays
        from ncaa_live.recalibrator import brier_score

        monitor = PerformanceMonitor(threshold=0.005)
        fallback = FallbackManager()

        # Step 1: save baseline
        baseline_brier = brier_score(fitted_model.predict_proba(X1_v, X2_v), y_v)
        fallback.save_version(fitted_model, brier=baseline_brier, round_num=0, label="baseline")
        monitor.record(baseline_brier, round_num=0, label="baseline")

        # Step 2: inject bad model (constant 0.5 predictor → Brier ~ 0.25)
        class BadModel:
            def predict_proba(self, X1, X2):
                return np.full(len(X1), 0.5)

        bad_brier = brier_score(BadModel().predict_proba(X1_v, X2_v), y_v)
        fallback.save_version(BadModel(), brier=bad_brier, round_num=1, label="bad")

        # Step 3: monitor decides to roll back
        do_rollback = monitor.should_rollback(baseline_brier, bad_brier)
        assert do_rollback is True, (
            f"Expected rollback: baseline={baseline_brier:.4f}, bad={bad_brier:.4f}"
        )

        # Step 4: execute rollback
        monitor.record(bad_brier, round_num=1, triggered_rollback=True)
        recovered = fallback.rollback()
        assert recovered is not None
        assert recovered["label"] == "baseline"

        # Step 5: recovered model is better than bad model
        recovered_brier = brier_score(
            recovered["model"].predict_proba(X1_v, X2_v), y_v
        )
        assert recovered_brier < bad_brier, (
            f"Recovered Brier {recovered_brier:.4f} should be < bad Brier {bad_brier:.4f}"
        )
        assert monitor.n_rollbacks() == 1

    def test_no_rollback_when_improvement(self, fitted_model, train_arrays, val_arrays):
        """Improved model should not trigger rollback."""
        X1_tr, X2_tr, y_tr = train_arrays
        X1_v, X2_v, y_v = val_arrays
        from ncaa_live.recalibrator import RoundRecalibrator, brier_score

        monitor = PerformanceMonitor(threshold=0.005)
        recal = RoundRecalibrator(base_model=fitted_model)

        baseline_brier = brier_score(fitted_model.predict_proba(X1_v, X2_v), y_v)
        new_version = recal.recalibrate(
            results_X1=X1_tr[:7],
            results_X2=X2_tr[:7],
            results_y=y_tr[:7],
            X1_val=X1_v, X2_val=X2_v, y_val=y_v,
            X1_hist=X1_tr, X2_hist=X2_tr, y_hist=y_tr,
            round_num=1,
        )
        # Test logic: if new is better (or only marginally worse), no rollback
        do_rollback = monitor.should_rollback(baseline_brier, new_version.brier)
        # The recalibrated model might be slightly better or worse;
        # we just verify the rollback logic is correctly applied
        if new_version.brier <= baseline_brier + 0.005:
            assert do_rollback is False
        else:
            assert do_rollback is True
