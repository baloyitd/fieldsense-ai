"""
test_leakage.py
===============
Tests for ncaa_submission_2026.certification.leakage_detector.

Covers:
- Clean DataFrame: no leakage reported
- Dirty DataFrame: injected 2026 rows detected
- inject_and_detect round-trip
- Multiple DataFrames scan
- Season column absence handled gracefully
- Row count in report
- Leaking seasons list correctness
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ncaa_submission_2026.certification.leakage_detector import (
    DataLeakageDetector,
    LeakageReport,
    TEST_SEASON,
    TRAIN_SEASONS,
)

from .conftest import SEASON, TEST_SEASON, TEAM_IDS


# ---------------------------------------------------------------------------
# DataLeakageDetector — clean data
# ---------------------------------------------------------------------------

class TestCleanData:
    def test_clean_train_df_no_leakage(self, train_df):
        det = DataLeakageDetector()
        report = det.check_dataframe(train_df, context="train")
        assert report.leakage_found is False

    def test_clean_report_leaking_seasons_empty(self, train_df):
        det = DataLeakageDetector()
        report = det.check_dataframe(train_df)
        assert report.leaking_seasons == []

    def test_clean_report_row_count_zero(self, train_df):
        det = DataLeakageDetector()
        report = det.check_dataframe(train_df)
        assert report.leaking_row_count == 0

    def test_returns_leakage_report(self, train_df):
        det = DataLeakageDetector()
        report = det.check_dataframe(train_df)
        assert isinstance(report, LeakageReport)

    def test_val_season_2025_is_not_leakage(self, feat_df):
        """2025 data is fine in training (only 2026+ is considered leakage)."""
        df = feat_df[feat_df["season"] == SEASON]
        det = DataLeakageDetector(test_season=TEST_SEASON)
        report = det.check_dataframe(df, context="val_2025")
        assert report.leakage_found is False

    def test_missing_season_column_handled(self):
        df = pd.DataFrame({"team_id": [1, 2], "win_pct": [0.8, 0.6]})
        det = DataLeakageDetector()
        report = det.check_dataframe(df, season_col="season")
        assert report.leakage_found is False
        assert "not present" in report.details


# ---------------------------------------------------------------------------
# DataLeakageDetector — injection and detection
# ---------------------------------------------------------------------------

class TestInjectionDetection:
    def test_injected_leakage_detected(self, train_df):
        det = DataLeakageDetector()
        dirty = det.inject_leakage(train_df, n_rows=5)
        report = det.check_dataframe(dirty, context="dirty")
        assert report.leakage_found is True

    def test_injected_row_count_matches(self, train_df):
        det = DataLeakageDetector()
        dirty = det.inject_leakage(train_df, n_rows=7)
        report = det.check_dataframe(dirty)
        assert report.leaking_row_count == 7

    def test_injected_season_in_report(self, train_df):
        det = DataLeakageDetector()
        dirty = det.inject_leakage(train_df, season=2026, n_rows=3)
        report = det.check_dataframe(dirty)
        assert 2026 in report.leaking_seasons

    def test_inject_and_detect_returns_tuple(self, train_df):
        det = DataLeakageDetector()
        result = det.inject_and_detect(train_df)
        assert isinstance(result, tuple) and len(result) == 2

    def test_inject_and_detect_dirty_df_is_larger(self, train_df):
        det = DataLeakageDetector()
        dirty, report = det.inject_and_detect(train_df, n_rows=4)
        assert len(dirty) == len(train_df) + 4

    def test_inject_and_detect_report_flags_leakage(self, train_df):
        det = DataLeakageDetector()
        _, report = det.inject_and_detect(train_df)
        assert report.leakage_found is True

    def test_injected_original_df_unchanged(self, train_df):
        """inject_leakage must not mutate the original DataFrame."""
        original_len = len(train_df)
        original_max_season = train_df["season"].max()
        det = DataLeakageDetector()
        _ = det.inject_leakage(train_df, n_rows=10)
        assert len(train_df) == original_len
        assert train_df["season"].max() == original_max_season

    def test_inject_custom_season_detected(self, train_df):
        det = DataLeakageDetector(test_season=2025)
        dirty = det.inject_leakage(train_df, season=2025, n_rows=2)
        report = det.check_dataframe(dirty)
        assert report.leakage_found is True
        assert 2025 in report.leaking_seasons


# ---------------------------------------------------------------------------
# DataLeakageDetector — check_many
# ---------------------------------------------------------------------------

class TestCheckMany:
    def test_check_many_clean_all_pass(self, train_df, feat_df):
        det = DataLeakageDetector()
        clean_2024 = feat_df[feat_df["season"] <= 2024]
        reports = det.check_many({"a": train_df, "b": clean_2024})
        assert all(not r.leakage_found for r in reports)
        assert not det.any_leakage(reports)

    def test_check_many_one_dirty_detected(self, train_df):
        det = DataLeakageDetector()
        dirty = det.inject_leakage(train_df, n_rows=3)
        reports = det.check_many({"clean": train_df, "dirty": dirty})
        assert det.any_leakage(reports) is True

    def test_check_many_returns_correct_contexts(self, train_df):
        det = DataLeakageDetector()
        reports = det.check_many({"alpha": train_df, "beta": train_df})
        contexts = {r.context for r in reports}
        assert contexts == {"alpha", "beta"}

    def test_any_leakage_false_when_all_clean(self, train_df):
        det = DataLeakageDetector()
        reports = det.check_many({"a": train_df})
        assert det.any_leakage(reports) is False


# ---------------------------------------------------------------------------
# LeakageReport — to_dict
# ---------------------------------------------------------------------------

class TestLeakageReportToDict:
    def test_to_dict_has_required_keys(self, train_df):
        det = DataLeakageDetector()
        report = det.check_dataframe(train_df)
        d = report.to_dict()
        for key in ("context", "leakage_found", "leaking_seasons",
                    "leaking_row_count", "details"):
            assert key in d

    def test_to_dict_leakage_found_is_bool(self, train_df):
        det = DataLeakageDetector()
        report = det.check_dataframe(train_df)
        assert isinstance(report.to_dict()["leakage_found"], bool)
