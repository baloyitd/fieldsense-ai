"""
test_submit.py
==============
Tests for ncaa_model.submit:
  1. submission_id / parse_submission_id round-trip.
  2. build_submission produces correct format (ID, Pred columns).
  3. All Pred values are in [0, 1] (with clipping).
  4. Number of rows = C(n_teams, 2) for each gender.
  5. ID format is YYYY_LowId_HighId with LowId < HighId.
  6. validate_submission catches format errors.
  7. CSV output can be written and re-read.
  8. Men's and Women's rows are separate and correctly labelled in the ID.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from ncaa_model.submit import (
    build_submission,
    parse_submission_id,
    submission_id,
    validate_submission,
)
from ncaa_model.tests.conftest import (
    M_TEAM_IDS, W_TEAM_IDS, VAL_SEASON,
)


# ---------------------------------------------------------------------------
# ID helpers
# ---------------------------------------------------------------------------

class TestSubmissionId:

    def test_format(self):
        assert submission_id(2026, 1101, 1201) == "2026_1101_1201"

    def test_zero_padded_not_needed(self):
        assert submission_id(2025, 10, 20) == "2025_10_20"

    def test_parse_round_trip(self):
        for season, lo, hi in [(2026, 1001, 1002), (2025, 3001, 3002)]:
            sid = submission_id(season, lo, hi)
            s2, l2, h2 = parse_submission_id(sid)
            assert s2 == season and l2 == lo and h2 == hi

    def test_parse_valid_id(self):
        s, lo, hi = parse_submission_id("2026_1101_1201")
        assert s == 2026 and lo == 1101 and hi == 1201

    def test_parse_malformed_raises(self):
        with pytest.raises(ValueError):
            parse_submission_id("bad_format")

    def test_parse_too_many_parts_raises(self):
        with pytest.raises(ValueError):
            parse_submission_id("2026_1_2_3")


# ---------------------------------------------------------------------------
# build_submission
# ---------------------------------------------------------------------------

class TestBuildSubmission:

    def test_returns_dataframe(self, fitted_model, feat_df):
        df = build_submission(
            fitted_model, feat_df,
            team_ids_m=M_TEAM_IDS[:4],
            team_ids_w=[],   # skip women's
            season=VAL_SEASON,
        )
        assert isinstance(df, pd.DataFrame)

    def test_required_columns(self, fitted_model, feat_df):
        df = build_submission(
            fitted_model, feat_df,
            team_ids_m=M_TEAM_IDS[:4],
            team_ids_w=[],
            season=VAL_SEASON,
        )
        assert set(df.columns) >= {"ID", "Pred"}

    def test_pred_in_0_1(self, fitted_model, feat_df):
        df = build_submission(
            fitted_model, feat_df,
            team_ids_m=M_TEAM_IDS[:6],
            season=VAL_SEASON,
        )
        assert (df["Pred"] >= 0.0).all()
        assert (df["Pred"] <= 1.0).all()

    def test_row_count_men_only(self, fitted_model, feat_df):
        n = 6
        df = build_submission(
            fitted_model, feat_df,
            team_ids_m=M_TEAM_IDS[:n],
            team_ids_w=[],   # empty list = skip women's
            season=VAL_SEASON,
        )
        expected = n * (n - 1) // 2
        assert len(df) == expected, f"Expected {expected} rows, got {len(df)}"

    def test_row_count_both_genders(self, fitted_model, feat_df):
        n = 4
        df = build_submission(
            fitted_model, feat_df,
            team_ids_m=M_TEAM_IDS[:n],
            team_ids_w=W_TEAM_IDS[:n],
            season=VAL_SEASON,
        )
        expected = 2 * (n * (n - 1) // 2)
        assert len(df) == expected

    def test_no_duplicate_ids(self, fitted_model, feat_df):
        df = build_submission(
            fitted_model, feat_df,
            team_ids_m=M_TEAM_IDS[:6],
            season=VAL_SEASON,
        )
        assert df["ID"].duplicated().sum() == 0

    def test_id_low_lt_high(self, fitted_model, feat_df):
        df = build_submission(
            fitted_model, feat_df,
            team_ids_m=M_TEAM_IDS[:6],
            season=VAL_SEASON,
        )
        for sid in df["ID"]:
            _, lo, hi = parse_submission_id(sid)
            assert lo < hi, f"lo >= hi in ID: {sid}"

    def test_season_in_id(self, fitted_model, feat_df):
        df = build_submission(
            fitted_model, feat_df,
            team_ids_m=M_TEAM_IDS[:4],
            season=VAL_SEASON,
        )
        for sid in df["ID"]:
            s, _, _ = parse_submission_id(sid)
            assert s == VAL_SEASON

    def test_sorted_by_id(self, fitted_model, feat_df):
        df = build_submission(
            fitted_model, feat_df,
            team_ids_m=M_TEAM_IDS[:6],
            season=VAL_SEASON,
        )
        assert list(df["ID"]) == sorted(df["ID"].tolist())

    def test_clipping_applied(self, fitted_model, feat_df):
        df = build_submission(
            fitted_model, feat_df,
            team_ids_m=M_TEAM_IDS[:6],
            season=VAL_SEASON,
            clip_probs=True,
        )
        assert (df["Pred"] >= 0.025).all()
        assert (df["Pred"] <= 0.975).all()

    def test_csv_written_and_loadable(self, fitted_model, feat_df, tmp_path):
        csv_path = tmp_path / "submission.csv"
        df = build_submission(
            fitted_model, feat_df,
            team_ids_m=M_TEAM_IDS[:4],
            season=VAL_SEASON,
            output_path=csv_path,
        )
        assert csv_path.exists()
        df_loaded = pd.read_csv(csv_path)
        assert list(df_loaded.columns) == ["ID", "Pred"]
        assert len(df_loaded) == len(df)

    def test_women_ids_in_submission(self, fitted_model, feat_df):
        df = build_submission(
            fitted_model, feat_df,
            team_ids_m=[],          # explicitly skip men's
            team_ids_w=W_TEAM_IDS[:4],
            season=VAL_SEASON,
        )
        assert len(df) > 0
        # Women's TeamIDs (3001-...) should appear in IDs
        for sid in df["ID"]:
            _, lo, _ = parse_submission_id(sid)
            assert lo in W_TEAM_IDS, f"Unexpected team_id {lo} in women's submission"


# ---------------------------------------------------------------------------
# validate_submission
# ---------------------------------------------------------------------------

class TestValidateSubmission:

    def test_valid_submission_passes(self, fitted_model, feat_df):
        df = build_submission(
            fitted_model, feat_df,
            team_ids_m=M_TEAM_IDS[:4],
            season=VAL_SEASON,
        )
        report = validate_submission(df, expected_season=VAL_SEASON)
        assert report["valid"] is True
        assert report["issues"] == []

    def test_missing_pred_column_detected(self):
        df = pd.DataFrame({"ID": ["2026_1_2"]})
        report = validate_submission(df)
        assert not report["valid"]
        assert any("Pred" in iss for iss in report["issues"])

    def test_missing_id_column_detected(self):
        df = pd.DataFrame({"Pred": [0.5]})
        report = validate_submission(df)
        assert not report["valid"]
        assert any("ID" in iss for iss in report["issues"])

    def test_out_of_range_pred_detected(self):
        df = pd.DataFrame({
            "ID": ["2026_1001_1002"],
            "Pred": [1.5],
        })
        report = validate_submission(df)
        assert not report["valid"]

    def test_duplicate_id_detected(self):
        df = pd.DataFrame({
            "ID": ["2026_1001_1002", "2026_1001_1002"],
            "Pred": [0.5, 0.6],
        })
        report = validate_submission(df)
        assert not report["valid"]

    def test_wrong_season_detected(self):
        df = pd.DataFrame({
            "ID": ["2025_1001_1002"],
            "Pred": [0.5],
        })
        report = validate_submission(df, expected_season=2026)
        assert not report["valid"]

    def test_n_rows_reported(self, fitted_model, feat_df):
        df = build_submission(
            fitted_model, feat_df,
            team_ids_m=M_TEAM_IDS[:4],
            season=VAL_SEASON,
        )
        report = validate_submission(df)
        assert report["n_rows"] == len(df)
