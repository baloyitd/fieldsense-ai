"""
test_submit.py
==============
Tests for ncaa_models.submit.

Covers
------
1. make_submission_id / parse_submission_id round-trip.
2. UNIT TEST 3: submission row count = C(N, 2) with correct ID formatting.
3. Lower TeamId always first in ID.
4. No duplicate IDs.
5. Sorted by ID.
6. Probabilities in [0.01, 0.99].
7. Clipping applied.
8. CSV output written and reloadable.
9. validate_submission catches format errors.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from ncaa_models.submit import (
    build_submission,
    make_submission_id,
    parse_submission_id,
    validate_submission,
)
from ncaa_models.tests.conftest import (
    M_TEAM_IDS,
    VAL_SEASON,
    W_TEAM_IDS,
    make_feat_df,
)


# ---------------------------------------------------------------------------
# ID helpers
# ---------------------------------------------------------------------------

class TestMakeSubmissionId:

    def test_format(self):
        assert make_submission_id(2026, 1101, 1201) == "2026_1101_1201"

    def test_format_with_small_ids(self):
        assert make_submission_id(2025, 10, 20) == "2025_10_20"

    def test_round_trip(self):
        for season, lo, hi in [(2026, 1001, 1002), (2025, 3001, 3999)]:
            sid = make_submission_id(season, lo, hi)
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

    def test_parse_non_integer_raises(self):
        with pytest.raises(ValueError):
            parse_submission_id("2026_abc_1002")


# ---------------------------------------------------------------------------
# UNIT TEST 3: build_submission row count = C(N, 2)
# ---------------------------------------------------------------------------

class TestBuildSubmission:

    @pytest.fixture(scope="class")
    def feat(self):
        """Feature DataFrame for the 8 M_TEAM_IDS at VAL_SEASON."""
        return make_feat_df(M_TEAM_IDS, [VAL_SEASON])

    def test_row_count_c_n_2(self, fitted_model, feat):
        """UNIT TEST 3 — C(N,2) rows for N teams."""
        n = 6
        team_ids = M_TEAM_IDS[:n]
        df = build_submission(fitted_model, feat, team_ids, VAL_SEASON)
        expected = n * (n - 1) // 2
        assert len(df) == expected, (
            f"Expected C({n},2)={expected} rows, got {len(df)}"
        )

    def test_row_count_all_8_teams(self, fitted_model, feat):
        n = 8
        df = build_submission(fitted_model, feat, M_TEAM_IDS[:n], VAL_SEASON)
        assert len(df) == math.comb(n, 2)

    def test_returns_dataframe(self, fitted_model, feat):
        df = build_submission(fitted_model, feat, M_TEAM_IDS[:4], VAL_SEASON)
        assert isinstance(df, pd.DataFrame)

    def test_has_id_and_pred_columns(self, fitted_model, feat):
        df = build_submission(fitted_model, feat, M_TEAM_IDS[:4], VAL_SEASON)
        assert set(df.columns) == {"ID", "Pred"}

    def test_lower_id_first_in_id(self, fitted_model, feat):
        df = build_submission(fitted_model, feat, M_TEAM_IDS[:6], VAL_SEASON)
        for sid in df["ID"]:
            _, lo, hi = parse_submission_id(sid)
            assert lo < hi, f"lo >= hi in ID '{sid}'"

    def test_no_duplicate_ids(self, fitted_model, feat):
        df = build_submission(fitted_model, feat, M_TEAM_IDS[:6], VAL_SEASON)
        assert df["ID"].duplicated().sum() == 0

    def test_sorted_by_id(self, fitted_model, feat):
        df = build_submission(fitted_model, feat, M_TEAM_IDS[:6], VAL_SEASON)
        assert list(df["ID"]) == sorted(df["ID"].tolist())

    def test_season_in_all_ids(self, fitted_model, feat):
        df = build_submission(fitted_model, feat, M_TEAM_IDS[:4], VAL_SEASON)
        for sid in df["ID"]:
            s, _, _ = parse_submission_id(sid)
            assert s == VAL_SEASON

    def test_probs_in_0_1(self, fitted_model, feat):
        df = build_submission(fitted_model, feat, M_TEAM_IDS[:6], VAL_SEASON)
        assert (df["Pred"] >= 0.0).all()
        assert (df["Pred"] <= 1.0).all()

    def test_clipping_applied(self, fitted_model, feat):
        """With clip_probs=True (default), Pred ∈ [0.01, 0.99]."""
        df = build_submission(
            fitted_model, feat, M_TEAM_IDS[:6], VAL_SEASON, clip_probs=True
        )
        assert (df["Pred"] >= 0.01).all()
        assert (df["Pred"] <= 0.99).all()

    def test_csv_written_and_loadable(self, fitted_model, feat, tmp_path):
        csv_path = tmp_path / "submission.csv"
        df = build_submission(
            fitted_model, feat, M_TEAM_IDS[:4], VAL_SEASON, output_path=csv_path
        )
        assert csv_path.exists()
        loaded = pd.read_csv(csv_path)
        assert list(loaded.columns) == ["ID", "Pred"]
        assert len(loaded) == len(df)

    def test_empty_team_list_returns_empty_df(self, fitted_model, feat):
        df = build_submission(fitted_model, feat, [], VAL_SEASON)
        assert len(df) == 0

    def test_single_team_returns_empty_df(self, fitted_model, feat):
        """C(1,2) = 0 pairs."""
        df = build_submission(fitted_model, feat, [M_TEAM_IDS[0]], VAL_SEASON)
        assert len(df) == 0

    def test_id_format_is_correct(self, fitted_model, feat):
        """Each ID should match YYYY_LO_HI with LO < HI."""
        df = build_submission(fitted_model, feat, M_TEAM_IDS[:4], VAL_SEASON)
        for sid in df["ID"]:
            parts = sid.split("_")
            assert len(parts) == 3, f"Unexpected ID format: '{sid}'"
            year, lo, hi = int(parts[0]), int(parts[1]), int(parts[2])
            assert year == VAL_SEASON
            assert lo < hi


# ---------------------------------------------------------------------------
# validate_submission
# ---------------------------------------------------------------------------

class TestValidateSubmission:

    def _valid_df(self):
        return pd.DataFrame({
            "ID": ["2026_1001_1002", "2026_1001_1003", "2026_1002_1003"],
            "Pred": [0.6, 0.7, 0.4],
        })

    def test_valid_submission_passes(self):
        report = validate_submission(self._valid_df(), expected_season=2026)
        assert report["valid"] is True
        assert report["issues"] == []

    def test_n_rows_reported(self):
        df = self._valid_df()
        report = validate_submission(df)
        assert report["n_rows"] == len(df)

    def test_missing_pred_detected(self):
        df = pd.DataFrame({"ID": ["2026_1_2"]})
        report = validate_submission(df)
        assert not report["valid"]
        assert any("Pred" in iss for iss in report["issues"])

    def test_missing_id_detected(self):
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
        assert any("season" in iss.lower() for iss in report["issues"])

    def test_lo_ge_hi_detected(self):
        df = pd.DataFrame({
            "ID": ["2026_1002_1001"],  # lo > hi
            "Pred": [0.5],
        })
        report = validate_submission(df)
        assert not report["valid"]

    def test_expected_n_rows_mismatch_detected(self):
        df = self._valid_df()
        report = validate_submission(df, expected_n_rows=100)
        assert not report["valid"]
        assert any("Row count" in iss for iss in report["issues"])
