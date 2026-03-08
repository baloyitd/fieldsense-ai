"""
test_ingest.py
==============
Tests for ncaa_data.ingest:
  1. All Kaggle file types load with correct dtypes and no null primary keys.
  2. Required columns are present for every file type.
  3. FileNotFoundError on missing required files.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest

from ncaa_data.ingest import (
    load_conferences,
    load_raw_data,
    load_results,
    load_seeds,
    load_teams,
)


# ---------------------------------------------------------------------------
# Basic loading
# ---------------------------------------------------------------------------

class TestLoadRawData:

    def test_returns_non_empty_dict(self, raw_data):
        assert isinstance(raw_data, dict)
        assert len(raw_data) > 0

    def test_required_keys_present(self, raw_data):
        for key in ("M_regular_compact", "W_regular_compact"):
            assert key in raw_data, f"Missing key: {key}"

    def test_optional_keys_present_when_files_exist(self, raw_data):
        """Detailed, tourney, seeds, conferences should all load from our fixtures."""
        for key in (
            "M_regular_detailed", "W_regular_detailed",
            "M_tourney_compact",  "W_tourney_compact",
            "M_seeds",            "W_seeds",
            "M_team_conferences", "W_team_conferences",
        ):
            assert key in raw_data, f"Optional key missing: {key}"

    def test_gender_column_values(self, raw_data):
        for key, df in raw_data.items():
            if "Gender" in df.columns:
                assert df["Gender"].isin(["M", "W"]).all(), (
                    f"{key}: unexpected Gender values"
                )

    def test_missing_required_file_raises(self, data_dir, tmp_path):
        """Copy fixture dir, remove compact results, expect FileNotFoundError."""
        import shutil as _shutil
        dest = tmp_path / "broken"
        _shutil.copytree(str(data_dir), str(dest))
        (dest / "MRegularSeasonCompactResults.csv").unlink()
        with pytest.raises(FileNotFoundError):
            load_raw_data(dest)


# ---------------------------------------------------------------------------
# Column validation (compact results)
# ---------------------------------------------------------------------------

class TestCompactResults:

    def test_required_columns(self, raw_data):
        required = [
            "Season", "DayNum", "WTeamID", "WScore",
            "LTeamID", "LScore", "WLoc", "NumOT",
        ]
        for gender in ("M", "W"):
            df = raw_data[f"{gender}_regular_compact"]
            for col in required:
                assert col in df.columns, f"{gender} compact missing: {col}"

    def test_no_null_primary_keys(self, raw_data):
        """WTeamID and LTeamID must never be null."""
        for gender in ("M", "W"):
            df = raw_data[f"{gender}_regular_compact"]
            assert df["WTeamID"].notna().all(), f"{gender} WTeamID has nulls"
            assert df["LTeamID"].notna().all(), f"{gender} LTeamID has nulls"

    def test_team_id_dtypes(self, raw_data):
        """Team IDs should be integer-castable."""
        for gender in ("M", "W"):
            df = raw_data[f"{gender}_regular_compact"]
            assert pd.api.types.is_integer_dtype(df["WTeamID"]) or \
                   df["WTeamID"].apply(lambda x: float(x).is_integer()).all()

    def test_scores_non_negative(self, raw_data):
        for gender in ("M", "W"):
            df = raw_data[f"{gender}_regular_compact"]
            assert (df["WScore"] >= 0).all()
            assert (df["LScore"] >= 0).all()

    def test_winner_score_greater_than_loser(self, raw_data):
        """In compact results (no OT edge-cases in our fixtures), W > L."""
        for gender in ("M", "W"):
            df = raw_data[f"{gender}_regular_compact"]
            assert (df["WScore"] > df["LScore"]).all(), (
                f"{gender}: some losing score >= winning score"
            )

    def test_wloc_valid_values(self, raw_data):
        for gender in ("M", "W"):
            df = raw_data[f"{gender}_regular_compact"]
            assert df["WLoc"].isin(["H", "A", "N"]).all()

    def test_season_range(self, raw_data):
        for gender in ("M", "W"):
            df = raw_data[f"{gender}_regular_compact"]
            assert df["Season"].between(2000, 2030).all()


# ---------------------------------------------------------------------------
# Column validation (detailed results)
# ---------------------------------------------------------------------------

class TestDetailedResults:

    def test_box_score_columns_present(self, raw_data):
        box_cols = [
            "WFGM", "WFGA", "WFGM3", "WFGA3", "WFTM", "WFTA",
            "WOR",  "WDR",  "WAST",  "WTO",   "WSTL", "WBLK",
            "LFGM", "LFGA", "LFGM3", "LFGA3", "LFTM", "LFTA",
            "LOR",  "LDR",  "LAST",  "LTO",   "LSTL",  "LBLK",
        ]
        for gender in ("M", "W"):
            key = f"{gender}_regular_detailed"
            if key not in raw_data:
                pytest.skip(f"{key} not present")
            df = raw_data[key]
            for col in box_cols:
                assert col in df.columns, f"{gender} detailed missing: {col}"

    def test_no_null_box_score_values(self, raw_data):
        """Our synthetic fixtures have no nulls in box-score columns."""
        num_cols = [
            "WFGM", "WFGA", "LFGM", "LFGA",
            "WOR", "WDR", "LOR", "LDR",
        ]
        for gender in ("M", "W"):
            key = f"{gender}_regular_detailed"
            if key not in raw_data:
                pytest.skip(f"{key} not present")
            df = raw_data[key]
            for col in num_cols:
                assert df[col].notna().all(), f"{gender} detailed {col} has nulls"

    def test_fgm_le_fga(self, raw_data):
        """FGM can never exceed FGA."""
        for gender in ("M", "W"):
            key = f"{gender}_regular_detailed"
            if key not in raw_data:
                pytest.skip(f"{key} not present")
            df = raw_data[key]
            assert (df["WFGM"] <= df["WFGA"]).all()
            assert (df["LFGM"] <= df["LFGA"]).all()


# ---------------------------------------------------------------------------
# Convenience loaders
# ---------------------------------------------------------------------------

class TestConvenienceLoaders:

    def test_load_teams(self, data_dir):
        df = load_teams(data_dir, "M")
        assert "TeamID" in df.columns
        assert "TeamName" in df.columns
        assert df["TeamID"].notna().all()
        assert len(df) > 0

    def test_load_seeds(self, data_dir):
        df = load_seeds(data_dir, "M")
        for col in ("Season", "Seed", "TeamID"):
            assert col in df.columns
        assert df["TeamID"].notna().all()

    def test_load_conferences(self, data_dir):
        df = load_conferences(data_dir, "M")
        assert df is not None
        assert "ConfAbbrev" in df.columns

    def test_load_conferences_missing_returns_none(self, tmp_path):
        """No conferences file → None (not an error)."""
        result = load_conferences(tmp_path, "M")
        assert result is None

    def test_load_results_compact(self, data_dir):
        df = load_results(data_dir, "M", detailed=False, tourney=False)
        assert "WTeamID" in df.columns
        assert "Gender" in df.columns

    def test_load_results_detailed(self, data_dir):
        df = load_results(data_dir, "M", detailed=True, tourney=False)
        assert "WFGM" in df.columns

    def test_load_results_women(self, data_dir):
        df = load_results(data_dir, "W", detailed=False)
        assert (df["Gender"] == "W").all()
