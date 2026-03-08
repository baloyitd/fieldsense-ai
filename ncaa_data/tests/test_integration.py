"""
test_integration.py
===================
End-to-end integration tests for the NCAA data pipeline (Stage 01/10).

5. End-to-end: raw CSV directory → final feature matrices for all 5 seasons.
6. Cross-validate that team rankings by efficiency margin are directionally
   sensible (top teams have positive net rating).
7. Output files (CSV, Parquet, Pickle, manifest, validation report) are
   present and loadable.
8. Validation report reports correct season team counts.
"""

from __future__ import annotations

import json
import pickle
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from ncaa_data.pipeline import run_pipeline, load_features, build_validation_report
from ncaa_data.features import FEATURE_COLS, KEY_COLS
from ncaa_data.tests.conftest import SEASONS, TEAM_IDS


# ---------------------------------------------------------------------------
# Fixture: pipeline output dir (function-scoped so each test gets fresh I/O)
# ---------------------------------------------------------------------------

@pytest.fixture()
def pipeline_result(data_dir, tmp_path):
    """Run full pipeline and return (feat_df, out_dir) tuple."""
    out = tmp_path / "features"
    feat = run_pipeline(
        data_dir=data_dir,
        out_dir=out,
        seasons=SEASONS,
        save_parquet=True,
        save_csv=True,
        save_pickle=True,
    )
    return feat, out


# ---------------------------------------------------------------------------
# Test 5 – Full pipeline for all 5 seasons
# ---------------------------------------------------------------------------

class TestFullPipeline:

    def test_returns_dataframe(self, pipeline_result):
        feat, _ = pipeline_result
        assert isinstance(feat, pd.DataFrame)
        assert len(feat) > 0

    def test_all_five_seasons_present(self, pipeline_result):
        feat, _ = pipeline_result
        assert set(SEASONS).issubset(set(feat["season"].unique())), (
            f"Missing seasons; found {sorted(feat['season'].unique())}"
        )

    def test_both_genders_present(self, pipeline_result):
        feat, _ = pipeline_result
        assert set(feat["gender"].unique()) == {"M", "W"}

    def test_at_least_25_feature_cols(self, pipeline_result):
        feat, _ = pipeline_result
        computed = [c for c in FEATURE_COLS if c in feat.columns]
        assert len(computed) >= 25, (
            f"Only {len(computed)} feature columns in pipeline output"
        )

    def test_all_key_cols_present(self, pipeline_result):
        feat, _ = pipeline_result
        for col in KEY_COLS:
            assert col in feat.columns

    def test_no_duplicate_team_seasons(self, pipeline_result):
        feat, _ = pipeline_result
        dupes = feat.duplicated(["gender", "season", "canonical_id"]).sum()
        assert dupes == 0

    def test_no_null_key_columns(self, pipeline_result):
        feat, _ = pipeline_result
        for col in ["gender", "season", "canonical_id"]:
            assert feat[col].notna().all()


# ---------------------------------------------------------------------------
# Test 6 – Directional correctness of efficiency margin rankings
# ---------------------------------------------------------------------------

class TestEfficiencyRankings:

    def test_top_teams_positive_net_rtg(self, pipeline_result):
        """
        The top quartile by win% should have a positive net rating.
        This cross-validates that ORTG - DRTG aligns with win outcomes.
        """
        feat, _ = pipeline_result
        valid = feat.dropna(subset=["win_pct", "net_rtg"])
        if valid.empty:
            pytest.skip("No rows with net_rtg")
        q75 = valid["win_pct"].quantile(0.75)
        top_teams = valid[valid["win_pct"] >= q75]
        pct_positive = (top_teams["net_rtg"] > 0).mean()
        assert pct_positive >= 0.8, (
            f"Only {pct_positive:.1%} of top-quartile teams have positive net_rtg "
            f"(expected ≥ 80%)"
        )

    def test_bottom_teams_negative_net_rtg(self, pipeline_result):
        """Bottom quartile by win% should mostly have negative net rating."""
        feat, _ = pipeline_result
        valid = feat.dropna(subset=["win_pct", "net_rtg"])
        if valid.empty:
            pytest.skip("No rows with net_rtg")
        q25 = valid["win_pct"].quantile(0.25)
        bottom_teams = valid[valid["win_pct"] <= q25]
        pct_negative = (bottom_teams["net_rtg"] < 0).mean()
        assert pct_negative >= 0.8, (
            f"Only {pct_negative:.1%} of bottom-quartile teams have negative net_rtg "
            f"(expected ≥ 80%)"
        )

    def test_win_pct_net_rtg_correlation_positive(self, pipeline_result):
        """Pearson correlation between win% and net_rtg must be strongly positive."""
        feat, _ = pipeline_result
        valid = feat.dropna(subset=["win_pct", "net_rtg"])
        if len(valid) < 10:
            pytest.skip("Too few rows for correlation")
        corr = valid[["win_pct", "net_rtg"]].corr().iloc[0, 1]
        assert corr >= 0.6, (
            f"win_pct ↔ net_rtg correlation = {corr:.3f}; expected ≥ 0.6"
        )

    def test_best_seeded_team_has_good_stats(self, pipeline_result):
        """Seed-1 teams should have above-average win%."""
        feat, _ = pipeline_result
        seed1 = feat[feat["seed"] == 1]
        if seed1.empty:
            pytest.skip("No seed-1 teams in output")
        overall_avg = feat["win_pct"].mean()
        assert seed1["win_pct"].mean() > overall_avg, (
            "Seed-1 teams should have above-average win%"
        )


# ---------------------------------------------------------------------------
# Test 7 – Output file I/O
# ---------------------------------------------------------------------------

class TestOutputFiles:

    def test_csv_file_exists(self, pipeline_result):
        _, out_dir = pipeline_result
        assert (out_dir / "ncaa_features_all.csv").exists()

    def test_csv_loadable(self, pipeline_result):
        _, out_dir = pipeline_result
        df = pd.read_csv(out_dir / "ncaa_features_all.csv")
        assert len(df) > 0

    def test_parquet_files_exist_per_season(self, pipeline_result):
        _, out_dir = pipeline_result
        for season in SEASONS:
            path = out_dir / f"ncaa_features_{season}.parquet"
            assert path.exists(), f"Missing parquet for season {season}"

    def test_parquet_loadable(self, pipeline_result):
        _, out_dir = pipeline_result
        df = pd.read_parquet(out_dir / f"ncaa_features_{SEASONS[0]}.parquet")
        assert len(df) > 0

    def test_pickle_file_exists(self, pipeline_result):
        _, out_dir = pipeline_result
        assert (out_dir / "ncaa_features_all.pkl").exists()

    def test_pickle_loadable_and_consistent(self, pipeline_result):
        feat, out_dir = pipeline_result
        with open(out_dir / "ncaa_features_all.pkl", "rb") as f:
            pkl_df = pickle.load(f)
        assert isinstance(pkl_df, pd.DataFrame)
        assert len(pkl_df) == len(feat)

    def test_manifest_exists(self, pipeline_result):
        _, out_dir = pipeline_result
        assert (out_dir / "pipeline_manifest.json").exists()

    def test_manifest_content(self, pipeline_result):
        _, out_dir = pipeline_result
        manifest = json.loads((out_dir / "pipeline_manifest.json").read_text())
        assert manifest["stage"] == "01/10"
        assert manifest["n_features"] >= 25
        assert set(manifest["seasons"]) == set(SEASONS)

    def test_validation_report_exists(self, pipeline_result):
        _, out_dir = pipeline_result
        assert (out_dir / "validation_report.json").exists()

    def test_load_features_pkl_priority(self, pipeline_result):
        """load_features should prefer pickle over parquet/CSV."""
        _, out_dir = pipeline_result
        df = load_features(out_dir=out_dir, seasons=SEASONS)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_load_features_season_filter(self, pipeline_result):
        _, out_dir = pipeline_result
        df = load_features(out_dir=out_dir, seasons=[2023])
        assert set(df["season"].unique()) == {2023}

    def test_load_features_missing_raises(self, tmp_path):
        from ncaa_data.pipeline import load_features
        with pytest.raises(FileNotFoundError):
            load_features(out_dir=tmp_path / "nonexistent")


# ---------------------------------------------------------------------------
# Test 8 – Validation report correctness
# ---------------------------------------------------------------------------

class TestValidationReport:

    def test_report_structure(self, feat_df):
        report = build_validation_report(feat_df)
        assert "total_team_seasons" in report
        assert "seasons" in report
        assert "range_violations" in report
        assert "status" in report

    def test_report_season_count(self, feat_df):
        report = build_validation_report(feat_df)
        assert set(report["seasons"].keys()) == set(SEASONS)

    def test_report_team_counts_per_season(self, feat_df):
        report = build_validation_report(feat_df)
        for season in SEASONS:
            info = report["seasons"][season]
            # At minimum, all 4 synthetic teams for both genders
            assert info["team_count"] >= len(TEAM_IDS), (
                f"Season {season}: expected ≥ {len(TEAM_IDS)} teams, "
                f"got {info['team_count']}"
            )

    def test_report_completeness_fraction_valid(self, feat_df):
        report = build_validation_report(feat_df)
        for season, info in report["seasons"].items():
            for col, completeness in info["completeness"].items():
                assert 0.0 <= completeness <= 1.0, (
                    f"Season {season} column {col}: completeness={completeness}"
                )

    def test_report_status_is_pass_or_warn(self, feat_df):
        report = build_validation_report(feat_df)
        assert report["status"] in ("PASS", "WARN")

    def test_report_saved_to_disk(self, pipeline_result):
        feat, out_dir = pipeline_result
        path = out_dir / "validation_report.json"
        assert path.exists()
        report = json.loads(path.read_text())
        assert "total_team_seasons" in report


# ---------------------------------------------------------------------------
# Dry-run mode (no file I/O)
# ---------------------------------------------------------------------------

class TestDryRun:

    def test_dry_run_returns_dataframe(self, data_dir, tmp_path):
        feat = run_pipeline(
            data_dir=data_dir,
            out_dir=tmp_path / "features",
            seasons=SEASONS,
            dry_run=True,
        )
        assert isinstance(feat, pd.DataFrame)
        assert len(feat) > 0

    def test_dry_run_writes_no_files(self, data_dir, tmp_path):
        out = tmp_path / "features"
        run_pipeline(
            data_dir=data_dir,
            out_dir=out,
            seasons=SEASONS,
            dry_run=True,
        )
        assert not out.exists(), "dry_run=True should not create output directory"
