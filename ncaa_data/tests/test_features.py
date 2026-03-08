"""
test_features.py
================
Tests for ncaa_data.features:
  1. compute_features returns a well-formed feature matrix.
  2. One row per (gender, season, team) — no duplicates.
  3. At least 25 feature columns computed.
  4. All features fall within domain-valid ranges.
  5. Derived formula invariants (net_rtg = ortg - drtg, etc.).
  6. Season filtering works correctly.
  7. Compact-only fallback fills NaN for box-score-derived features.

Domain-valid ranges
-------------------
  ORTG / DRTG    : [60, 140]  points per 100 possessions
  Tempo          : [55, 85]   possessions per game (proxy for pace per 40 min)
  FG%            : [0.25, 0.70]
  3P%            : [0.15, 0.55]
  FT%            : [0.40, 0.95]
  ORB rate       : [0.10, 0.55]
  DRB rate       : [0.40, 0.90]  (typically 65-75% in D1; wider range used)
  Win %          : [0.00, 1.00]
  SOS            : [0.20, 0.80]
  Net rating     : [-60, 60]
  Scoring margin : [-40, 40]
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ncaa_data.features import compute_features, FEATURE_COLS, KEY_COLS
from ncaa_data.tests.conftest import SEASONS, TEAM_IDS, _compact


# Domain-valid ranges expected for real (and well-formed synthetic) data
_RANGES = {
    "ortg":           (60,   140),
    "drtg":           (60,   140),
    "tempo":          (55,    85),
    "fg_pct":         (0.25,  0.70),
    "fg3_pct":        (0.15,  0.55),
    "ft_pct":         (0.40,  0.95),
    "orb_rate":       (0.10,  0.55),
    "drb_rate":       (0.40,  0.90),
    "win_pct":        (0.00,  1.00),
    "sos":            (0.20,  0.80),
    "net_rtg":        (-60,    60),
    "scoring_margin": (-40,    40),
}


# ---------------------------------------------------------------------------
# Shape and structure
# ---------------------------------------------------------------------------

class TestShape:

    def test_returns_dataframe(self, feat_df):
        assert isinstance(feat_df, pd.DataFrame)

    def test_no_duplicate_team_seasons(self, feat_df):
        dupes = feat_df.duplicated(["gender", "season", "canonical_id"]).sum()
        assert dupes == 0, f"{dupes} duplicate (gender, season, canonical_id) rows"

    def test_at_least_25_feature_columns(self, feat_df):
        computed = [c for c in FEATURE_COLS if c in feat_df.columns]
        assert len(computed) >= 25, (
            f"Only {len(computed)} feature cols present; need ≥ 25.\n"
            f"Present: {computed}"
        )

    def test_key_columns_present(self, feat_df):
        for col in KEY_COLS:
            assert col in feat_df.columns, f"Missing key column: {col}"

    def test_both_genders_in_output(self, feat_df):
        assert set(feat_df["gender"].unique()) == {"M", "W"}

    def test_all_seasons_in_output(self, feat_df):
        assert set(SEASONS).issubset(set(feat_df["season"].unique()))

    def test_all_teams_have_rows(self, feat_df):
        """Every synthetic team should appear in every season."""
        for gender in ("M", "W"):
            for season in SEASONS:
                subset = feat_df[
                    (feat_df["gender"] == gender) & (feat_df["season"] == season)
                ]
                assert len(subset) >= len(TEAM_IDS), (
                    f"{gender} season {season}: expected ≥ {len(TEAM_IDS)} teams, "
                    f"got {len(subset)}"
                )

    def test_games_played_positive(self, feat_df):
        assert (feat_df["games_played"] > 0).all()

    def test_no_null_key_columns(self, feat_df):
        for col in ["gender", "season", "canonical_id", "team_id"]:
            assert feat_df[col].notna().all(), f"Null in key column: {col}"


# ---------------------------------------------------------------------------
# Domain-valid range assertions
# ---------------------------------------------------------------------------

class TestFeatureRanges:

    @pytest.mark.parametrize("col,lo,hi", [
        (col, lo, hi) for col, (lo, hi) in _RANGES.items()
    ])
    def test_range(self, feat_df, col, lo, hi):
        if col not in feat_df.columns:
            pytest.skip(f"Column {col!r} not present in feature matrix")
        valid = feat_df[col].dropna()
        if valid.empty:
            pytest.skip(f"Column {col!r} is all NaN (compact-only fallback)")
        violations = valid[(valid < lo) | (valid > hi)]
        assert violations.empty, (
            f"{col}: {len(violations)} values outside [{lo}, {hi}]\n"
            f"  min={valid.min():.4f}  max={valid.max():.4f}\n"
            f"  out-of-range sample: {violations.head(5).tolist()}"
        )

    def test_win_pct_bounded(self, feat_df):
        wp = feat_df["win_pct"].dropna()
        assert (wp >= 0).all() and (wp <= 1).all()

    def test_fg_pct_le_1(self, feat_df):
        pct = feat_df["fg_pct"].dropna()
        assert (pct <= 1.0).all()

    def test_seed_valid_range(self, feat_df):
        """Seed must be 1-16 (seeded) or 17 (unseeded)."""
        assert feat_df["seed"].between(1, 17).all(), (
            f"Invalid seeds: {feat_df['seed'].unique()}"
        )


# ---------------------------------------------------------------------------
# Formula invariants
# ---------------------------------------------------------------------------

class TestFormulaInvariants:

    def test_scoring_margin_equals_ppg_diff(self, feat_df):
        diff = (
            feat_df["ppg_scored"] - feat_df["ppg_allowed"] - feat_df["scoring_margin"]
        ).abs()
        assert (diff < 1e-6).all(), "scoring_margin ≠ ppg_scored - ppg_allowed"

    def test_net_rtg_equals_ortg_minus_drtg(self, feat_df):
        valid = feat_df.dropna(subset=["ortg", "drtg", "net_rtg"])
        if valid.empty:
            pytest.skip("No rows with all three ratings")
        diff = (valid["ortg"] - valid["drtg"] - valid["net_rtg"]).abs()
        assert (diff < 1e-6).all(), "net_rtg ≠ ortg - drtg"

    def test_efg_pct_ge_fg_pct(self, feat_df):
        """eFG% ≥ FG% (3-pointers count 1.5x)."""
        valid = feat_df.dropna(subset=["efg_pct", "fg_pct"])
        assert (valid["efg_pct"] >= valid["fg_pct"] - 1e-9).all(), (
            "eFG% should be ≥ FG%"
        )

    def test_high_win_pct_implies_positive_margin(self, feat_df):
        """Teams with win% > 0.70 should have a positive scoring margin."""
        good = feat_df[feat_df["win_pct"] > 0.70]
        if good.empty:
            pytest.skip("No teams with win% > 0.70")
        assert (good["scoring_margin"] > 0).all(), (
            "High-win-pct teams should have positive scoring margin"
        )

    def test_ot_rate_bounded(self, feat_df):
        assert (feat_df["ot_rate"] >= 0).all()
        assert (feat_df["ot_rate"] <= 1).all()


# ---------------------------------------------------------------------------
# Efficiency directionality
# ---------------------------------------------------------------------------

class TestEfficiencyDirectionality:

    def test_best_team_has_positive_net_rtg(self, feat_df):
        """Team with highest win% should have positive net rating."""
        valid = feat_df.dropna(subset=["net_rtg"])
        if valid.empty:
            pytest.skip("No rows with net_rtg")
        best = valid.sort_values("win_pct", ascending=False).iloc[0]
        assert best["net_rtg"] > 0, (
            f"Best team (win_pct={best['win_pct']:.3f}) has net_rtg={best['net_rtg']:.2f}"
        )

    def test_worst_team_has_negative_net_rtg(self, feat_df):
        valid = feat_df.dropna(subset=["net_rtg"])
        if valid.empty:
            pytest.skip("No rows with net_rtg")
        worst = valid.sort_values("win_pct").iloc[0]
        assert worst["net_rtg"] < 0, (
            f"Worst team (win_pct={worst['win_pct']:.3f}) has net_rtg={worst['net_rtg']:.2f}"
        )

    def test_top_teams_ranked_by_net_rtg_match_win_pct(self, feat_df):
        """
        Pearson correlation between win_pct and net_rtg should be strongly
        positive (≥ 0.6) — efficiency margin should be predictive of wins.
        """
        valid = feat_df.dropna(subset=["win_pct", "net_rtg"])
        if len(valid) < 10:
            pytest.skip("Insufficient data for correlation check")
        corr = valid[["win_pct", "net_rtg"]].corr().iloc[0, 1]
        assert corr >= 0.6, (
            f"win_pct ↔ net_rtg correlation too low: {corr:.3f} (expected ≥ 0.6)"
        )


# ---------------------------------------------------------------------------
# Season filtering
# ---------------------------------------------------------------------------

class TestSeasonFilter:

    def test_single_season_filter(self, games_df):
        feat = compute_features(games_df, seasons=[2023])
        assert (feat["season"] == 2023).all()
        assert 2022 not in feat["season"].values
        assert 2024 not in feat["season"].values

    def test_multi_season_filter(self, games_df):
        feat = compute_features(games_df, seasons=[2021, 2025])
        assert set(feat["season"].unique()) == {2021, 2025}

    def test_invalid_season_raises(self, games_df):
        with pytest.raises(ValueError):
            compute_features(games_df, seasons=[1899])


# ---------------------------------------------------------------------------
# Compact-only fallback
# ---------------------------------------------------------------------------

class TestCompactFallback:

    def test_nan_for_box_score_features_without_detailed(self, raw_data):
        """When only compact results are available, box-score features are NaN."""
        from ncaa_data.normalize import normalize_all
        # Build raw_data dict with only compact results
        compact_only = {
            k: v for k, v in raw_data.items()
            if "detailed" not in k
        }
        games = normalize_all(compact_only)
        feat = compute_features(games, seasons=SEASONS)
        box_cols = ["fg_pct", "fg3_pct", "ft_pct", "ortg", "drtg", "net_rtg"]
        for col in box_cols:
            if col in feat.columns:
                assert feat[col].isna().all(), (
                    f"{col} should be all-NaN without detailed stats"
                )

    def test_basic_features_not_nan_without_detailed(self, raw_data):
        """win_pct, ppg_scored, etc. should still be computed from compact data."""
        from ncaa_data.normalize import normalize_all
        compact_only = {
            k: v for k, v in raw_data.items()
            if "detailed" not in k
        }
        games = normalize_all(compact_only)
        feat = compute_features(games, seasons=SEASONS)
        for col in ["win_pct", "ppg_scored", "ppg_allowed", "scoring_margin"]:
            assert feat[col].notna().all(), (
                f"{col} should not be NaN even without detailed stats"
            )


# ---------------------------------------------------------------------------
# Seed and conference
# ---------------------------------------------------------------------------

class TestSeedAndConference:

    def test_seed_column_present(self, feat_df):
        assert "seed" in feat_df.columns

    def test_seeded_teams_seed_1_to_16(self, feat_df):
        seeded = feat_df[feat_df["seed"] < 17]
        assert seeded["seed"].between(1, 16).all()

    def test_team_1001_seed_1_all_seasons(self, feat_df):
        t1 = feat_df[(feat_df["team_id"] == 1001) & (feat_df["gender"] == "M")]
        assert (t1["seed"] == 1).all(), (
            f"Team 1001 seeds: {t1['seed'].unique()}"
        )

    def test_conf_id_non_negative_integer(self, feat_df):
        assert "conf_id" in feat_df.columns
        assert (feat_df["conf_id"] >= 0).all()
        assert feat_df["conf_id"].dtype in (int, "int64", "int32")

    def test_acc_and_b10_get_different_conf_ids(self, feat_df):
        acc_teams = feat_df[feat_df["team_id"].isin([1001, 1002])]
        b10_teams = feat_df[feat_df["team_id"].isin([1003, 1004])]
        if acc_teams.empty or b10_teams.empty:
            pytest.skip("Cannot find ACC/B10 teams")
        acc_id = acc_teams["conf_id"].iloc[0]
        b10_id = b10_teams["conf_id"].iloc[0]
        assert acc_id != b10_id, "ACC and B10 should have different conf_id"
