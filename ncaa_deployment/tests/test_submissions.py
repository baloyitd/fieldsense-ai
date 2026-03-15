"""
test_submissions.py
===================
Tests for ncaa_deployment.submission.

Covers:
- Both submissions pass Kaggle format validation
- Distinct prediction distributions: KL divergence > 0.001
- Conservative predictions are closer to 0.5 than aggressive
- SubmissionVariant structure and save/load
- kl_divergence function
- No duplicate IDs in either submission
- Predictions clipped to [0.01, 0.99]
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ncaa_deployment.submission import (
    SubmissionBuilder,
    SubmissionVariant,
    kl_divergence,
)

from .conftest import TEAM_IDS, SEASON


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_builder(feat_df) -> SubmissionBuilder:
    from ncaa_models.baseline import FEATURE_COLS
    return SubmissionBuilder(
        feat_df=feat_df,
        team_ids=TEAM_IDS,
        season=SEASON,
        feature_cols=FEATURE_COLS,
    )


# ---------------------------------------------------------------------------
# SubmissionBuilder — conservative
# ---------------------------------------------------------------------------

class TestConservativeSubmission:
    def test_passes_kaggle_validation(self, strong_model, feat_df):
        builder = _make_builder(feat_df)
        variant = builder.build_conservative(strong_model, temperature=5.0)
        result = variant.validate(expected_season=SEASON)
        assert result["valid"] is True, f"Validation failed: {result}"

    def test_no_duplicate_ids(self, strong_model, feat_df):
        builder = _make_builder(feat_df)
        variant = builder.build_conservative(strong_model, temperature=5.0)
        assert variant.df["ID"].nunique() == len(variant.df)

    def test_predictions_in_valid_range(self, strong_model, feat_df):
        builder = _make_builder(feat_df)
        variant = builder.build_conservative(strong_model, temperature=5.0)
        assert (variant.df["Pred"] >= 0.01).all()
        assert (variant.df["Pred"] <= 0.99).all()

    def test_name_is_conservative(self, strong_model, feat_df):
        builder = _make_builder(feat_df)
        variant = builder.build_conservative(strong_model, temperature=5.0)
        assert variant.name == "conservative"

    def test_temperature_stored(self, strong_model, feat_df):
        builder = _make_builder(feat_df)
        variant = builder.build_conservative(strong_model, temperature=5.0)
        assert variant.temperature == 5.0

    def test_conservative_closer_to_half(self, strong_model, feat_df):
        """Conservative predictions should have lower std than raw predictions."""
        builder = _make_builder(feat_df)
        cons = builder.build_conservative(strong_model, temperature=5.0)
        aggr = builder.build_aggressive(strong_model, temperature=1.0)
        std_cons = float(cons.df["Pred"].std())
        std_aggr = float(aggr.df["Pred"].std())
        assert std_cons < std_aggr + 0.01


# ---------------------------------------------------------------------------
# SubmissionBuilder — aggressive
# ---------------------------------------------------------------------------

class TestAggressiveSubmission:
    def test_passes_kaggle_validation(self, strong_model, feat_df):
        builder = _make_builder(feat_df)
        variant = builder.build_aggressive(strong_model)
        result = variant.validate(expected_season=SEASON)
        assert result["valid"] is True, f"Validation failed: {result}"

    def test_no_duplicate_ids(self, strong_model, feat_df):
        builder = _make_builder(feat_df)
        variant = builder.build_aggressive(strong_model)
        assert variant.df["ID"].nunique() == len(variant.df)

    def test_predictions_in_valid_range(self, strong_model, feat_df):
        builder = _make_builder(feat_df)
        variant = builder.build_aggressive(strong_model)
        assert (variant.df["Pred"] >= 0.01).all()
        assert (variant.df["Pred"] <= 0.99).all()

    def test_name_is_aggressive(self, strong_model, feat_df):
        builder = _make_builder(feat_df)
        variant = builder.build_aggressive(strong_model)
        assert variant.name == "aggressive"

    def test_len_matches_n_matchups(self, strong_model, feat_df):
        builder = _make_builder(feat_df)
        variant = builder.build_aggressive(strong_model)
        # C(8, 2) = 28 matchups
        import math
        expected = math.comb(len(TEAM_IDS), 2)
        assert len(variant.df) == expected


# ---------------------------------------------------------------------------
# Distinct distributions
# ---------------------------------------------------------------------------

class TestDistinctDistributions:
    def test_kl_divergence_gt_threshold(self, strong_model, feat_df):
        """Conservative vs aggressive must have KL divergence > 0.001."""
        builder = _make_builder(feat_df)
        conservative = builder.build_conservative(strong_model, temperature=5.0)
        aggressive = builder.build_aggressive(strong_model, temperature=1.0)
        kl = kl_divergence(conservative, aggressive)
        assert kl > 0.001, (
            f"KL divergence too small: {kl:.6f}. "
            "Conservative and aggressive submissions are too similar."
        )

    def test_kl_divergence_with_dataframes(self, strong_model, feat_df):
        """kl_divergence should accept raw DataFrames."""
        builder = _make_builder(feat_df)
        conservative = builder.build_conservative(strong_model, temperature=5.0)
        aggressive = builder.build_aggressive(strong_model, temperature=1.0)
        kl = kl_divergence(conservative.df, aggressive.df)
        assert kl > 0.001

    def test_kl_identical_distributions_near_zero(self, strong_model, feat_df):
        """KL divergence of identical distributions should be near zero."""
        builder = _make_builder(feat_df)
        v1 = builder.build_aggressive(strong_model, temperature=1.0)
        v2 = builder.build_aggressive(strong_model, temperature=1.0)
        kl = kl_divergence(v1, v2)
        assert kl < 1e-4

    def test_conservative_mean_closer_to_half(self, strong_model, feat_df):
        """Conservative mean should be closer to 0.5 than aggressive mean."""
        builder = _make_builder(feat_df)
        cons = builder.build_conservative(strong_model, temperature=5.0)
        aggr = builder.build_aggressive(strong_model, temperature=1.0)
        dist_cons = abs(float(cons.df["Pred"].mean()) - 0.5)
        dist_aggr = abs(float(aggr.df["Pred"].mean()) - 0.5)
        assert dist_cons <= dist_aggr + 0.05

    def test_both_predictions_differ_element_wise(self, strong_model, feat_df):
        """At least some individual predictions must differ between variants."""
        builder = _make_builder(feat_df)
        cons = builder.build_conservative(strong_model, temperature=5.0)
        aggr = builder.build_aggressive(strong_model, temperature=1.0)
        merged = cons.df.merge(aggr.df, on="ID", suffixes=("_c", "_a"))
        diffs = (merged["Pred_c"] - merged["Pred_a"]).abs()
        assert diffs.max() > 0.001


# ---------------------------------------------------------------------------
# SubmissionVariant structure
# ---------------------------------------------------------------------------

class TestSubmissionVariant:
    def test_len(self, strong_model, feat_df):
        builder = _make_builder(feat_df)
        v = builder.build_aggressive(strong_model)
        assert len(v) == len(v.df)

    def test_validate_returns_dict(self, strong_model, feat_df):
        builder = _make_builder(feat_df)
        v = builder.build_aggressive(strong_model)
        result = v.validate(expected_season=SEASON)
        assert isinstance(result, dict)
        assert "valid" in result

    def test_save_csv(self, strong_model, feat_df, tmp_path):
        builder = _make_builder(feat_df)
        v = builder.build_aggressive(strong_model)
        path = tmp_path / "sub.csv"
        v.save_csv(str(path))
        assert path.exists()
        reloaded = pd.read_csv(path)
        assert "ID" in reloaded.columns
        assert "Pred" in reloaded.columns
