"""
test_baseline.py
================
Tests for ncaa_model.baseline.LogisticBaseline:
  1. Model fits without error on synthetic data.
  2. predict_proba_matchup returns float in [0, 1].
  3. Symmetry: P(A beats B) + P(B beats A) = 1.
  4. Better-featured team should be predicted as more likely to win.
  5. Serialisation (save / load) preserves predictions exactly.
  6. Unfitted model raises RuntimeError on predict.
  7. Model satisfies NCAAPredictor protocol.
  8. Brier score on synthetic validation set < 0.25 (better than random).
"""

from __future__ import annotations

import pickle
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ncaa_model.baseline import LogisticBaseline
from ncaa_model.evaluate import NCAAPredictor, brier_score, evaluate_season
from ncaa_model.matchup import DEFAULT_FEATURE_COLS, impute_features
from ncaa_model.tests.conftest import (
    M_TEAM_IDS, TRAIN_SEASONS, VAL_SEASON,
    make_feat_df, make_tourney_df, TIER_STATS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fitted(feat_df, tourney_df):
    model = LogisticBaseline()
    model.fit(feat_df, tourney_df, TRAIN_SEASONS)
    return model


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

class TestFit:

    def test_fit_returns_self(self, feat_df, tourney_df):
        model = LogisticBaseline()
        result = model.fit(feat_df, tourney_df, TRAIN_SEASONS)
        assert result is model

    def test_is_fitted_after_fit(self, feat_df, tourney_df):
        model = LogisticBaseline()
        assert not model.is_fitted
        model.fit(feat_df, tourney_df, TRAIN_SEASONS)
        assert model.is_fitted

    def test_feature_cols_accessible(self, fitted):
        assert isinstance(fitted.feature_cols, list)
        assert len(fitted.feature_cols) > 0

    def test_default_feature_cols_used(self):
        model = LogisticBaseline()
        assert model.feature_cols == DEFAULT_FEATURE_COLS

    def test_custom_feature_cols(self, feat_df, tourney_df):
        subset = ["net_rtg", "win_pct", "seed"]
        model = LogisticBaseline(feature_cols=subset)
        model.fit(feat_df, tourney_df, TRAIN_SEASONS)
        assert model.feature_cols == subset

    def test_unfitted_raises(self):
        model = LogisticBaseline()
        feat = pd.Series({c: 0.0 for c in DEFAULT_FEATURE_COLS})
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict_proba_matchup(feat, feat)


# ---------------------------------------------------------------------------
# predict_proba_matchup
# ---------------------------------------------------------------------------

class TestPredictProba:

    def test_returns_float(self, fitted, feat_df):
        feat_imp = impute_features(feat_df, DEFAULT_FEATURE_COLS)
        fidx = (
            feat_imp[(feat_imp["gender"] == "M") & (feat_imp["season"] == VAL_SEASON)]
            .set_index("team_id")[DEFAULT_FEATURE_COLS]
        )
        fa = fidx.loc[M_TEAM_IDS[0]]   # tier-1 team
        fb = fidx.loc[M_TEAM_IDS[-1]]  # tier-4 team
        p = fitted.predict_proba_matchup(fa, fb)
        assert isinstance(p, float)

    def test_probability_in_0_1(self, fitted, feat_df):
        feat_imp = impute_features(feat_df, DEFAULT_FEATURE_COLS)
        fidx = (
            feat_imp[(feat_imp["gender"] == "M") & (feat_imp["season"] == VAL_SEASON)]
            .set_index("team_id")[DEFAULT_FEATURE_COLS]
        )
        for tida, tidb in [(M_TEAM_IDS[0], M_TEAM_IDS[-1]),
                           (M_TEAM_IDS[4], M_TEAM_IDS[8]),
                           (M_TEAM_IDS[2], M_TEAM_IDS[2])]:
            p = fitted.predict_proba_matchup(fidx.loc[tida], fidx.loc[tidb])
            assert 0.0 <= p <= 1.0, f"P={p} out of range for {tida} vs {tidb}"

    def test_symmetry(self, fitted, feat_df):
        """P(A beats B) + P(B beats A) must equal 1.0."""
        feat_imp = impute_features(feat_df, DEFAULT_FEATURE_COLS)
        fidx = (
            feat_imp[(feat_imp["gender"] == "M") & (feat_imp["season"] == VAL_SEASON)]
            .set_index("team_id")[DEFAULT_FEATURE_COLS]
        )
        for i in range(0, N := min(8, len(M_TEAM_IDS) - 1)):
            fa = fidx.loc[M_TEAM_IDS[i]]
            fb = fidx.loc[M_TEAM_IDS[i + 1]]
            p_ab = fitted.predict_proba_matchup(fa, fb)
            p_ba = fitted.predict_proba_matchup(fb, fa)
            assert abs(p_ab + p_ba - 1.0) < 1e-9, (
                f"Symmetry violated: P(A|B)={p_ab:.6f} + P(B|A)={p_ba:.6f} ≠ 1"
            )

    def test_better_team_predicted_to_win(self, fitted, feat_df):
        """Tier-1 team should be predicted to beat tier-4 team (p > 0.5)."""
        feat_imp = impute_features(feat_df, DEFAULT_FEATURE_COLS)
        fidx = (
            feat_imp[(feat_imp["gender"] == "M") & (feat_imp["season"] == VAL_SEASON)]
            .set_index("team_id")[DEFAULT_FEATURE_COLS]
        )
        tier1 = fidx.loc[M_TEAM_IDS[0]]    # best team (tier 1)
        tier4 = fidx.loc[M_TEAM_IDS[-1]]   # worst team (tier 4)

        # Ensure tier1 has lower TeamID → it is "team_a" in our convention
        if M_TEAM_IDS[0] < M_TEAM_IDS[-1]:
            p = fitted.predict_proba_matchup(tier1, tier4)
            assert p > 0.5, f"Expected tier-1 to be favoured; got P={p:.3f}"

    def test_equal_teams_near_50pct(self, fitted, feat_df):
        """When two teams have identical features, prediction ≈ 0.5."""
        feat_imp = impute_features(feat_df, DEFAULT_FEATURE_COLS)
        fidx = (
            feat_imp[(feat_imp["gender"] == "M") & (feat_imp["season"] == VAL_SEASON)]
            .set_index("team_id")[DEFAULT_FEATURE_COLS]
        )
        fa = fidx.loc[M_TEAM_IDS[0]]
        # Predict self-vs-self
        p = fitted.predict_proba_matchup(fa, fa)
        assert abs(p - 0.5) < 0.05, (
            f"Self-vs-self prediction should be ≈ 0.5, got {p:.4f}"
        )

    def test_predict_proba_batch_consistent(self, fitted, feat_df):
        """predict_proba_batch should match individual calls."""
        from ncaa_model.matchup import build_prediction_pairs
        team_ids = M_TEAM_IDS[:4]
        X, meta = build_prediction_pairs(
            feat_df, team_ids, "M", VAL_SEASON, DEFAULT_FEATURE_COLS
        )
        probs_batch = fitted.predict_proba_batch(X)
        feat_imp = impute_features(feat_df, DEFAULT_FEATURE_COLS)
        fidx = (
            feat_imp[(feat_imp["gender"] == "M") & (feat_imp["season"] == VAL_SEASON)]
            .set_index("team_id")[DEFAULT_FEATURE_COLS]
        )
        for i, row in meta.iterrows():
            fa = fidx.loc[row["team_id_low"]]
            fb = fidx.loc[row["team_id_high"]]
            p_single = fitted.predict_proba_matchup(fa, fb)
            assert abs(probs_batch[i] - p_single) < 1e-9


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------

class TestProtocol:

    def test_implements_ncaa_predictor_protocol(self, fitted):
        assert isinstance(fitted, NCAAPredictor), (
            "LogisticBaseline does not satisfy NCAAPredictor protocol"
        )

    def test_feature_cols_is_list(self, fitted):
        assert isinstance(fitted.feature_cols, list)

    def test_repr_contains_key_info(self, fitted):
        r = repr(fitted)
        assert "LogisticBaseline" in r
        assert "fitted" in r


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

class TestSerialisation:

    def test_save_and_load_preserves_predictions(self, fitted, feat_df):
        feat_imp = impute_features(feat_df, DEFAULT_FEATURE_COLS)
        fidx = (
            feat_imp[(feat_imp["gender"] == "M") & (feat_imp["season"] == VAL_SEASON)]
            .set_index("team_id")[DEFAULT_FEATURE_COLS]
        )
        fa = fidx.loc[M_TEAM_IDS[0]]
        fb = fidx.loc[M_TEAM_IDS[-1]]
        p_before = fitted.predict_proba_matchup(fa, fb)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = Path(f.name)

        try:
            fitted.save(path)
            loaded = LogisticBaseline.load(path)
            p_after = loaded.predict_proba_matchup(fa, fb)
            assert abs(p_before - p_after) < 1e-12
        finally:
            path.unlink(missing_ok=True)

    def test_load_sets_is_fitted(self, fitted):
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = Path(f.name)
        try:
            fitted.save(path)
            loaded = LogisticBaseline.load(path)
            assert loaded.is_fitted
        finally:
            path.unlink(missing_ok=True)

    def test_load_preserves_feature_cols(self, fitted):
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = Path(f.name)
        try:
            fitted.save(path)
            loaded = LogisticBaseline.load(path)
            assert loaded.feature_cols == fitted.feature_cols
        finally:
            path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Brier score sanity
# ---------------------------------------------------------------------------

class TestBrierScore:

    def test_brier_better_than_random(self, fitted, feat_df, tourney_df):
        """
        Fitted model should achieve Brier < 0.25 (better than always-0.5 baseline).

        Note: on real NCAA data, the target is < 0.20.  For 16-team synthetic
        data with moderate signal, we assert < 0.25 as a sanity check.
        """
        result = evaluate_season(
            fitted, feat_df, tourney_df, VAL_SEASON, gender="both"
        )
        assert result.brier_score < 0.25, (
            f"Brier score {result.brier_score:.4f} ≥ 0.25 "
            "(model performs worse than always-0.5 baseline)"
        )

    def test_brier_not_zero(self, fitted, feat_df, tourney_df):
        """Perfect Brier=0 would indicate data leakage; verify it's > 0."""
        result = evaluate_season(
            fitted, feat_df, tourney_df, VAL_SEASON, gender="both"
        )
        assert result.brier_score > 0.0

    def test_n_games_positive(self, fitted, feat_df, tourney_df):
        result = evaluate_season(
            fitted, feat_df, tourney_df, VAL_SEASON, gender="both"
        )
        assert result.n_games > 0
