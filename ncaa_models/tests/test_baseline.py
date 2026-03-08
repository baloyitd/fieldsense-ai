"""
test_baseline.py
================
Unit tests for ncaa_models.baseline.LogisticBaseline.

Covers
------
1. Abstract interface enforcement (MatchupPredictor ABC).
2. Predict-before-fit raises RuntimeError.
3. Basic fit + predict_proba cycle.
4. UNIT TEST 1: Probabilities strictly in [0.01, 0.99] for all matchups
   including extreme edge cases.
5. Symmetry property: P(A,B) + P(B,A) = 1.
6. Equal teams produce predictions near 0.50.
7. Better team (clearly superior stats) receives higher probability.
8. Save / load round-trip preserves predictions.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from ncaa_models.base import MatchupPredictor
from ncaa_models.baseline import (
    FEATURE_COLS,
    PROB_CLIP_HIGH,
    PROB_CLIP_LOW,
    LogisticBaseline,
)
from ncaa_models.tests.conftest import M_TEAM_IDS, TRAIN_SEASONS


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class TestMatchupPredictorABC:

    def test_cannot_instantiate_abc_directly(self):
        """MatchupPredictor is abstract and must not be directly instantiated."""
        with pytest.raises(TypeError):
            MatchupPredictor()  # type: ignore[abstract]

    def test_logistic_baseline_is_subclass(self):
        assert issubclass(LogisticBaseline, MatchupPredictor)

    def test_logistic_baseline_is_instance(self):
        model = LogisticBaseline()
        assert isinstance(model, MatchupPredictor)


# ---------------------------------------------------------------------------
# Lifecycle guard
# ---------------------------------------------------------------------------

class TestPredictBeforeFit:

    def test_predict_raises_before_fit(self):
        model = LogisticBaseline()
        X = np.random.rand(3, len(FEATURE_COLS))
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict_proba(X, X)

    def test_is_fitted_false_before_fit(self):
        assert not LogisticBaseline()._is_fitted


# ---------------------------------------------------------------------------
# Basic fit + predict
# ---------------------------------------------------------------------------

class TestBasicFitPredict:

    def test_fit_sets_is_fitted(self, train_arrays):
        X1, X2, y = train_arrays
        model = LogisticBaseline()
        model.fit(X1, X2, y)
        assert model._is_fitted

    def test_predict_returns_ndarray(self, fitted_model, val_arrays):
        X1, X2, _ = val_arrays
        probs = fitted_model.predict_proba(X1, X2)
        assert isinstance(probs, np.ndarray)

    def test_predict_shape_matches_input(self, fitted_model, val_arrays):
        X1, X2, _ = val_arrays
        probs = fitted_model.predict_proba(X1, X2)
        assert probs.shape == (len(X1),)

    def test_feature_cols_attribute(self):
        model = LogisticBaseline()
        assert model.feature_cols == FEATURE_COLS

    def test_custom_feature_cols(self, train_arrays):
        X1, X2, y = train_arrays
        subset = ["net_rtg", "win_pct", "seed"]
        subset_idx = [FEATURE_COLS.index(c) for c in subset]
        model = LogisticBaseline(feature_cols=subset)
        model.fit(X1[:, subset_idx], X2[:, subset_idx], y)
        probs = model.predict_proba(X1[:, subset_idx], X2[:, subset_idx])
        assert probs.shape == (len(X1),)


# ---------------------------------------------------------------------------
# UNIT TEST 1: Probability clipping
# ---------------------------------------------------------------------------

class TestProbabilityClipping:
    """
    UNIT TEST 1: verify logistic regression outputs probabilities strictly
    in [0.01, 0.99] for all matchups including edge cases.
    """

    def test_all_probs_at_least_clip_low(self, fitted_model, val_arrays):
        X1, X2, _ = val_arrays
        probs = fitted_model.predict_proba(X1, X2)
        assert (probs >= PROB_CLIP_LOW).all(), (
            f"Some probabilities < {PROB_CLIP_LOW}: min={probs.min():.6f}"
        )

    def test_all_probs_at_most_clip_high(self, fitted_model, val_arrays):
        X1, X2, _ = val_arrays
        probs = fitted_model.predict_proba(X1, X2)
        assert (probs <= PROB_CLIP_HIGH).all(), (
            f"Some probabilities > {PROB_CLIP_HIGH}: max={probs.max():.6f}"
        )

    def test_extreme_matchup_clipped(self, train_arrays):
        """
        An extreme team differential should produce probability at the clip boundary,
        not outside [0.01, 0.99].
        """
        X1, X2, y = train_arrays
        model = LogisticBaseline()
        model.fit(X1, X2, y)

        n_features = X1.shape[1]
        # Wildly dominant team1 vs a placeholder team2
        X_extreme_1 = np.full((1, n_features), 1000.0)
        X_extreme_2 = np.zeros((1, n_features))
        prob_high = model.predict_proba(X_extreme_1, X_extreme_2)
        assert prob_high[0] <= PROB_CLIP_HIGH

        # Wildly inferior team1
        prob_low = model.predict_proba(X_extreme_2, X_extreme_1)
        assert prob_low[0] >= PROB_CLIP_LOW

    def test_identical_teams_near_50pct(self, fitted_model, val_arrays):
        X1, _, _ = val_arrays
        # Same team vs itself — should be clipped to around 0.5
        probs = fitted_model.predict_proba(X1, X1)
        assert (probs >= PROB_CLIP_LOW).all()
        assert (probs <= PROB_CLIP_HIGH).all()
        # All predictions should be near 0.5 when teams are equal
        assert np.allclose(probs, 0.5, atol=0.05), (
            f"Self-matchup predictions far from 0.5: {probs}"
        )

    def test_single_matchup(self, fitted_model, val_arrays):
        """Single-row predict_proba works correctly."""
        X1, X2, _ = val_arrays
        prob = fitted_model.predict_proba(X1[:1], X2[:1])
        assert prob.shape == (1,)
        assert PROB_CLIP_LOW <= prob[0] <= PROB_CLIP_HIGH


# ---------------------------------------------------------------------------
# Symmetry
# ---------------------------------------------------------------------------

class TestSymmetry:

    def test_p_ab_plus_p_ba_equals_one(self, fitted_model, val_arrays):
        """P(A beats B) + P(B beats A) = 1 for every matchup."""
        X1, X2, _ = val_arrays
        p_ab = fitted_model.predict_proba(X1, X2)
        p_ba = fitted_model.predict_proba(X2, X1)
        np.testing.assert_allclose(
            p_ab + p_ba, 1.0, atol=1e-6,
            err_msg="Symmetry violated: P(A,B) + P(B,A) != 1",
        )

    def test_stronger_team_higher_prob(self, fitted_model, val_arrays):
        """
        The best team (rank 0, seed 1) should have probability > 0.5
        against the worst team (rank 7, seed 8).
        """
        X1, X2, _ = val_arrays
        # Best team: X1[0] (team 1001, seed 1)
        # Worst team: X2[-1] — but just use known positions
        # X1 indexed by matchup; use a direct 1-row predict
        from ncaa_models.tests.conftest import make_feat_df, M_TEAM_IDS, ALL_SEASONS
        fd = make_feat_df(M_TEAM_IDS, [2025])
        feat_idx = fd.set_index("team_id")[FEATURE_COLS]
        x_best = feat_idx.loc[M_TEAM_IDS[0]].values.reshape(1, -1)  # seed 1
        x_worst = feat_idx.loc[M_TEAM_IDS[7]].values.reshape(1, -1)  # seed 8

        prob = fitted_model.predict_proba(x_best, x_worst)
        assert prob[0] > 0.5, (
            f"Best team should have prob > 0.5 vs worst team, got {prob[0]:.4f}"
        )


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

class TestSaveLoad:

    def test_save_creates_file(self, fitted_model, tmp_path):
        path = tmp_path / "model.pkl"
        fitted_model.save(path)
        assert path.exists()

    def test_load_produces_same_predictions(self, fitted_model, val_arrays, tmp_path):
        X1, X2, _ = val_arrays
        path = tmp_path / "model.pkl"
        fitted_model.save(path)

        loaded = LogisticBaseline.load(path)
        probs_orig = fitted_model.predict_proba(X1, X2)
        probs_loaded = loaded.predict_proba(X1, X2)
        np.testing.assert_allclose(probs_orig, probs_loaded, atol=1e-9)

    def test_load_preserves_feature_cols(self, fitted_model, tmp_path):
        path = tmp_path / "model.pkl"
        fitted_model.save(path)
        loaded = LogisticBaseline.load(path)
        assert loaded.feature_cols == fitted_model.feature_cols

    def test_save_unfitted_raises(self):
        model = LogisticBaseline()
        with pytest.raises(RuntimeError):
            model.save("/tmp/ncaa_test_unfitted.pkl")

    def test_repr_contains_status(self):
        model = LogisticBaseline()
        assert "unfitted" in repr(model)
