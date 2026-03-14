"""
test_ensemble.py
================
Unit tests for Stage 06 ensemble module.

Tests cover:
- SimpleAverageEnsemble: averaging, shape, edge cases
- BrierWeightedEnsemble: weight computation, weight positivity, weighting math
- ContextualEnsemble: context dispatch, shape
- MetaLearnerEnsemble: stacking weights, positivity, OOF fitting
- Edge cases: identical teams → ~0.5, extreme mismatches → [0.90, 0.99]
"""

from __future__ import annotations

import numpy as np
import pytest

from ncaa_models.baseline import FEATURE_COLS, LogisticBaseline
from ncaa_models.ensemble import (
    BrierWeightedEnsemble,
    ContextualEnsemble,
    SimpleAverageEnsemble,
    build_simple_ensemble,
    build_weighted_ensemble,
)
from ncaa_models.evaluate import compute_brier_score
from ncaa_models.meta_learner import MetaLearnerEnsemble
from ncaa_models.upset_detector import UpsetDetector

from .conftest import (
    M_TEAM_IDS,
    TRAIN_SEASONS,
    VAL_SEASON,
    make_feat_df,
    make_tourney_df,
)

FEATURE_DIM = len(FEATURE_COLS)
FEAT_IDX = {c: i for i, c in enumerate(FEATURE_COLS)}


# ---------------------------------------------------------------------------
# Shared helpers and fixtures
# ---------------------------------------------------------------------------

def _build_arrays():
    """Return (X1_tr, X2_tr, y_tr, X1_v, X2_v, y_v) from synthetic data."""
    from ncaa_models.cv import build_matchup_df
    feat_df = make_feat_df(M_TEAM_IDS, TRAIN_SEASONS + [VAL_SEASON])
    tourney_df = make_tourney_df(M_TEAM_IDS, TRAIN_SEASONS + [VAL_SEASON])
    matchup_df = build_matchup_df(feat_df, tourney_df, FEATURE_COLS, is_tourney=True)

    train = matchup_df[matchup_df["season"].isin(TRAIN_SEASONS)]
    val = matchup_df[matchup_df["season"] == VAL_SEASON]

    X1_tr = np.stack(train["X_team1"].values)
    X2_tr = np.stack(train["X_team2"].values)
    y_tr = train["y"].values.astype(int)
    X1_v = np.stack(val["X_team1"].values)
    X2_v = np.stack(val["X_team2"].values)
    y_v = val["y"].values.astype(int)
    return X1_tr, X2_tr, y_tr, X1_v, X2_v, y_v


def _make_lr() -> LogisticBaseline:
    return LogisticBaseline()


def _make_lr2() -> LogisticBaseline:
    """Second LR variant with different regularisation."""
    return LogisticBaseline(C=0.1)


def _seed1_features() -> np.ndarray:
    """Feature vector for a seed-1 quality team (rank 0)."""
    X = np.zeros(FEATURE_DIM)
    fi = FEAT_IDX
    X[fi["net_rtg"]] = 20.0
    X[fi["ortg"]] = 118.0
    X[fi["drtg"]] = 98.0
    X[fi["win_pct"]] = 0.88
    X[fi["seed"]] = 1.0
    X[fi["possessions_pg"]] = 70.0
    X[fi["fg_pct"]] = 0.45
    X[fi["fg3_pct"]] = 0.36
    X[fi["tov_rate"]] = 0.14
    X[fi["games_played"]] = 30
    return X


def _seed8_features() -> np.ndarray:
    """Feature vector for a seed-8 quality team (rank 7)."""
    X = np.zeros(FEATURE_DIM)
    fi = FEAT_IDX
    X[fi["net_rtg"]] = -20.0
    X[fi["ortg"]] = 88.0
    X[fi["drtg"]] = 110.0
    X[fi["win_pct"]] = 0.18
    X[fi["seed"]] = 8.0
    X[fi["possessions_pg"]] = 73.5
    X[fi["fg_pct"]] = 0.41
    X[fi["fg3_pct"]] = 0.32
    X[fi["tov_rate"]] = 0.22
    X[fi["games_played"]] = 30
    return X


# ---------------------------------------------------------------------------
# SimpleAverageEnsemble
# ---------------------------------------------------------------------------

class TestSimpleAverageEnsemble:
    def test_predict_proba_shape(self):
        X1_tr, X2_tr, y_tr, X1_v, X2_v, y_v = _build_arrays()
        ens = SimpleAverageEnsemble([_make_lr(), _make_lr2()])
        ens.fit(X1_tr, X2_tr, y_tr)
        out = ens.predict_proba(X1_v, X2_v)
        assert out.shape == (len(y_v),)

    def test_probs_in_unit_interval(self):
        X1_tr, X2_tr, y_tr, X1_v, X2_v, y_v = _build_arrays()
        ens = SimpleAverageEnsemble([_make_lr(), _make_lr2()])
        ens.fit(X1_tr, X2_tr, y_tr)
        out = ens.predict_proba(X1_v, X2_v)
        assert np.all(out >= 0.0) and np.all(out <= 1.0)

    def test_is_arithmetic_mean(self):
        """Output should equal the arithmetic mean of component predictions."""
        X1_tr, X2_tr, y_tr, X1_v, X2_v, y_v = _build_arrays()
        m1, m2 = _make_lr(), _make_lr2()
        m1.fit(X1_tr, X2_tr, y_tr)
        m2.fit(X1_tr, X2_tr, y_tr)
        p1 = m1.predict_proba(X1_v, X2_v)
        p2 = m2.predict_proba(X1_v, X2_v)
        expected = (p1 + p2) / 2.0

        ens = SimpleAverageEnsemble([_make_lr(), _make_lr2()])
        ens.fit(X1_tr, X2_tr, y_tr)
        out = ens.predict_proba(X1_v, X2_v)
        np.testing.assert_array_almost_equal(out, expected, decimal=6)

    def test_single_model_passthrough(self):
        """With one model, output = that model's predictions."""
        X1_tr, X2_tr, y_tr, X1_v, X2_v, y_v = _build_arrays()
        m = _make_lr()
        m.fit(X1_tr, X2_tr, y_tr)
        expected = m.predict_proba(X1_v, X2_v)

        ens = SimpleAverageEnsemble([_make_lr()])
        ens.fit(X1_tr, X2_tr, y_tr)
        out = ens.predict_proba(X1_v, X2_v)
        np.testing.assert_array_almost_equal(out, expected, decimal=6)

    def test_empty_models_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            SimpleAverageEnsemble([])

    def test_fit_returns_self(self):
        X1_tr, X2_tr, y_tr, *_ = _build_arrays()
        ens = SimpleAverageEnsemble([_make_lr()])
        result = ens.fit(X1_tr, X2_tr, y_tr)
        assert result is ens

    def test_repr(self):
        ens = SimpleAverageEnsemble([_make_lr()])
        assert "SimpleAverageEnsemble" in repr(ens)

    def test_build_factory(self):
        X1_tr, X2_tr, y_tr, X1_v, X2_v, y_v = _build_arrays()
        ens = build_simple_ensemble([_make_lr(), _make_lr2()], X1_tr, X2_tr, y_tr)
        out = ens.predict_proba(X1_v, X2_v)
        assert out.shape == (len(y_v),)


# ---------------------------------------------------------------------------
# BrierWeightedEnsemble
# ---------------------------------------------------------------------------

class TestBrierWeightedEnsemble:
    def test_weights_sum_to_one(self):
        X1_tr, X2_tr, y_tr, X1_v, X2_v, y_v = _build_arrays()
        ens = BrierWeightedEnsemble([_make_lr(), _make_lr2()])
        ens.fit(X1_tr, X2_tr, y_tr, X1_v, X2_v, y_v)
        assert abs(ens.weights_.sum() - 1.0) < 1e-9

    def test_all_positive_weights(self):
        """SPEC REQUIREMENT: All component models have strictly positive weights."""
        X1_tr, X2_tr, y_tr, X1_v, X2_v, y_v = _build_arrays()
        ens = BrierWeightedEnsemble([_make_lr(), _make_lr2()])
        ens.fit(X1_tr, X2_tr, y_tr, X1_v, X2_v, y_v)
        assert np.all(ens.weights_ > 0.0), (
            f"Some weights are not strictly positive: {ens.weights_}"
        )

    def test_weights_shape(self):
        X1_tr, X2_tr, y_tr, X1_v, X2_v, y_v = _build_arrays()
        ens = BrierWeightedEnsemble([_make_lr(), _make_lr2()])
        ens.fit(X1_tr, X2_tr, y_tr, X1_v, X2_v, y_v)
        assert ens.weights_.shape == (2,)

    def test_predict_proba_shape(self):
        X1_tr, X2_tr, y_tr, X1_v, X2_v, y_v = _build_arrays()
        ens = BrierWeightedEnsemble([_make_lr(), _make_lr2()])
        ens.fit(X1_tr, X2_tr, y_tr, X1_v, X2_v, y_v)
        out = ens.predict_proba(X1_v, X2_v)
        assert out.shape == (len(y_v),)

    def test_probs_in_unit_interval(self):
        X1_tr, X2_tr, y_tr, X1_v, X2_v, y_v = _build_arrays()
        ens = BrierWeightedEnsemble([_make_lr(), _make_lr2()])
        ens.fit(X1_tr, X2_tr, y_tr, X1_v, X2_v, y_v)
        out = ens.predict_proba(X1_v, X2_v)
        assert np.all(out >= 0.0) and np.all(out <= 1.0)

    def test_in_sample_weights_when_no_val(self):
        """Without val data, training Brier is used — weights still sum to 1."""
        X1_tr, X2_tr, y_tr, *_ = _build_arrays()
        ens = BrierWeightedEnsemble([_make_lr(), _make_lr2()])
        ens.fit(X1_tr, X2_tr, y_tr)
        assert ens.weights_ is not None
        assert abs(ens.weights_.sum() - 1.0) < 1e-9

    def test_predict_before_fit_raises(self):
        ens = BrierWeightedEnsemble([_make_lr()])
        with pytest.raises(RuntimeError, match="fitted"):
            ens.predict_proba(np.zeros((1, FEATURE_DIM)), np.zeros((1, FEATURE_DIM)))

    def test_repr_after_fit(self):
        X1_tr, X2_tr, y_tr, X1_v, X2_v, y_v = _build_arrays()
        ens = BrierWeightedEnsemble([_make_lr(), _make_lr2()])
        ens.fit(X1_tr, X2_tr, y_tr, X1_v, X2_v, y_v)
        assert "BrierWeightedEnsemble" in repr(ens)

    def test_build_factory(self):
        X1_tr, X2_tr, y_tr, X1_v, X2_v, y_v = _build_arrays()
        ens = build_weighted_ensemble(
            [_make_lr(), _make_lr2()], X1_tr, X2_tr, y_tr, X1_v, X2_v, y_v
        )
        assert ens.weights_ is not None


# ---------------------------------------------------------------------------
# ContextualEnsemble
# ---------------------------------------------------------------------------

class TestContextualEnsemble:
    @pytest.fixture(scope="class")
    def fitted_ensemble(self):
        X1_tr, X2_tr, y_tr, X1_v, X2_v, y_v = _build_arrays()
        models = [_make_lr(), _make_lr2()]
        # Fit base models first so UpsetDetector can use one for classification
        lr_for_detector = _make_lr()
        lr_for_detector.fit(X1_tr, X2_tr, y_tr)
        detector = UpsetDetector(model=lr_for_detector, feature_cols=FEATURE_COLS)
        ens = ContextualEnsemble(models, upset_detector=detector)
        ens.fit(X1_tr, X2_tr, y_tr)
        return ens, X1_v, X2_v, y_v

    def test_predict_proba_shape(self, fitted_ensemble):
        ens, X1_v, X2_v, y_v = fitted_ensemble
        out = ens.predict_proba(X1_v, X2_v)
        assert out.shape == (len(y_v),)

    def test_probs_in_unit_interval(self, fitted_ensemble):
        ens, X1_v, X2_v, y_v = fitted_ensemble
        out = ens.predict_proba(X1_v, X2_v)
        assert np.all(out >= 0.0) and np.all(out <= 1.0)

    def test_fit_returns_self(self):
        X1_tr, X2_tr, y_tr, *_ = _build_arrays()
        lr = _make_lr()
        lr.fit(X1_tr, X2_tr, y_tr)
        detector = UpsetDetector(model=lr, feature_cols=FEATURE_COLS)
        ens = ContextualEnsemble([_make_lr()], upset_detector=detector)
        result = ens.fit(X1_tr, X2_tr, y_tr)
        assert result is ens

    def test_custom_weight_profiles(self):
        X1_tr, X2_tr, y_tr, X1_v, X2_v, y_v = _build_arrays()
        lr = _make_lr()
        lr.fit(X1_tr, X2_tr, y_tr)
        detector = UpsetDetector(model=lr, feature_cols=FEATURE_COLS)

        from ncaa_models.upset_detector import BLOWOUT_LIKELY, COMPETITIVE, UPSET_PLAUSIBLE
        profiles = {
            BLOWOUT_LIKELY: np.array([0.9, 0.1]),
            COMPETITIVE: np.array([0.5, 0.5]),
            UPSET_PLAUSIBLE: np.array([0.1, 0.9]),
        }
        ens = ContextualEnsemble([_make_lr(), _make_lr2()], detector, profiles)
        ens.fit(X1_tr, X2_tr, y_tr)
        out = ens.predict_proba(X1_v, X2_v)
        assert out.shape == (len(y_v),)

    def test_repr(self):
        lr = _make_lr()
        detector = UpsetDetector(model=lr, feature_cols=FEATURE_COLS)
        ens = ContextualEnsemble([_make_lr()], upset_detector=detector)
        assert "ContextualEnsemble" in repr(ens)


# ---------------------------------------------------------------------------
# MetaLearnerEnsemble
# ---------------------------------------------------------------------------

class TestMetaLearnerEnsemble:
    def test_fit_predict_shape(self):
        X1_tr, X2_tr, y_tr, X1_v, X2_v, y_v = _build_arrays()
        meta = MetaLearnerEnsemble([_make_lr(), _make_lr2()])
        meta.fit(X1_tr, X2_tr, y_tr)
        out = meta.predict_proba(X1_v, X2_v)
        assert out.shape == (len(y_v),)

    def test_probs_in_unit_interval(self):
        X1_tr, X2_tr, y_tr, X1_v, X2_v, y_v = _build_arrays()
        meta = MetaLearnerEnsemble([_make_lr(), _make_lr2()])
        meta.fit(X1_tr, X2_tr, y_tr)
        out = meta.predict_proba(X1_v, X2_v)
        assert np.all(out >= 0.01) and np.all(out <= 0.99)

    def test_strictly_positive_weights(self):
        """SPEC REQUIREMENT: All component models have strictly positive stacking weights."""
        X1_tr, X2_tr, y_tr, *_ = _build_arrays()
        meta = MetaLearnerEnsemble([_make_lr(), _make_lr2()])
        meta.fit(X1_tr, X2_tr, y_tr)
        assert np.all(meta.weights_ > 0.0), (
            f"MetaLearnerEnsemble weights not all positive: {meta.weights_}"
        )

    def test_weights_sum_to_one(self):
        X1_tr, X2_tr, y_tr, *_ = _build_arrays()
        meta = MetaLearnerEnsemble([_make_lr(), _make_lr2()])
        meta.fit(X1_tr, X2_tr, y_tr)
        assert abs(meta.weights_.sum() - 1.0) < 1e-9

    def test_weights_shape(self):
        X1_tr, X2_tr, y_tr, *_ = _build_arrays()
        meta = MetaLearnerEnsemble([_make_lr(), _make_lr2()])
        meta.fit(X1_tr, X2_tr, y_tr)
        assert meta.weights_.shape == (2,)

    def test_three_models_positive_weights(self):
        """Positive weights guaranteed with 3 component models."""
        X1_tr, X2_tr, y_tr, *_ = _build_arrays()
        meta = MetaLearnerEnsemble([_make_lr(), _make_lr2(), _make_lr()])
        meta.fit(X1_tr, X2_tr, y_tr)
        assert np.all(meta.weights_ > 0.0)

    def test_predict_before_fit_raises(self):
        meta = MetaLearnerEnsemble([_make_lr()])
        with pytest.raises(RuntimeError, match="fitted"):
            meta.predict_proba(np.zeros((1, FEATURE_DIM)), np.zeros((1, FEATURE_DIM)))

    def test_oof_fit(self):
        """fit_oof() produces the same interface as fit()."""
        from ncaa_models.cv import build_matchup_df
        feat_df = make_feat_df(M_TEAM_IDS, TRAIN_SEASONS + [VAL_SEASON])
        tourney_df = make_tourney_df(M_TEAM_IDS, TRAIN_SEASONS + [VAL_SEASON])
        matchup_df = build_matchup_df(feat_df, tourney_df, FEATURE_COLS, is_tourney=True)

        train_df = matchup_df[matchup_df["season"].isin(TRAIN_SEASONS)]
        val_df = matchup_df[matchup_df["season"] == VAL_SEASON]

        meta = MetaLearnerEnsemble([_make_lr(), _make_lr2()])
        meta.fit_oof(train_df, FEATURE_COLS, cv_seasons=[2023, 2024])

        X1_v = np.stack(val_df["X_team1"].values)
        X2_v = np.stack(val_df["X_team2"].values)
        out = meta.predict_proba(X1_v, X2_v)
        assert out.shape == (len(val_df),)
        assert np.all(out >= 0.01) and np.all(out <= 0.99)

    def test_oof_weights_positive(self):
        """OOF-fitted weights also strictly positive."""
        from ncaa_models.cv import build_matchup_df
        feat_df = make_feat_df(M_TEAM_IDS, TRAIN_SEASONS + [VAL_SEASON])
        tourney_df = make_tourney_df(M_TEAM_IDS, TRAIN_SEASONS + [VAL_SEASON])
        matchup_df = build_matchup_df(feat_df, tourney_df, FEATURE_COLS, is_tourney=True)
        train_df = matchup_df[matchup_df["season"].isin(TRAIN_SEASONS)]

        meta = MetaLearnerEnsemble([_make_lr(), _make_lr2()])
        meta.fit_oof(train_df, FEATURE_COLS, cv_seasons=[2023, 2024])
        assert np.all(meta.weights_ > 0.0)

    def test_repr(self):
        meta = MetaLearnerEnsemble([_make_lr()])
        assert "MetaLearnerEnsemble" in repr(meta)


# ---------------------------------------------------------------------------
# Edge-case tests
# ---------------------------------------------------------------------------

class TestEnsembleEdgeCases:
    """
    SPEC REQUIREMENT:
    - Identical teams → P(team1 wins) ≈ 0.5
    - Extreme mismatches (1 vs 16-equivalent) → P(favourite wins) ∈ [0.90, 0.99]
    """

    @pytest.fixture(scope="class")
    def fitted_models(self):
        X1_tr, X2_tr, y_tr, *_ = _build_arrays()
        models = [_make_lr(), _make_lr2()]
        ens = SimpleAverageEnsemble(models)
        ens.fit(X1_tr, X2_tr, y_tr)
        return ens

    def test_identical_teams_near_half(self, fitted_models):
        """
        When both teams have identical features, P(team1 wins) should be ≈ 0.5.
        Tolerance: [0.35, 0.65] to allow for numerical noise.
        """
        x = _seed1_features()
        p = fitted_models.predict_proba(x.reshape(1, -1), x.reshape(1, -1))[0]
        assert 0.35 <= p <= 0.65, (
            f"Identical teams prediction = {p:.4f}, expected near 0.5"
        )

    def test_extreme_mismatch_seed1_vs_seed8(self, fitted_models):
        """
        Seed-1 features vs seed-8 features: P(seed1 wins) should be high.

        SPEC: 'extreme mismatches (1-vs-16) → probability in [0.90, 0.99]'
        (Seed 8 is the largest gap in our synthetic 8-team bracket.)
        """
        x1 = _seed1_features()
        x8 = _seed8_features()
        # team1 = lower ID → seed1 should dominate
        p = fitted_models.predict_proba(x1.reshape(1, -1), x8.reshape(1, -1))[0]
        assert 0.90 <= p <= 0.99, (
            f"Extreme mismatch prediction = {p:.4f}, expected in [0.90, 0.99]"
        )

    def test_extreme_mismatch_reversed(self, fitted_models):
        """Reversed: seed8 as team1 vs seed1 as team2 → P(team1 wins) should be LOW."""
        x1 = _seed1_features()
        x8 = _seed8_features()
        p = fitted_models.predict_proba(x8.reshape(1, -1), x1.reshape(1, -1))[0]
        assert p <= 0.10, (
            f"Reversed extreme mismatch = {p:.4f}, expected ≤ 0.10"
        )

    def test_symmetry_property(self, fitted_models):
        """P(A beats B) + P(B beats A) ≈ 1 for any matchup."""
        x1 = _seed1_features()
        x8 = _seed8_features()
        p_ab = fitted_models.predict_proba(x1.reshape(1, -1), x8.reshape(1, -1))[0]
        p_ba = fitted_models.predict_proba(x8.reshape(1, -1), x1.reshape(1, -1))[0]
        # Perfect symmetry is a property of individual LogisticBaseline models
        # but not of arbitrary ensembles.  Check it's roughly symmetric.
        assert abs(p_ab + p_ba - 1.0) < 0.05, (
            f"Symmetry broken: P(A>B)={p_ab:.4f}, P(B>A)={p_ba:.4f}"
        )

    def test_weighted_ensemble_extreme_mismatch(self):
        """BrierWeightedEnsemble also gives high probability for extreme mismatch."""
        X1_tr, X2_tr, y_tr, X1_v, X2_v, y_v = _build_arrays()
        ens = BrierWeightedEnsemble([_make_lr(), _make_lr2()])
        ens.fit(X1_tr, X2_tr, y_tr, X1_v, X2_v, y_v)

        x1 = _seed1_features()
        x8 = _seed8_features()
        p = ens.predict_proba(x1.reshape(1, -1), x8.reshape(1, -1))[0]
        assert 0.90 <= p <= 0.99, (
            f"Weighted ensemble extreme mismatch = {p:.4f}, expected in [0.90, 0.99]"
        )

    def test_meta_learner_extreme_mismatch(self):
        """MetaLearnerEnsemble gives high probability for extreme mismatch."""
        X1_tr, X2_tr, y_tr, *_ = _build_arrays()
        meta = MetaLearnerEnsemble([_make_lr(), _make_lr2()])
        meta.fit(X1_tr, X2_tr, y_tr)

        x1 = _seed1_features()
        x8 = _seed8_features()
        p = meta.predict_proba(x1.reshape(1, -1), x8.reshape(1, -1))[0]
        assert 0.90 <= p <= 0.99, (
            f"MetaLearnerEnsemble extreme mismatch = {p:.4f}, expected in [0.90, 0.99]"
        )
