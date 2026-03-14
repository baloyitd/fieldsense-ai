"""
test_cvs.py
===========
Unit tests for ncaa_models.cvs — CVSComputer and TrainingDistribution.

Tests cover:
- CVS correctly rejects implausible combinations (80% 3P% + 30 turnovers → CVS < 0.92)
- Valid perturbations achieve CVS ≥ 0.92
- All component scores (bounds, distribution, contradiction) work correctly
"""

from __future__ import annotations

import numpy as np
import pytest

from ncaa_models.baseline import FEATURE_COLS
from ncaa_models.cvs import (
    CVSComputer,
    DEFAULT_CVS_THRESHOLD,
    TrainingDistribution,
    _D1_BOUNDS,
    _D1_MEANS,
    _D1_STDS,
)

FEATURE_DIM = len(FEATURE_COLS)
FEAT_IDX = {c: i for i, c in enumerate(FEATURE_COLS)}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_cvs():
    """CVSComputer with default D1 bounds."""
    return CVSComputer()


@pytest.fixture
def canonical_X():
    """A typical seed-4 level team's feature vector — should be fully valid."""
    X = np.array([
        30.0,  # games_played
        0.65,  # win_pct
        78.0,  # ppg_scored
        72.0,  # ppg_allowed
        6.0,   # scoring_margin
        0.75,  # home_win_pct
        0.55,  # away_win_pct
        0.65,  # neutral_win_pct
        0.65,  # win_pct_last10
        0.45,  # fg_pct
        0.36,  # fg3_pct
        0.72,  # ft_pct
        0.52,  # efg_pct
        0.54,  # ts_pct
        0.30,  # orb_rate
        0.70,  # drb_rate
        72.0,  # possessions_pg
        108.0, # ortg
        100.0, # drtg
        8.0,   # net_rtg
        72.0,  # tempo
        0.17,  # tov_rate
        0.58,  # ast_rate
        0.09,  # blk_rate
        0.08,  # stl_rate
        0.05,  # ot_rate
        0.52,  # sos
        4.0,   # seed
    ], dtype=float)
    return X


@pytest.fixture
def implausible_X(canonical_X):
    """80% 3P% AND 30+ turnovers: both outside D1 bounds → CVS < 0.92."""
    X = canonical_X.copy()
    X[FEAT_IDX["fg3_pct"]] = 0.80  # Way outside [0.20, 0.50]
    X[FEAT_IDX["tov_rate"]] = 0.43  # 30 turnovers / 70 poss ≈ 0.43, outside [0.07, 0.35]
    return X


@pytest.fixture
def training_dist_from_data():
    """TrainingDistribution built from a small synthetic dataset."""
    rng = np.random.default_rng(42)
    # Generate 50 plausible feature vectors around canonical values
    means = np.array([_D1_MEANS.get(c, 0.5) for c in FEATURE_COLS])
    stds = np.array([_D1_STDS.get(c, 0.1) for c in FEATURE_COLS])
    X = rng.normal(means, stds * 0.5, size=(50, FEATURE_DIM))
    return TrainingDistribution.from_data(X, FEATURE_COLS)


# ---------------------------------------------------------------------------
# TestTrainingDistribution
# ---------------------------------------------------------------------------

class TestTrainingDistribution:

    def test_from_data_shape(self):
        X = np.random.randn(30, FEATURE_DIM)
        dist = TrainingDistribution.from_data(X, FEATURE_COLS)
        assert dist.bounds_lo.shape == (FEATURE_DIM,)
        assert dist.bounds_hi.shape == (FEATURE_DIM,)
        assert dist.mean.shape == (FEATURE_DIM,)
        assert dist.std.shape == (FEATURE_DIM,)

    def test_from_data_bounds_correct(self):
        rng = np.random.default_rng(0)
        X = rng.uniform(0, 1, size=(20, FEATURE_DIM))
        dist = TrainingDistribution.from_data(X, FEATURE_COLS)
        np.testing.assert_array_almost_equal(dist.bounds_lo, X.min(axis=0))
        np.testing.assert_array_almost_equal(dist.bounds_hi, X.max(axis=0))

    def test_std_floored_at_1e6(self):
        X = np.ones((10, FEATURE_DIM))  # constant → std=0
        dist = TrainingDistribution.from_data(X, FEATURE_COLS)
        assert (dist.std >= 1e-6).all()

    def test_default_uses_d1_bounds(self):
        dist = TrainingDistribution.default(FEATURE_COLS)
        expected_lo = np.array([_D1_BOUNDS.get(c, (-1e9, 1e9))[0] for c in FEATURE_COLS])
        np.testing.assert_array_almost_equal(dist.bounds_lo, expected_lo)

    def test_wrong_shape_raises(self):
        X = np.random.randn(10, FEATURE_DIM + 1)
        with pytest.raises(ValueError):
            TrainingDistribution.from_data(X, FEATURE_COLS)


# ---------------------------------------------------------------------------
# TestCVSComputer
# ---------------------------------------------------------------------------

class TestCVSComputer:

    def test_canonical_vector_cvs_high(self, default_cvs, canonical_X):
        """A realistic feature vector should achieve CVS ≥ 0.92."""
        cvs = default_cvs.compute(canonical_X, canonical_X)
        assert cvs >= DEFAULT_CVS_THRESHOLD, (
            f"Canonical (unperturbed) vector got CVS={cvs:.4f} < {DEFAULT_CVS_THRESHOLD}"
        )

    def test_implausible_combination_cvs_below_threshold(self, default_cvs, canonical_X, implausible_X):
        """80% 3P% AND 30 turnovers → CVS < 0.92 (spec requirement)."""
        cvs = default_cvs.compute(canonical_X, implausible_X)
        assert cvs < DEFAULT_CVS_THRESHOLD, (
            f"Implausible vector (fg3_pct=0.80, tov_rate=0.43) got CVS={cvs:.4f} ≥ {DEFAULT_CVS_THRESHOLD}"
        )

    def test_implausible_not_valid(self, default_cvs, canonical_X, implausible_X):
        assert not default_cvs.is_valid(canonical_X, implausible_X)

    def test_cvs_in_unit_interval(self, default_cvs, canonical_X, implausible_X):
        for X in [canonical_X, implausible_X]:
            cvs = default_cvs.compute(canonical_X, X)
            assert 0.0 <= cvs <= 1.0

    def test_hot_shooting_small_valid(self, default_cvs, canonical_X):
        """Hot shooting +0.10 on FG3% stays within bounds → CVS ≥ 0.92."""
        X_p = canonical_X.copy()
        X_p[FEAT_IDX["fg3_pct"]] += 0.10   # 0.36 → 0.46 (within [0.20, 0.50])
        cvs = default_cvs.compute(canonical_X, X_p)
        assert cvs >= DEFAULT_CVS_THRESHOLD, (
            f"Small hot-shooting perturbation got CVS={cvs:.4f} < threshold"
        )

    def test_extreme_fg3_pct_fails_cvs(self, default_cvs, canonical_X):
        """FG3% = 0.75 (historically impossible) → CVS < 0.92."""
        X_p = canonical_X.copy()
        X_p[FEAT_IDX["fg3_pct"]] = 0.75
        cvs = default_cvs.compute(canonical_X, X_p)
        assert cvs < DEFAULT_CVS_THRESHOLD

    def test_extreme_tov_rate_fails_cvs(self, default_cvs, canonical_X):
        """TOV rate = 0.45 → CVS < 0.92."""
        X_p = canonical_X.copy()
        X_p[FEAT_IDX["tov_rate"]] = 0.45
        cvs = default_cvs.compute(canonical_X, X_p)
        assert cvs < DEFAULT_CVS_THRESHOLD

    def test_batch_compute_length(self, default_cvs, canonical_X):
        perturbed = [canonical_X + np.random.randn(FEATURE_DIM) * 0.001 for _ in range(5)]
        scores = default_cvs.batch_compute(canonical_X, perturbed)
        assert len(scores) == 5

    def test_batch_compute_values_match_individual(self, default_cvs, canonical_X):
        rng = np.random.default_rng(77)
        perturbed = [canonical_X + rng.standard_normal(FEATURE_DIM) * 0.001 for _ in range(5)]
        batch = default_cvs.batch_compute(canonical_X, perturbed)
        individual = [default_cvs.compute(canonical_X, xp) for xp in perturbed]
        np.testing.assert_allclose(batch, individual)

    def test_custom_weights_sum_to_one(self):
        with pytest.raises(ValueError, match="weights must sum to 1.0"):
            CVSComputer(weights=(0.5, 0.3, 0.3))

    def test_from_training_data(self, training_dist_from_data, canonical_X):
        cvs = CVSComputer(training_dist=training_dist_from_data)
        score = cvs.compute(canonical_X, canonical_X)
        assert 0.0 <= score <= 1.0

    def test_bounds_score_all_in_bounds(self, default_cvs, canonical_X):
        score = default_cvs._bounds_score(canonical_X)
        assert score == pytest.approx(1.0)

    def test_bounds_score_one_out(self, default_cvs, canonical_X):
        X_p = canonical_X.copy()
        X_p[FEAT_IDX["fg3_pct"]] = 0.80  # Out of bounds
        score = default_cvs._bounds_score(X_p)
        expected = (FEATURE_DIM - 1) / FEATURE_DIM
        assert score == pytest.approx(expected)

    def test_distribution_score_in_bounds(self, default_cvs, canonical_X):
        """Canonical vector: z-scores near 0 → distribution score ≈ 1.0."""
        score = default_cvs._distribution_score(canonical_X)
        assert score == pytest.approx(1.0)

    def test_distribution_score_extreme_penalised(self, default_cvs, canonical_X):
        X_p = canonical_X.copy()
        X_p[FEAT_IDX["fg3_pct"]] = 0.80   # z ≈ 9 → huge penalty
        score = default_cvs._distribution_score(X_p)
        assert score < 0.5

    def test_contradiction_score_valid(self, default_cvs, canonical_X):
        """No contradictions in canonical vector."""
        score = default_cvs._contradiction_score(canonical_X)
        assert score == pytest.approx(1.0)

    def test_contradiction_score_extreme_fg3(self, default_cvs, canonical_X):
        X_p = canonical_X.copy()
        X_p[FEAT_IDX["fg3_pct"]] = 0.60   # triggers "impossible FG3%" contradiction
        score = default_cvs._contradiction_score(X_p)
        assert score < 1.0

    def test_repr(self, default_cvs):
        assert "CVSComputer" in repr(default_cvs)
        assert "threshold" in repr(default_cvs)

    def test_rest_advantage_perturbation_valid_cvs(self, default_cvs, canonical_X):
        """Rest advantage perturbation stays within historical bounds → CVS ≥ 0.92."""
        from ncaa_models.counterfactual import _apply_rest_advantage
        X_p = _apply_rest_advantage(canonical_X, 0.75, FEAT_IDX)
        cvs = default_cvs.compute(canonical_X, X_p)
        assert cvs >= DEFAULT_CVS_THRESHOLD, (
            f"Rest advantage (sev=0.75) got CVS={cvs:.4f} < threshold"
        )

    def test_injury_perturbation_valid_cvs(self, default_cvs, canonical_X):
        """Moderate injury perturbation is plausible → CVS ≥ 0.92."""
        from ncaa_models.counterfactual import _apply_injury
        X_p = _apply_injury(canonical_X, 0.50, FEAT_IDX)
        cvs = default_cvs.compute(canonical_X, X_p)
        assert cvs >= DEFAULT_CVS_THRESHOLD, (
            f"Injury (sev=0.50) got CVS={cvs:.4f} < threshold"
        )
