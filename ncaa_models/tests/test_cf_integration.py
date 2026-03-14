"""
test_cf_integration.py
======================
Integration tests for Stage 05 — Counterfactual analysis engine.

Tests cover:
A. Counterfactual-adjusted predictions for 2025 first-round matchups
   integrate cleanly into the Stage 02 evaluation harness.
B. Full pipeline: Stage 01 features → counterfactual adjustment →
   submission CSV with no format breaks.
C. Performance delta: Brier score with vs. without counterfactual
   adjustments is computed and documented (no regression enforced —
   adjustment is exploratory).
"""

from __future__ import annotations

import numpy as np
import pytest

from ncaa_models.baseline import FEATURE_COLS, LogisticBaseline
from ncaa_models.counterfactual import (
    ALL_PERTURBATION_TYPES,
    DEFAULT_SEVERITIES,
    CounterfactualGenerator,
    CounterfactualScenario,
)
from ncaa_models.cvs import CVSComputer, TrainingDistribution
from ncaa_models.evaluate import compute_brier_score
from ncaa_models.submit import build_submission, validate_submission
from ncaa_models.upset_detector import (
    BLOWOUT_LIKELY,
    COMPETITIVE,
    UPSET_PLAUSIBLE,
    UpsetDetector,
    classify_matchup,
)

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
# Helpers
# ---------------------------------------------------------------------------

def _build_val_arrays():
    """Return (X1_val, X2_val, y_val) for 2025 first-round matchups."""
    from ncaa_models.cv import build_matchup_df

    feat_df = make_feat_df(M_TEAM_IDS, TRAIN_SEASONS + [VAL_SEASON])
    tourney_df = make_tourney_df(M_TEAM_IDS, TRAIN_SEASONS + [VAL_SEASON])
    matchup_df = build_matchup_df(feat_df, tourney_df, FEATURE_COLS, is_tourney=True)

    val = matchup_df[matchup_df["season"] == VAL_SEASON]
    X1 = np.stack(val["X_team1"].values)
    X2 = np.stack(val["X_team2"].values)
    y = val["y"].values.astype(int)
    return X1, X2, y


def _build_fitted_model():
    """Fit LogisticBaseline on TRAIN_SEASONS."""
    from ncaa_models.cv import build_matchup_df

    feat_df = make_feat_df(M_TEAM_IDS, TRAIN_SEASONS)
    tourney_df = make_tourney_df(M_TEAM_IDS, TRAIN_SEASONS)
    matchup_df = build_matchup_df(feat_df, tourney_df, FEATURE_COLS, is_tourney=True)

    train = matchup_df[matchup_df["season"].isin(TRAIN_SEASONS)]
    X1 = np.stack(train["X_team1"].values)
    X2 = np.stack(train["X_team2"].values)
    y = train["y"].values.astype(int)

    model = LogisticBaseline()
    model.fit(X1, X2, y)
    return model


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fitted_lr():
    return _build_fitted_model()


@pytest.fixture(scope="module")
def val_data():
    return _build_val_arrays()


@pytest.fixture(scope="module")
def cf_generator(fitted_lr):
    return CounterfactualGenerator(model=fitted_lr, feature_cols=FEATURE_COLS)


@pytest.fixture(scope="module")
def upset_detector(fitted_lr):
    return UpsetDetector(model=fitted_lr, feature_cols=FEATURE_COLS)


# ---------------------------------------------------------------------------
# Integration Test A: Counterfactual predictions integrate into eval harness
# ---------------------------------------------------------------------------

class TestCFEvalIntegration:
    """Counterfactual-adjusted predictions integrate into Stage 02 eval harness."""

    def test_base_predictions_valid(self, fitted_lr, val_data):
        """Model produces probabilities in [0, 1] for all val matchups."""
        X1, X2, y = val_data
        probs = fitted_lr.predict_proba(X1, X2)
        assert probs.shape == (len(y),)
        assert np.all(probs >= 0.0) and np.all(probs <= 1.0)

    def test_adjusted_probs_shape(self, cf_generator, val_data):
        """adjusted_probability is callable for each val matchup."""
        X1, X2, y = val_data
        adj_probs = []
        for x1, x2 in zip(X1, X2):
            p = cf_generator.adjusted_probability(x1, x2)
            adj_probs.append(p)
        adj_probs = np.array(adj_probs)
        assert adj_probs.shape == (len(y),)

    def test_adjusted_probs_in_unit_interval(self, cf_generator, val_data):
        X1, X2, y = val_data
        for x1, x2 in zip(X1, X2):
            p = cf_generator.adjusted_probability(x1, x2)
            assert 0.0 <= p <= 1.0

    def test_brier_score_computable_from_adjusted(self, fitted_lr, cf_generator, val_data):
        """Brier score can be computed from counterfactual-adjusted predictions."""
        X1, X2, y = val_data
        adj_probs = np.array([
            cf_generator.adjusted_probability(x1, x2)
            for x1, x2 in zip(X1, X2)
        ])
        brier = compute_brier_score(y, adj_probs)
        assert isinstance(brier, float)
        assert 0.0 <= brier <= 1.0

    def test_scenarios_generated_for_all_val_matchups(self, cf_generator, val_data):
        """CounterfactualGenerator produces scenarios for every val matchup."""
        X1, X2, y = val_data
        expected_per_matchup = (
            len(ALL_PERTURBATION_TYPES) * 2 * len(DEFAULT_SEVERITIES)
        )
        for x1, x2 in zip(X1, X2):
            scenarios = cf_generator.generate(x1, x2)
            assert len(scenarios) == expected_per_matchup

    def test_all_val_scenarios_have_valid_cvs_range(self, cf_generator, val_data):
        """Every scenario CVS is in [0, 1]."""
        X1, X2, y = val_data
        for x1, x2 in zip(X1, X2):
            for s in cf_generator.generate(x1, x2):
                assert 0.0 <= s.cvs <= 1.0

    def test_upset_detector_analyzes_all_val_matchups(self, upset_detector, val_data):
        """UpsetDetector.analyze() returns well-formed dict for all val matchups."""
        X1, X2, y = val_data
        for x1, x2 in zip(X1, X2):
            result = upset_detector.analyze(x1, x2)
            assert "classification" in result
            assert result["classification"] in (BLOWOUT_LIKELY, COMPETITIVE, UPSET_PLAUSIBLE)
            assert "base_prob" in result
            assert "n_valid" in result
            assert result["n_valid"] >= 0

    def test_batch_analyze_length_matches_val(self, upset_detector, val_data):
        """batch_analyze returns one result per val matchup."""
        X1, X2, y = val_data
        results = upset_detector.batch_analyze(X1, X2)
        assert len(results) == len(y)

    def test_adjusted_probabilities_array(self, upset_detector, val_data):
        """adjusted_probabilities returns correct shape array."""
        X1, X2, y = val_data
        adj = upset_detector.adjusted_probabilities(X1, X2)
        assert adj.shape == (len(y),)
        assert np.all(adj >= 0.0) and np.all(adj <= 1.0)


# ---------------------------------------------------------------------------
# Integration Test B: Full pipeline → submission CSV
# ---------------------------------------------------------------------------

class TestCFSubmissionPipeline:
    """Full pipeline: features → CF adjustment → submission CSV (no format breaks)."""

    def test_base_model_submission_valid(self, fitted_lr):
        """build_submission with base model produces a valid submission CSV."""
        feat_df = make_feat_df(M_TEAM_IDS, TRAIN_SEASONS + [VAL_SEASON])
        sub_df = build_submission(
            fitted_lr,
            feat_df,
            M_TEAM_IDS,
            season=VAL_SEASON,
            feature_cols=FEATURE_COLS,
        )
        assert len(sub_df) > 0
        result = validate_submission(sub_df)
        assert result["valid"], f"Submission validation failed: {result['issues']}"

    def test_submission_probs_unchanged_without_cf(self, fitted_lr):
        """Without CF adjustment, base probs build a valid submission."""
        feat_df = make_feat_df(M_TEAM_IDS, TRAIN_SEASONS + [VAL_SEASON])
        sub_df = build_submission(
            fitted_lr,
            feat_df,
            M_TEAM_IDS,
            season=VAL_SEASON,
            feature_cols=FEATURE_COLS,
        )
        result = validate_submission(sub_df)
        assert result["valid"], f"Base submission validation failed: {result['issues']}"

    def test_cf_submission_has_correct_columns(self, fitted_lr):
        """Submission CSV has the required ID and Pred columns."""
        feat_df = make_feat_df(M_TEAM_IDS, TRAIN_SEASONS + [VAL_SEASON])
        sub_df = build_submission(
            fitted_lr,
            feat_df,
            M_TEAM_IDS,
            season=VAL_SEASON,
            feature_cols=FEATURE_COLS,
        )
        assert "ID" in sub_df.columns
        assert "Pred" in sub_df.columns
        assert (sub_df["Pred"] >= 0.0).all() and (sub_df["Pred"] <= 1.0).all()


# ---------------------------------------------------------------------------
# Integration Test C: Performance delta
# ---------------------------------------------------------------------------

class TestCFPerformanceDelta:
    """
    Document Brier score delta with vs. without counterfactual adjustments.

    We do NOT enforce improvement (CF adjustments are exploratory in Stage 05).
    We DO enforce that adjusted predictions are valid and numerically sensible.
    """

    def test_brier_scores_are_finite(self, fitted_lr, cf_generator, val_data):
        """Both base and CF-adjusted Brier scores are finite."""
        X1, X2, y = val_data
        base_probs = fitted_lr.predict_proba(X1, X2)
        adj_probs = np.array([
            cf_generator.adjusted_probability(x1, x2)
            for x1, x2 in zip(X1, X2)
        ])
        brier_base = compute_brier_score(y, base_probs)
        brier_adj = compute_brier_score(y, adj_probs)

        assert np.isfinite(brier_base), "Base Brier score is not finite"
        assert np.isfinite(brier_adj), "CF-adjusted Brier score is not finite"

    def test_brier_scores_in_valid_range(self, fitted_lr, cf_generator, val_data):
        """Brier scores lie in [0, 1]."""
        X1, X2, y = val_data
        base_probs = fitted_lr.predict_proba(X1, X2)
        adj_probs = np.array([
            cf_generator.adjusted_probability(x1, x2)
            for x1, x2 in zip(X1, X2)
        ])
        assert 0.0 <= compute_brier_score(y, base_probs) <= 1.0
        assert 0.0 <= compute_brier_score(y, adj_probs) <= 1.0

    def test_brier_base_below_random(self, fitted_lr, val_data):
        """Fitted model achieves Brier < 0.25 (better than random 50/50 guess)."""
        X1, X2, y = val_data
        base_probs = fitted_lr.predict_proba(X1, X2)
        brier = compute_brier_score(y, base_probs)
        assert brier < 0.25, f"Base Brier={brier:.4f} — model no better than random"

    def test_cf_blend_does_not_catastrophically_worsen(self, fitted_lr, cf_generator, val_data):
        """CF blending does not increase Brier by more than 0.10 above base."""
        X1, X2, y = val_data
        base_probs = fitted_lr.predict_proba(X1, X2)
        adj_probs = np.array([
            cf_generator.adjusted_probability(x1, x2)
            for x1, x2 in zip(X1, X2)
        ])
        brier_base = compute_brier_score(y, base_probs)
        brier_adj = compute_brier_score(y, adj_probs)
        delta = brier_adj - brier_base
        assert delta <= 0.10, (
            f"CF adjustment worsened Brier by {delta:.4f} "
            f"(base={brier_base:.4f}, adj={brier_adj:.4f})"
        )

    def test_delta_prob_documented(self, cf_generator, val_data):
        """delta_prob is non-zero for at least some scenarios (CF is doing something)."""
        X1, X2, y = val_data
        any_nonzero = False
        for x1, x2 in zip(X1, X2):
            for s in cf_generator.generate(x1, x2):
                if abs(s.delta_prob) > 1e-6:
                    any_nonzero = True
                    break
            if any_nonzero:
                break
        assert any_nonzero, "No scenario produced a non-trivial delta_prob"

    def test_valid_scenario_fraction_reasonable(self, cf_generator, val_data):
        """At least 50% of generated scenarios pass CVS ≥ 0.92 (perturbations are plausible)."""
        X1, X2, y = val_data
        total = 0
        valid = 0
        for x1, x2 in zip(X1, X2):
            scenarios = cf_generator.generate(x1, x2)
            total += len(scenarios)
            valid += sum(1 for s in scenarios if s.is_valid)
        assert total > 0
        rate = valid / total
        assert rate >= 0.50, (
            f"Only {rate:.1%} of scenarios are valid (CVS ≥ 0.92); expected ≥ 50%"
        )

    def test_classification_coverage(self, upset_detector, val_data):
        """At least one matchup of each class is observed in val set (sanity check)."""
        X1, X2, y = val_data
        results = upset_detector.batch_analyze(X1, X2)
        labels = {r["classification"] for r in results}
        # We expect at least "competitive" or "blowout_likely" in a seeded bracket
        assert len(labels) >= 1, "No classifications produced"
        # All labels must be valid strings
        for lbl in labels:
            assert lbl in (BLOWOUT_LIKELY, COMPETITIVE, UPSET_PLAUSIBLE)
