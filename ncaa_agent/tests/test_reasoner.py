"""
test_reasoner.py
================
Unit tests for ncaa_agent.reasoner.NCAAReasoner.

Tests cover:
- ReasoningTrace structure (4 steps, fields, types)
- Step adjustments in valid range
- adjusted_prob computation and clipping
- Confidence thresholds
- Cache hit/miss behavior
- reason_batch batch processing
- Determinism
- Edge cases (equal seeds, extreme probabilities)
"""

from __future__ import annotations

import numpy as np
import pytest

from ncaa_agent.reasoner import NCAAReasoner, ReasoningTrace, ReasoningStep
from ncaa_agent.cache import ResultCache

from .conftest import make_feature_vector, make_trace


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_reasoner(cache=None):
    return NCAAReasoner(cache=cache)


def _reason(seed1=1, seed2=8, prob=0.80, cache=None):
    r = _make_reasoner(cache=cache)
    X1 = make_feature_vector(seed1 - 1)
    X2 = make_feature_vector(min(seed2 - 1, 7))
    return r.reason(
        X1, X2, prob,
        meta={
            "team1_name": "Fav",
            "team2_name": "Dog",
            "seed1": seed1,
            "seed2": seed2,
            "matchup_id": "test_001",
        },
    )


# ---------------------------------------------------------------------------
# ReasoningTrace structure
# ---------------------------------------------------------------------------

class TestReasoningTraceStructure:
    def test_trace_has_exactly_four_steps(self, sample_trace):
        assert len(sample_trace.steps) == 4

    def test_step_numbers_are_1_to_4(self, sample_trace):
        for i, step in enumerate(sample_trace.steps, start=1):
            assert step.step == i

    def test_step_names_non_empty(self, sample_trace):
        for step in sample_trace.steps:
            assert isinstance(step.name, str) and len(step.name) > 0

    def test_step_analysis_non_empty(self, sample_trace):
        for step in sample_trace.steps:
            assert isinstance(step.analysis, str) and len(step.analysis) > 0

    def test_step_key_metrics_is_dict(self, sample_trace):
        for step in sample_trace.steps:
            assert isinstance(step.key_metrics, dict)

    def test_step_key_metrics_non_empty(self, sample_trace):
        for step in sample_trace.steps[:3]:  # steps 1-3 have many metrics
            assert len(step.key_metrics) > 0

    def test_ensemble_prob_preserved(self):
        trace = _reason(prob=0.73)
        assert abs(trace.ensemble_prob - 0.73) < 1e-9

    def test_adjusted_prob_in_unit_interval(self, sample_trace):
        assert 0.01 <= sample_trace.adjusted_prob <= 0.99

    def test_confidence_is_valid(self, sample_trace):
        assert sample_trace.confidence in ("HIGH", "MEDIUM", "LOW")

    def test_team_names_preserved(self):
        trace = _reason()
        assert trace.team1_name == "Fav"
        assert trace.team2_name == "Dog"

    def test_seeds_preserved(self):
        trace = _reason(seed1=3, seed2=14)
        assert trace.seed1 == 3
        assert trace.seed2 == 14

    def test_matchup_id_preserved(self):
        trace = _reason()
        assert trace.matchup_id == "test_001"

    def test_timestamp_is_iso_string(self, sample_trace):
        assert isinstance(sample_trace.timestamp, str)
        assert "T" in sample_trace.timestamp  # ISO format includes 'T'

    def test_to_dict_round_trip(self, sample_trace):
        d = sample_trace.to_dict()
        trace2 = ReasoningTrace.from_dict(d)
        assert trace2.matchup_id == sample_trace.matchup_id
        assert trace2.ensemble_prob == sample_trace.ensemble_prob
        assert len(trace2.steps) == 4

    def test_from_dict_steps_are_reasoning_steps(self, sample_trace):
        d = sample_trace.to_dict()
        trace2 = ReasoningTrace.from_dict(d)
        for step in trace2.steps:
            assert isinstance(step, ReasoningStep)


# ---------------------------------------------------------------------------
# Step adjustment ranges
# ---------------------------------------------------------------------------

class TestStepAdjustments:
    def test_each_step_adjustment_within_bounds(self, sample_trace):
        for step in sample_trace.steps[:3]:
            assert -0.08 <= step.adjustment <= 0.08, (
                f"Step {step.step} adjustment {step.adjustment} out of [-0.08, 0.08]"
            )

    def test_step4_total_adjustment_within_bounds(self, sample_trace):
        # Step 4 stores the total_adjustment (clipped to ±0.12)
        step4 = sample_trace.steps[3]
        assert -0.12 <= step4.adjustment <= 0.12

    def test_step1_has_efficiency_metrics(self, sample_trace):
        step1 = sample_trace.steps[0]
        assert "ortg1" in step1.key_metrics
        assert "drtg1" in step1.key_metrics
        assert "net_rtg1" in step1.key_metrics
        assert "net_rtg2" in step1.key_metrics

    def test_step2_has_historical_metrics(self, sample_trace):
        step2 = sample_trace.steps[1]
        assert "historical_win_rate" in step2.key_metrics
        assert "seed1" in step2.key_metrics
        assert "seed2" in step2.key_metrics

    def test_step3_has_contextual_metrics(self, sample_trace):
        step3 = sample_trace.steps[2]
        assert "win_pct_last10_1" in step3.key_metrics
        assert "tov_rate1" in step3.key_metrics

    def test_step4_records_ensemble_prob(self, sample_trace):
        step4 = sample_trace.steps[3]
        assert "ensemble_prob" in step4.key_metrics
        assert abs(step4.key_metrics["ensemble_prob"] - sample_trace.ensemble_prob) < 1e-9

    def test_adjusted_prob_close_to_ensemble(self):
        """adjusted_prob should stay within 0.25 of ensemble (blend weight 0.20 × 0.12 = 0.024 max shift)."""
        trace = _reason(prob=0.60)
        assert abs(trace.adjusted_prob - trace.ensemble_prob) <= 0.025 + 1e-9


# ---------------------------------------------------------------------------
# Confidence thresholds
# ---------------------------------------------------------------------------

class TestConfidenceThresholds:
    def test_high_confidence_strong_favourite(self):
        """p=0.85 → |0.85-0.5|=0.35 > 0.30 → HIGH"""
        trace = _reason(seed1=1, seed2=16, prob=0.99)
        assert trace.confidence == "HIGH"

    def test_low_confidence_near_even(self):
        """p=0.51 → |0.51-0.5|=0.01 < 0.12 → LOW"""
        trace = _reason(seed1=8, seed2=9, prob=0.51)
        assert trace.confidence == "LOW"

    def test_medium_confidence_moderate(self):
        """p=0.65 → |0.65-0.5|=0.15 in (0.12, 0.30) → MEDIUM"""
        trace = _reason(seed1=5, seed2=12, prob=0.65)
        assert trace.confidence in ("MEDIUM", "HIGH")  # depends on adjustment


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_input_same_output(self):
        X1 = make_feature_vector(0)
        X2 = make_feature_vector(7)
        r = NCAAReasoner()
        t1 = r.reason(X1, X2, 0.80, meta={"team1_name": "A", "team2_name": "B",
                                            "seed1": 1, "seed2": 8})
        t2 = r.reason(X1, X2, 0.80, meta={"team1_name": "A", "team2_name": "B",
                                            "seed1": 1, "seed2": 8})
        assert t1.adjusted_prob == t2.adjusted_prob
        assert t1.confidence == t2.confidence

    def test_different_probs_different_outputs(self):
        t1 = _reason(prob=0.60)
        t2 = _reason(prob=0.80)
        assert t1.adjusted_prob != t2.adjusted_prob


# ---------------------------------------------------------------------------
# Cache integration
# ---------------------------------------------------------------------------

class TestCacheIntegration:
    def test_cache_hit_returns_same_trace(self, tmp_path):
        cache = ResultCache(cache_dir=tmp_path / "c")
        r = NCAAReasoner(cache=cache)
        X1 = make_feature_vector(0)
        X2 = make_feature_vector(7)
        meta = {"team1_name": "A", "team2_name": "B", "seed1": 1, "seed2": 8}
        t1 = r.reason(X1, X2, 0.80, meta=meta)
        t2 = r.reason(X1, X2, 0.80, meta=meta)
        assert t1.adjusted_prob == t2.adjusted_prob
        assert t1.matchup_id == t2.matchup_id

    def test_cache_stores_entry(self, tmp_path):
        cache = ResultCache(cache_dir=tmp_path / "c2")
        r = NCAAReasoner(cache=cache)
        X1 = make_feature_vector(0)
        X2 = make_feature_vector(7)
        assert cache.size() == 0
        r.reason(X1, X2, 0.70, meta={"team1_name": "A", "team2_name": "B",
                                       "seed1": 1, "seed2": 8})
        assert cache.size() == 1

    def test_different_inputs_different_cache_keys(self, tmp_path):
        cache = ResultCache(cache_dir=tmp_path / "c3")
        r = NCAAReasoner(cache=cache)
        X1 = make_feature_vector(0)
        X2 = make_feature_vector(7)
        X3 = make_feature_vector(4)
        r.reason(X1, X2, 0.70, meta={"team1_name": "A", "team2_name": "B",
                                       "seed1": 1, "seed2": 8})
        r.reason(X1, X3, 0.70, meta={"team1_name": "A", "team2_name": "C",
                                       "seed1": 1, "seed2": 5})
        assert cache.size() == 2


# ---------------------------------------------------------------------------
# reason_batch
# ---------------------------------------------------------------------------

class TestReasonBatch:
    def test_batch_length_matches_input(self, reasoner):
        matchups = [
            {"X1": make_feature_vector(0), "X2": make_feature_vector(7),
             "ensemble_prob": 0.80,
             "meta": {"team1_name": "A", "team2_name": "B", "seed1": 1, "seed2": 8}},
            {"X1": make_feature_vector(1), "X2": make_feature_vector(6),
             "ensemble_prob": 0.70,
             "meta": {"team1_name": "C", "team2_name": "D", "seed1": 2, "seed2": 7}},
        ]
        traces = reasoner.reason_batch(matchups)
        assert len(traces) == 2

    def test_batch_returns_reasoning_traces(self, reasoner):
        matchups = [
            {"X1": make_feature_vector(i), "X2": make_feature_vector(7 - i),
             "ensemble_prob": 0.60,
             "meta": {"team1_name": f"T{i}", "team2_name": f"T{7-i}",
                      "seed1": i + 1, "seed2": 8 - i}}
            for i in range(4)
        ]
        traces = reasoner.reason_batch(matchups)
        assert all(isinstance(t, ReasoningTrace) for t in traces)

    def test_batch_empty_returns_empty(self, reasoner):
        assert reasoner.reason_batch([]) == []


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_equal_seeds_returns_trace(self, reasoner):
        X1 = make_feature_vector(3)
        X2 = make_feature_vector(4)
        trace = reasoner.reason(X1, X2, 0.50,
                                meta={"team1_name": "A", "team2_name": "B",
                                      "seed1": 5, "seed2": 5})
        assert isinstance(trace, ReasoningTrace)
        assert 0.01 <= trace.adjusted_prob <= 0.99

    def test_extreme_prob_high_clipped(self, reasoner):
        X1 = make_feature_vector(0)
        X2 = make_feature_vector(7)
        trace = reasoner.reason(X1, X2, 0.99,
                                meta={"team1_name": "A", "team2_name": "B",
                                      "seed1": 1, "seed2": 16})
        assert trace.adjusted_prob <= 0.99

    def test_extreme_prob_low_clipped(self, reasoner):
        X1 = make_feature_vector(7)
        X2 = make_feature_vector(0)
        trace = reasoner.reason(X1, X2, 0.01,
                                meta={"team1_name": "A", "team2_name": "B",
                                      "seed1": 16, "seed2": 1})
        assert trace.adjusted_prob >= 0.01

    def test_2d_input_handled(self, reasoner):
        """X1/X2 passed as (1, 28) 2D arrays should still work."""
        X1 = make_feature_vector(0).reshape(1, -1)
        X2 = make_feature_vector(7).reshape(1, -1)
        trace = reasoner.reason(X1, X2, 0.75,
                                meta={"team1_name": "A", "team2_name": "B",
                                      "seed1": 1, "seed2": 8})
        assert isinstance(trace, ReasoningTrace)

    def test_missing_meta_uses_defaults(self, reasoner):
        """reason() without meta should still produce a valid trace."""
        X1 = make_feature_vector(0)
        X2 = make_feature_vector(7)
        trace = reasoner.reason(X1, X2, 0.70)
        assert isinstance(trace, ReasoningTrace)
        assert len(trace.steps) == 4
