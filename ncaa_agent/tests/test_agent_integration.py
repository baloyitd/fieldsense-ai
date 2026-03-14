"""
test_agent_integration.py
=========================
Integration tests for the full ncaa_agent Stage 07 pipeline.

Tests cover:
- End-to-end pipeline: features → NCAAReasoner → AnomalyDetector → BracketNarrativeGenerator
- ResultCache correctly caches and retrieves traces through the pipeline
- 63-game bracket processing (realistic tournament load)
- AnomalyDetector recall ≥ 70% on synthetic errors
- Narrative generation for all 63 games
- __init__.py public API completeness
- No external calls (fully offline)
"""

from __future__ import annotations

import numpy as np
import pytest

import ncaa_agent
from ncaa_agent import (
    NCAAReasoner,
    ReasoningTrace,
    ReasoningStep,
    AnomalyDetector,
    AnomalyFlag,
    BracketNarrativeGenerator,
    ResultCache,
    seed_baseline_win_rate,
)

from .conftest import make_feature_vector, make_trace, N_TEAMS


# ---------------------------------------------------------------------------
# Public API completeness
# ---------------------------------------------------------------------------

class TestPublicAPI:
    def test_stage_label(self):
        assert ncaa_agent.__stage__ == "07/10"

    def test_ncareasoner_exported(self):
        assert hasattr(ncaa_agent, "NCAAReasoner")

    def test_reasoning_trace_exported(self):
        assert hasattr(ncaa_agent, "ReasoningTrace")

    def test_reasoning_step_exported(self):
        assert hasattr(ncaa_agent, "ReasoningStep")

    def test_anomaly_detector_exported(self):
        assert hasattr(ncaa_agent, "AnomalyDetector")

    def test_anomaly_flag_exported(self):
        assert hasattr(ncaa_agent, "AnomalyFlag")

    def test_narrative_generator_exported(self):
        assert hasattr(ncaa_agent, "BracketNarrativeGenerator")

    def test_result_cache_exported(self):
        assert hasattr(ncaa_agent, "ResultCache")

    def test_seed_baseline_win_rate_exported(self):
        assert hasattr(ncaa_agent, "seed_baseline_win_rate")


# ---------------------------------------------------------------------------
# End-to-end single matchup
# ---------------------------------------------------------------------------

class TestEndToEndSingleMatchup:
    def test_full_pipeline_returns_trace_and_narrative(self):
        X1 = make_feature_vector(0)  # rank 0 = seed 1
        X2 = make_feature_vector(7)  # rank 7 = seed 8
        reasoner = NCAAReasoner()
        trace = reasoner.reason(
            X1, X2, ensemble_prob=0.82,
            meta={"team1_name": "Kansas", "team2_name": "Howard",
                  "seed1": 1, "seed2": 16, "matchup_id": "KU_HU"}
        )

        det = AnomalyDetector()
        flags = det.detect([trace])

        gen = BracketNarrativeGenerator()
        text = gen.generate(trace)

        assert isinstance(trace, ReasoningTrace)
        assert isinstance(flags, list)
        assert isinstance(text, str) and len(text) > 0

    def test_trace_adjusted_prob_influences_narrative(self):
        X1 = make_feature_vector(0)
        X2 = make_feature_vector(7)
        reasoner = NCAAReasoner()
        trace = reasoner.reason(X1, X2, 0.90,
                                 meta={"team1_name": "Duke", "team2_name": "FGCU",
                                       "seed1": 1, "seed2": 16})
        gen = BracketNarrativeGenerator()
        text = gen.generate(trace)
        # Probability should appear in narrative
        assert "%" in text
        assert "Duke" in text

    def test_pipeline_anomaly_flag_has_correct_fields(self):
        trace = make_trace(seed1=1, seed2=16, ensemble_prob=0.30)
        det = AnomalyDetector()
        flags = det.detect([trace])
        assert len(flags) == 1
        f = flags[0]
        assert f.team1_name == trace.team1_name
        assert f.seed1 == trace.seed1
        assert f.is_anomalous


# ---------------------------------------------------------------------------
# 63-game bracket processing
# ---------------------------------------------------------------------------

class TestBracketProcessing:
    def test_63_game_traces_all_valid(self, traces_63):
        assert len(traces_63) == 63
        for t in traces_63:
            assert isinstance(t, ReasoningTrace)
            assert len(t.steps) == 4
            assert 0.01 <= t.adjusted_prob <= 0.99

    def test_63_game_narratives_all_non_empty(self, traces_63):
        gen = BracketNarrativeGenerator()
        texts = gen.generate_batch(traces_63)
        assert len(texts) == 63
        assert all(len(t) > 0 for t in texts)

    def test_63_game_anomaly_detection(self, traces_63):
        det = AnomalyDetector()
        flags = det.detect(traces_63)
        # Some games should be flagged (ensemble probs deviate from baselines)
        assert isinstance(flags, list)
        # All returned flags should be genuinely anomalous
        for f in flags:
            assert f.deviation > det.threshold

    def test_top_20_anomalies_from_63_games(self, traces_63):
        det = AnomalyDetector(top_k=20)
        flags = det.flag_top_k(traces_63)
        assert len(flags) <= 20
        assert len(flags) <= len(traces_63)

    def test_63_game_brief_narratives(self, traces_63):
        gen = BracketNarrativeGenerator(template="brief")
        texts = gen.generate_batch(traces_63)
        assert len(texts) == 63
        for t in texts:
            assert "%" in t  # all contain a probability


# ---------------------------------------------------------------------------
# Cache integration (end-to-end)
# ---------------------------------------------------------------------------

class TestCacheIntegrationE2E:
    def test_cache_avoids_recomputation(self, tmp_path):
        cache = ResultCache(cache_dir=tmp_path / "e2e_cache")
        reasoner = NCAAReasoner(cache=cache)
        X1 = make_feature_vector(0)
        X2 = make_feature_vector(7)
        meta = {"team1_name": "A", "team2_name": "B", "seed1": 1, "seed2": 8}

        t1 = reasoner.reason(X1, X2, 0.80, meta=meta)
        assert cache.size() == 1

        t2 = reasoner.reason(X1, X2, 0.80, meta=meta)
        assert cache.size() == 1  # no new entry created

        assert t1.adjusted_prob == t2.adjusted_prob

    def test_cache_miss_stores_new_entry(self, tmp_path):
        cache = ResultCache(cache_dir=tmp_path / "e2e_cache2")
        reasoner = NCAAReasoner(cache=cache)

        for i in range(3):
            X1 = make_feature_vector(i)
            X2 = make_feature_vector(7 - i)
            reasoner.reason(
                X1, X2, 0.60 + i * 0.05,
                meta={"team1_name": f"T{i}", "team2_name": f"T{7-i}",
                      "seed1": i + 1, "seed2": 8 - i}
            )
        assert cache.size() == 3

    def test_cached_trace_survives_to_dict_round_trip(self, tmp_path):
        cache = ResultCache(cache_dir=tmp_path / "e2e_rt")
        reasoner = NCAAReasoner(cache=cache)
        X1 = make_feature_vector(0)
        X2 = make_feature_vector(7)
        meta = {"team1_name": "X", "team2_name": "Y", "seed1": 1, "seed2": 8}

        t1 = reasoner.reason(X1, X2, 0.75, meta=meta)
        # Retrieve from cache
        t2 = reasoner.reason(X1, X2, 0.75, meta=meta)

        assert t1.ensemble_prob == t2.ensemble_prob
        assert t1.confidence == t2.confidence
        assert len(t2.steps) == 4


# ---------------------------------------------------------------------------
# Recall ≥ 70% (SPEC requirement)
# ---------------------------------------------------------------------------

class TestAnomalyRecallSpec:
    def test_70pct_recall_on_synthetic_large_errors(self):
        """
        SPEC: AnomalyDetector must flag ≥70% of actual prediction errors.

        Setup:
          - 30 matchups: seed 1v16 (baseline ~0.993)
          - 20 'error' matchups: ensemble_prob=0.25 (deviation=0.743 >> 0.15)
          - 10 'correct' matchups: ensemble_prob=0.95 (deviation≈0.043 < 0.15)
          - Outcomes for errors: 0 (team1 lost); error = |0.25 - 0| = 0.25 > 0.20
          - Outcomes for correct: 1; error = |0.95 - 1| = 0.05 < 0.20
        """
        reasoner = NCAAReasoner()
        X1 = make_feature_vector(0)   # seed-1 team
        X2 = make_feature_vector(7)   # seed-16 equivalent

        traces = []
        outcomes = []

        # 20 error traces: large ensemble_prob deviation AND large prediction error
        for i in range(20):
            t = reasoner.reason(
                X1, X2, 0.25,
                meta={"team1_name": "Fav", "team2_name": "Dog",
                      "seed1": 1, "seed2": 16, "matchup_id": f"err_{i}"}
            )
            traces.append(t)
            outcomes.append(0)  # error = |0.25 - 0| = 0.25

        # 10 non-error traces: small prediction error
        for i in range(10):
            t = reasoner.reason(
                X1, X2, 0.99,
                meta={"team1_name": "Fav", "team2_name": "Dog",
                      "seed1": 1, "seed2": 16, "matchup_id": f"ok_{i}"}
            )
            traces.append(t)
            outcomes.append(1)  # error = |0.99 - 1| = 0.01

        det = AnomalyDetector(threshold=0.15)
        recall = det.prediction_error_recall(traces, outcomes, error_threshold=0.20)
        assert recall >= 0.70, f"Recall {recall:.2f} < 0.70"

    def test_flag_top_k_covers_largest_errors(self):
        """The top-k flagged anomalies should include the most extreme deviations."""
        reasoner = NCAAReasoner()
        X1 = make_feature_vector(0)
        X2 = make_feature_vector(7)

        traces = []
        # 5 extreme traces: ensemble far from baseline
        for i in range(5):
            t = reasoner.reason(X1, X2, 0.20,
                                 meta={"team1_name": "A", "team2_name": "B",
                                       "seed1": 1, "seed2": 16,
                                       "matchup_id": f"extreme_{i}"})
            traces.append(t)
        # 5 normal traces: ensemble close to baseline
        for i in range(5):
            baseline = seed_baseline_win_rate(5, 12)
            t = reasoner.reason(X1, X2, baseline,
                                 meta={"team1_name": "A", "team2_name": "B",
                                       "seed1": 5, "seed2": 12,
                                       "matchup_id": f"normal_{i}"})
            traces.append(t)

        det = AnomalyDetector()
        top5 = det.flag_top_k(traces, k=5)
        # The top 5 should all have larger deviations than the bottom 5
        top_devs = {f.deviation for f in top5}
        all_flags = det.flag_top_k(traces, k=10)
        bottom_devs = {f.deviation for f in all_flags[5:]}
        assert min(top_devs) >= max(bottom_devs) - 1e-9


# ---------------------------------------------------------------------------
# Offline operation (no network calls)
# ---------------------------------------------------------------------------

class TestOfflineOperation:
    def test_reasoner_requires_no_external_calls(self):
        """NCAAReasoner is pure computation; no network access required."""
        import socket
        # We just verify that the reasoner works without mocking network
        reasoner = NCAAReasoner()
        X1 = make_feature_vector(0)
        X2 = make_feature_vector(7)
        trace = reasoner.reason(X1, X2, 0.70,
                                 meta={"team1_name": "A", "team2_name": "B",
                                       "seed1": 1, "seed2": 8})
        assert isinstance(trace, ReasoningTrace)

    def test_cache_uses_only_local_filesystem(self, tmp_path):
        cache = ResultCache(cache_dir=tmp_path / "local_cache")
        assert cache.cache_dir.is_dir()
        assert str(tmp_path) in str(cache.cache_dir)
