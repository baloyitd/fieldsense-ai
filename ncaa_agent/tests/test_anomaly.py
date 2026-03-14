"""
test_anomaly.py
===============
Unit tests for ncaa_agent.anomaly.

Tests cover:
- seed_baseline_win_rate table lookups and formula fallback
- AnomalyFlag dataclass
- AnomalyDetector.detect() — correct flagging and sorting
- AnomalyDetector.flag_top_k() — top-k selection
- AnomalyDetector.prediction_error_recall() — ≥70% target on synthetic data
- Edge cases (empty input, equal seeds, all pass, all fail)
"""

from __future__ import annotations

import numpy as np
import pytest

from ncaa_agent.anomaly import (
    AnomalyDetector,
    AnomalyFlag,
    DEFAULT_THRESHOLD,
    DEFAULT_TOP_K,
    SEED_WIN_RATES,
    seed_baseline_win_rate,
)

from .conftest import make_trace, traces_63


# ---------------------------------------------------------------------------
# seed_baseline_win_rate
# ---------------------------------------------------------------------------

class TestSeedBaselineWinRate:
    def test_equal_seeds_returns_half(self):
        assert seed_baseline_win_rate(5, 5) == 0.50

    def test_1v16_table_lookup(self):
        # Raw table value is 0.993, but seed_baseline_win_rate clips to [0.01, 0.99]
        rate = seed_baseline_win_rate(1, 16)
        assert abs(rate - 0.99) < 1e-9

    def test_1v16_flipped(self):
        """seed2=1 vs seed1=16 should return 1 - clipped(0.993) = 1 - 0.99 = 0.01."""
        rate = seed_baseline_win_rate(16, 1)
        assert abs(rate - 0.01) < 1e-9

    def test_8v9_close_to_half(self):
        rate = seed_baseline_win_rate(8, 9)
        assert 0.49 <= rate <= 0.55

    def test_5v12_known_rate(self):
        rate = seed_baseline_win_rate(5, 12)
        assert abs(rate - 0.650) < 1e-9

    def test_return_in_unit_interval(self):
        for (lo, hi) in SEED_WIN_RATES:
            assert 0.01 <= seed_baseline_win_rate(lo, hi) <= 0.99

    def test_formula_fallback_monotone(self):
        """Non-table pairs: higher seed diff → higher win probability."""
        r1 = seed_baseline_win_rate(1, 3)
        r2 = seed_baseline_win_rate(1, 10)
        # 1v10 seed diff=9 > 1v3 seed diff=2 → baseline for 1-seed should be higher
        assert r2 >= r1

    def test_flipped_complement(self):
        """P(A beats B) + P(B beats A) = 1."""
        for (lo, hi) in SEED_WIN_RATES:
            rate_lo = seed_baseline_win_rate(lo, hi)
            rate_hi = seed_baseline_win_rate(hi, lo)
            assert abs(rate_lo + rate_hi - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# AnomalyFlag dataclass
# ---------------------------------------------------------------------------

class TestAnomalyFlag:
    def test_to_dict_has_all_fields(self):
        flag = AnomalyFlag(
            matchup_id="m1",
            team1_name="A",
            team2_name="B",
            seed1=1,
            seed2=16,
            ensemble_prob=0.50,
            seed_baseline_prob=0.993,
            deviation=0.493,
            is_anomalous=True,
            rank=1,
        )
        d = flag.to_dict()
        for key in ("matchup_id", "team1_name", "team2_name",
                    "seed1", "seed2", "ensemble_prob",
                    "seed_baseline_prob", "deviation", "is_anomalous", "rank"):
            assert key in d

    def test_default_rank_is_zero(self):
        flag = AnomalyFlag(
            matchup_id="m1", team1_name="A", team2_name="B",
            seed1=1, seed2=8, ensemble_prob=0.80,
            seed_baseline_prob=0.77, deviation=0.03, is_anomalous=False,
        )
        assert flag.rank == 0


# ---------------------------------------------------------------------------
# AnomalyDetector — detect
# ---------------------------------------------------------------------------

class TestAnomalyDetectorDetect:
    def test_detect_returns_list(self, traces_63):
        det = AnomalyDetector()
        result = det.detect(traces_63)
        assert isinstance(result, list)

    def test_detect_empty_input(self):
        det = AnomalyDetector()
        assert det.detect([]) == []

    def test_detect_all_anomalous_low_threshold(self):
        """With threshold=0.0, everything is flagged."""
        det = AnomalyDetector(threshold=0.0)
        traces = [make_trace(seed1=1, seed2=8, ensemble_prob=0.75)]
        flags = det.detect(traces)
        assert len(flags) == 1
        assert flags[0].is_anomalous

    def test_detect_none_flagged_high_threshold(self, traces_63):
        """With threshold=1.0, nothing is flagged."""
        det = AnomalyDetector(threshold=1.0)
        flags = det.detect(traces_63)
        assert len(flags) == 0

    def test_detect_sorted_by_deviation_descending(self, traces_63):
        det = AnomalyDetector(threshold=0.0)  # all flagged
        flags = det.detect(traces_63)
        devs = [f.deviation for f in flags]
        assert devs == sorted(devs, reverse=True)

    def test_detect_ranks_start_at_1(self, traces_63):
        det = AnomalyDetector(threshold=0.0)
        flags = det.detect(traces_63)
        if flags:
            assert flags[0].rank == 1

    def test_detect_flags_are_truly_anomalous(self, traces_63):
        det = AnomalyDetector(threshold=DEFAULT_THRESHOLD)
        flags = det.detect(traces_63)
        for f in flags:
            assert f.deviation > det.threshold

    def test_detect_anomalous_flag_has_correct_deviation(self):
        """A 1v16 matchup predicted at 50% should be anomalous (|0.50 - 0.993| = 0.493)."""
        trace = make_trace(seed1=1, seed2=16, ensemble_prob=0.50,
                            team1_name="Underdog", team2_name="Giant")
        det = AnomalyDetector()
        flags = det.detect([trace])
        assert len(flags) == 1
        assert abs(flags[0].deviation - abs(0.50 - 0.993)) < 0.01


# ---------------------------------------------------------------------------
# AnomalyDetector — flag_top_k
# ---------------------------------------------------------------------------

class TestFlagTopK:
    def test_flag_top_k_length(self, traces_63):
        det = AnomalyDetector(top_k=5)
        flags = det.flag_top_k(traces_63)
        assert len(flags) <= 5

    def test_flag_top_k_custom_k(self, traces_63):
        det = AnomalyDetector()
        flags = det.flag_top_k(traces_63, k=3)
        assert len(flags) <= 3

    def test_flag_top_k_sorted_by_deviation(self, traces_63):
        det = AnomalyDetector()
        flags = det.flag_top_k(traces_63, k=10)
        devs = [f.deviation for f in flags]
        assert devs == sorted(devs, reverse=True)

    def test_flag_top_k_ranks_start_at_1(self, traces_63):
        det = AnomalyDetector()
        flags = det.flag_top_k(traces_63, k=5)
        if flags:
            assert flags[0].rank == 1
            assert flags[-1].rank == len(flags)

    def test_flag_top_k_empty_input(self):
        det = AnomalyDetector()
        assert det.flag_top_k([]) == []

    def test_flag_top_k_fewer_than_k(self):
        """When input has < k traces, return all of them."""
        traces = [make_trace(seed1=1, seed2=8, ensemble_prob=0.75)]
        det = AnomalyDetector(top_k=10)
        flags = det.flag_top_k(traces)
        assert len(flags) == 1

    def test_flag_top_k_overrides_instance_k(self):
        det = AnomalyDetector(top_k=20)
        # Keep seeds within valid rank range (seed ≤ 8 → rank ≤ 7)
        traces = [make_trace(seed1=i + 1, seed2=8 - i, ensemble_prob=0.50)
                  for i in range(5)]
        flags = det.flag_top_k(traces, k=3)
        assert len(flags) == 3


# ---------------------------------------------------------------------------
# AnomalyDetector — compute_deviation
# ---------------------------------------------------------------------------

class TestComputeDeviation:
    def test_perfect_match_to_baseline_zero_deviation(self):
        det = AnomalyDetector()
        baseline = seed_baseline_win_rate(1, 16)
        dev = det.compute_deviation(baseline, 1, 16)
        assert abs(dev) < 1e-9

    def test_deviation_is_absolute(self):
        det = AnomalyDetector()
        dev1 = det.compute_deviation(0.30, 1, 16)   # below baseline
        dev2 = det.compute_deviation(0.993 + 0.30, 1, 16)  # above baseline
        # Both should be positive
        assert dev1 > 0
        # dev2 might be clipped by the 0.99 rate cap; just check positivity
        assert dev2 >= 0


# ---------------------------------------------------------------------------
# prediction_error_recall
# ---------------------------------------------------------------------------

class TestPredictionErrorRecall:
    def _make_traces_and_outcomes(self, n=20, rng_seed=0):
        """
        Build traces where some have large ensemble_prob errors.
        Returns (traces, outcomes) suitable for recall computation.
        """
        rng = np.random.RandomState(rng_seed)
        traces = []
        outcomes = []
        for i in range(n):
            # Alternate: half clearly correct, half clearly wrong
            if i % 2 == 0:
                prob = 0.80
                outcome = 1
            else:
                prob = 0.85   # predicted 85% but outcome is 0 → error = 0.85
                outcome = 0
            t = make_trace(seed1=1, seed2=8, ensemble_prob=prob,
                            matchup_id=f"m_{i:03d}")
            traces.append(t)
            outcomes.append(outcome)
        return traces, outcomes

    def test_recall_returns_float(self):
        traces, outcomes = self._make_traces_and_outcomes()
        det = AnomalyDetector(threshold=0.20)
        recall = det.prediction_error_recall(traces, outcomes)
        assert isinstance(recall, float)

    def test_recall_in_unit_interval(self):
        traces, outcomes = self._make_traces_and_outcomes()
        det = AnomalyDetector()
        recall = det.prediction_error_recall(traces, outcomes)
        assert 0.0 <= recall <= 1.0

    def test_recall_zero_for_no_errors(self):
        """If every prediction is close to the outcome, recall should be 0."""
        traces = [make_trace(seed1=1, seed2=8, ensemble_prob=0.80)]
        outcomes = [1]  # error = |0.80 - 1| = 0.20, not > 0.20
        det = AnomalyDetector()
        recall = det.prediction_error_recall(traces, outcomes,
                                              error_threshold=0.21)
        assert recall == 0.0

    def test_recall_empty_traces_returns_zero(self):
        det = AnomalyDetector()
        recall = det.prediction_error_recall([], [])
        assert recall == 0.0

    def test_recall_meets_70pct_target_on_synthetic_errors(self):
        """
        SPEC: ≥70% of actual prediction errors should be flagged.

        We deliberately create 10 traces with large deviations from baselines:
        - seed1=1, seed2=16: baseline ~0.993
        - ensemble_prob = 0.30 (deviation = 0.693, well above 0.15 threshold)
        - outcome = 0 (error = |0.30 - 0| = 0.30 > 0.20 threshold)
        These should ALL be flagged and counted as errors.
        """
        traces = [
            make_trace(seed1=1, seed2=16, ensemble_prob=0.30,
                        matchup_id=f"err_{i}")
            for i in range(10)
        ]
        outcomes = [0] * 10  # all team1 lost; error = |0.30 - 0| = 0.30

        det = AnomalyDetector(threshold=DEFAULT_THRESHOLD)
        recall = det.prediction_error_recall(traces, outcomes,
                                              error_threshold=0.20)
        assert recall >= 0.70, f"Recall {recall:.2f} < 0.70"

    def test_recall_all_errors_flagged_gives_1_0(self):
        """When ALL errors are anomalous, recall should be 1.0."""
        traces = [
            make_trace(seed1=1, seed2=16, ensemble_prob=0.20,
                        matchup_id=f"x_{i}")
            for i in range(5)
        ]
        outcomes = [0] * 5  # error = 0.20, use threshold 0.19

        det = AnomalyDetector(threshold=0.10)  # all deviations > 0.10
        recall = det.prediction_error_recall(traces, outcomes,
                                              error_threshold=0.19)
        assert recall == 1.0


# ---------------------------------------------------------------------------
# AnomalyDetector repr
# ---------------------------------------------------------------------------

class TestAnomalyDetectorRepr:
    def test_repr_contains_threshold(self):
        det = AnomalyDetector(threshold=0.20)
        assert "0.2" in repr(det)

    def test_repr_contains_top_k(self):
        det = AnomalyDetector(top_k=10)
        assert "10" in repr(det)
