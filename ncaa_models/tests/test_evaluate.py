"""
test_evaluate.py
================
Tests for ncaa_models.evaluate.

Covers
------
1. UNIT TEST 2: Brier score verified against hand-calculated examples.
2. Edge cases: perfect predictions, worst predictions, all-same predictions.
3. calibration_report structure and ECE bounds.
4. per_seed_analysis categories and structure.
5. per_round_analysis structure.
6. JSON output from save_json parameter.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ncaa_models.evaluate import (
    calibration_report,
    compute_brier_score,
    per_round_analysis,
    per_seed_analysis,
)


# ---------------------------------------------------------------------------
# UNIT TEST 2: Brier score formula
# ---------------------------------------------------------------------------

class TestBrierScore:
    """
    UNIT TEST 2: Validate compute_brier_score against hand-calculated examples.

      predict 0.8, actual 1 → (0.8-1)^2 = 0.04
      predict 0.3, actual 0 → (0.3-0)^2 = 0.09
      mean of above = 0.065
    """

    def test_hand_calculated_example(self):
        preds = [0.8, 0.3]
        actuals = [1, 0]
        result = compute_brier_score(preds, actuals)
        assert abs(result - 0.065) < 1e-9, (
            f"Expected Brier=0.065, got {result}"
        )

    def test_single_correct_high_confidence(self):
        # predict 0.8, actual 1 → (0.8-1)^2 = 0.04
        assert abs(compute_brier_score([0.8], [1]) - 0.04) < 1e-9

    def test_single_correct_low_confidence(self):
        # predict 0.3, actual 0 → (0.3-0)^2 = 0.09
        assert abs(compute_brier_score([0.3], [0]) - 0.09) < 1e-9

    def test_perfect_predictions_zero(self):
        preds = [1.0, 1.0, 0.0, 0.0]
        actuals = [1, 1, 0, 0]
        assert compute_brier_score(preds, actuals) == pytest.approx(0.0, abs=1e-9)

    def test_worst_predictions(self):
        # Always predict 1.0 when actual is 0, and vice versa → BS = 1.0
        preds = [1.0, 0.0]
        actuals = [0, 1]
        assert compute_brier_score(preds, actuals) == pytest.approx(1.0, abs=1e-9)

    def test_random_baseline_near_025(self):
        # Always predict 0.5 → BS = (0.5-y)^2 = 0.25 for all y ∈ {0,1}
        preds = [0.5] * 100
        actuals = [1] * 50 + [0] * 50
        assert compute_brier_score(preds, actuals) == pytest.approx(0.25, abs=1e-9)

    def test_returns_float(self):
        result = compute_brier_score([0.5], [1])
        assert isinstance(result, float)

    def test_three_game_example(self):
        # predict [0.7, 0.6, 0.4], actual [1, 0, 1]
        # (0.3)^2 = 0.09, (0.6)^2 = 0.36, (0.6)^2 = 0.36; mean = 0.27
        preds = [0.7, 0.6, 0.4]
        actuals = [1, 0, 1]
        expected = ((0.3**2) + (0.6**2) + (0.6**2)) / 3
        assert compute_brier_score(preds, actuals) == pytest.approx(expected, abs=1e-9)

    def test_array_input_accepted(self):
        preds = np.array([0.8, 0.3])
        actuals = np.array([1, 0])
        assert compute_brier_score(preds, actuals) == pytest.approx(0.065, abs=1e-9)


# ---------------------------------------------------------------------------
# calibration_report
# ---------------------------------------------------------------------------

class TestCalibrationReport:

    def _make_data(self, n=100):
        rng = np.random.default_rng(0)
        preds = rng.uniform(0.05, 0.95, n)
        actuals = (rng.random(n) < preds).astype(int)
        return preds, actuals

    def test_returns_dict_with_required_keys(self):
        p, a = self._make_data()
        report = calibration_report(p, a)
        assert "bins" in report
        assert "ece" in report
        assert "brier" in report

    def test_bins_is_list(self):
        p, a = self._make_data()
        report = calibration_report(p, a)
        assert isinstance(report["bins"], list)

    def test_bin_dicts_have_required_keys(self):
        p, a = self._make_data()
        report = calibration_report(p, a)
        for b in report["bins"]:
            for key in ("bin_centre", "mean_pred", "fraction_pos", "n"):
                assert key in b, f"Missing key '{key}' in bin dict"

    def test_brier_matches_standalone(self):
        p, a = self._make_data()
        report = calibration_report(p, a)
        expected = compute_brier_score(p, a)
        assert abs(report["brier"] - expected) < 1e-9

    def test_ece_non_negative(self):
        p, a = self._make_data()
        report = calibration_report(p, a)
        assert report["ece"] >= 0.0

    def test_n_bins_respected(self):
        p = np.linspace(0.01, 0.99, 100)
        a = (p > 0.5).astype(int)
        report = calibration_report(p, a, n_bins=5)
        # Should have at most 5 non-empty bins
        assert len(report["bins"]) <= 5

    def test_saves_json(self, tmp_path):
        p, a = self._make_data()
        json_path = tmp_path / "cal.json"
        calibration_report(p, a, save_json=json_path)
        assert json_path.exists()
        data = json.loads(json_path.read_text())
        assert "bins" in data and "ece" in data


# ---------------------------------------------------------------------------
# per_seed_analysis
# ---------------------------------------------------------------------------

class TestPerSeedAnalysis:

    def _make_data(self):
        # Favourites (seed1 < seed2): team1 wins 80%
        # Upsets (seed1 > seed2): team1 wins 30%
        rng = np.random.default_rng(42)
        n = 50
        seeds1 = np.concatenate([np.ones(n // 2, int), np.full(n // 2, 8, int)])
        seeds2 = np.concatenate([np.full(n // 2, 8, int), np.ones(n // 2, int)])
        preds = np.concatenate([
            np.full(n // 2, 0.80),  # favourites predict high
            np.full(n // 2, 0.30),  # upsets predict low
        ])
        actuals = np.concatenate([
            (rng.random(n // 2) < 0.80).astype(int),
            (rng.random(n // 2) < 0.30).astype(int),
        ])
        return preds, actuals, seeds1, seeds2

    def test_returns_dict(self):
        p, a, s1, s2 = self._make_data()
        result = per_seed_analysis(p, a, s1, s2)
        assert isinstance(result, dict)

    def test_has_favourite_and_upset_keys(self):
        p, a, s1, s2 = self._make_data()
        result = per_seed_analysis(p, a, s1, s2)
        assert "favourite" in result
        assert "upset" in result

    def test_category_has_required_keys(self):
        p, a, s1, s2 = self._make_data()
        result = per_seed_analysis(p, a, s1, s2)
        for cat in ("favourite", "upset"):
            for key in ("n", "accuracy", "brier", "mean_pred", "fraction_pos"):
                assert key in result[cat], f"Missing '{key}' in '{cat}'"

    def test_n_sums_to_total(self):
        p, a, s1, s2 = self._make_data()
        result = per_seed_analysis(p, a, s1, s2)
        total_n = sum(result[c]["n"] for c in ("favourite", "upset") if c in result)
        total_n += result.get("equal_seed", {}).get("n", 0)
        assert total_n == len(p)

    def test_accuracy_in_0_1(self):
        p, a, s1, s2 = self._make_data()
        result = per_seed_analysis(p, a, s1, s2)
        for cat in ("favourite", "upset"):
            acc = result[cat]["accuracy"]
            assert 0.0 <= acc <= 1.0

    def test_saves_json(self, tmp_path):
        p, a, s1, s2 = self._make_data()
        json_path = tmp_path / "seed.json"
        per_seed_analysis(p, a, s1, s2, save_json=json_path)
        assert json_path.exists()

    def test_by_seed_diff_quintile_present(self):
        p, a, s1, s2 = self._make_data()
        result = per_seed_analysis(p, a, s1, s2)
        assert "by_seed_diff_quintile" in result


# ---------------------------------------------------------------------------
# per_round_analysis
# ---------------------------------------------------------------------------

class TestPerRoundAnalysis:

    def _make_data(self):
        # 7 games: 4 in R1, 2 in R2, 1 Final
        rounds = np.array([1, 1, 1, 1, 2, 2, 3])
        preds = np.array([0.8, 0.7, 0.6, 0.9, 0.75, 0.65, 0.80])
        actuals = np.array([1, 1, 0, 1, 1, 0, 1])
        return preds, actuals, rounds

    def test_returns_dict(self):
        p, a, r = self._make_data()
        result = per_round_analysis(p, a, r)
        assert isinstance(result, dict)

    def test_has_expected_rounds(self):
        p, a, r = self._make_data()
        result = per_round_analysis(p, a, r)
        assert 1 in result and 2 in result and 3 in result

    def test_round_has_required_keys(self):
        p, a, r = self._make_data()
        result = per_round_analysis(p, a, r)
        for rnd, data in result.items():
            for key in ("n", "brier", "accuracy"):
                assert key in data, f"Round {rnd} missing '{key}'"

    def test_n_in_round1_is_4(self):
        p, a, r = self._make_data()
        result = per_round_analysis(p, a, r)
        assert result[1]["n"] == 4

    def test_total_n_matches(self):
        p, a, r = self._make_data()
        result = per_round_analysis(p, a, r)
        total = sum(v["n"] for v in result.values())
        assert total == len(p)

    def test_brier_non_negative(self):
        p, a, r = self._make_data()
        result = per_round_analysis(p, a, r)
        for v in result.values():
            assert v["brier"] >= 0.0

    def test_saves_json(self, tmp_path):
        p, a, r = self._make_data()
        json_path = tmp_path / "rounds.json"
        per_round_analysis(p, a, r, save_json=json_path)
        assert json_path.exists()
        data = json.loads(json_path.read_text())
        # Keys are strings in JSON
        assert "1" in data or 1 in data
