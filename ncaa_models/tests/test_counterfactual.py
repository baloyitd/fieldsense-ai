"""
test_counterfactual.py
======================
Unit tests for ncaa_models.counterfactual — CounterfactualGenerator and
perturbation functions.

Tests cover:
- Each perturbation produces basketball-valid feature deltas
- Deterministic outputs: same seed + same input → identical scenario traces
- Output shape, field types, delta_prob direction
"""

from __future__ import annotations

import numpy as np
import pytest

from ncaa_models.baseline import FEATURE_COLS
from ncaa_models.counterfactual import (
    ALL_PERTURBATION_TYPES,
    BOTH_TEAMS,
    DEFAULT_SEVERITIES,
    COLD_SHOOTING,
    FOUL_TROUBLE,
    HOT_SHOOTING,
    INJURY,
    REST_ADVANTAGE,
    CounterfactualGenerator,
    CounterfactualScenario,
    _apply_cold_shooting,
    _apply_foul_trouble,
    _apply_hot_shooting,
    _apply_injury,
    _apply_rest_advantage,
)

FEATURE_DIM = len(FEATURE_COLS)
FEAT_IDX = {c: i for i, c in enumerate(FEATURE_COLS)}


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def canonical_team():
    """Feature vector for a seed-4 level team."""
    X = np.array([
        30.0,   # games_played
        0.65,   # win_pct
        78.0,   # ppg_scored
        72.0,   # ppg_allowed
        6.0,    # scoring_margin
        0.75,   # home_win_pct
        0.55,   # away_win_pct
        0.65,   # neutral_win_pct
        0.65,   # win_pct_last10
        0.45,   # fg_pct
        0.36,   # fg3_pct
        0.72,   # ft_pct
        0.52,   # efg_pct
        0.54,   # ts_pct
        0.30,   # orb_rate
        0.70,   # drb_rate
        72.0,   # possessions_pg
        108.0,  # ortg
        100.0,  # drtg
        8.0,    # net_rtg
        72.0,   # tempo
        0.17,   # tov_rate
        0.58,   # ast_rate
        0.09,   # blk_rate
        0.08,   # stl_rate
        0.05,   # ot_rate
        0.52,   # sos
        4.0,    # seed
    ], dtype=float)
    assert X.shape == (FEATURE_DIM,)
    return X


@pytest.fixture
def opponent_team(canonical_team):
    """Stronger seed-1 team."""
    X = canonical_team.copy()
    X[FEAT_IDX["seed"]] = 1.0
    X[FEAT_IDX["net_rtg"]] = 20.0
    X[FEAT_IDX["ortg"]] = 118.0
    X[FEAT_IDX["ppg_scored"]] = 85.0
    X[FEAT_IDX["win_pct"]] = 0.88
    return X


@pytest.fixture
def simple_model():
    """Toy model: returns 0.5 for all matchups (neutral baseline)."""
    def _predict(X1, X2):
        return np.full(X1.shape[0], 0.5)
    return _predict


@pytest.fixture
def net_rtg_model():
    """Model: P(team1 wins) = sigmoid(0.15 * net_rtg_diff)."""
    def _predict(X1, X2):
        diff = X1[:, FEAT_IDX["net_rtg"]] - X2[:, FEAT_IDX["net_rtg"]]
        return 1.0 / (1.0 + np.exp(-0.15 * diff))
    return _predict


@pytest.fixture
def generator(simple_model):
    return CounterfactualGenerator(model=simple_model, feature_cols=FEATURE_COLS)


# ---------------------------------------------------------------------------
# TestPerturbationBounds — each perturbation stays within basketball-valid bounds
# ---------------------------------------------------------------------------

class TestPerturbationBounds:
    """Each perturbation produces feature deltas within basketball-valid ranges."""

    VALID_BOUNDS = {
        "fg_pct":   (0.20, 0.65),
        "fg3_pct":  (0.18, 0.62),
        "efg_pct":  (0.25, 0.72),
        "ts_pct":   (0.30, 0.76),
        "ortg":     (70.0, 140.0),
        "drtg":     (70.0, 140.0),
        "net_rtg":  (-50.0, 50.0),
        "ppg_scored": (40.0, 110.0),
        "tov_rate": (0.05, 0.40),
    }

    def _check_bounds(self, X: np.ndarray) -> None:
        for feat, (lo, hi) in self.VALID_BOUNDS.items():
            val = X[FEAT_IDX[feat]]
            assert lo <= val <= hi, (
                f"Feature '{feat}'={val:.4f} outside [{lo}, {hi}] after perturbation"
            )

    @pytest.mark.parametrize("severity", [0.0, 0.25, 0.50, 0.75, 1.0])
    def test_injury_bounds(self, canonical_team, severity):
        X_p = _apply_injury(canonical_team, severity, FEAT_IDX)
        self._check_bounds(X_p)

    @pytest.mark.parametrize("severity", [0.0, 0.25, 0.50, 0.75, 1.0])
    def test_foul_trouble_bounds(self, canonical_team, severity):
        X_p = _apply_foul_trouble(canonical_team, severity, FEAT_IDX)
        self._check_bounds(X_p)

    @pytest.mark.parametrize("severity", [0.0, 0.25, 0.50, 0.75, 1.0])
    def test_hot_shooting_bounds(self, canonical_team, severity):
        X_p = _apply_hot_shooting(canonical_team, severity, FEAT_IDX)
        self._check_bounds(X_p)

    @pytest.mark.parametrize("severity", [0.0, 0.25, 0.50, 0.75, 1.0])
    def test_cold_shooting_bounds(self, canonical_team, severity):
        X_p = _apply_cold_shooting(canonical_team, severity, FEAT_IDX)
        self._check_bounds(X_p)

    @pytest.mark.parametrize("severity", [0.0, 0.25, 0.50, 0.75, 1.0])
    def test_rest_advantage_bounds(self, canonical_team, severity):
        X_p = _apply_rest_advantage(canonical_team, severity, FEAT_IDX)
        self._check_bounds(X_p)

    def test_hot_shooting_3p_delta_max_15pp(self, canonical_team):
        """FG3% adjustment ≤ +15 pp from season average (spec requirement)."""
        base_fg3 = canonical_team[FEAT_IDX["fg3_pct"]]
        for sev in np.linspace(0, 1, 11):
            X_p = _apply_hot_shooting(canonical_team, sev, FEAT_IDX)
            delta = X_p[FEAT_IDX["fg3_pct"]] - base_fg3
            assert delta <= 0.151, f"FG3% delta {delta:.3f} > 15pp at severity {sev:.2f}"

    def test_cold_shooting_3p_delta_max_15pp(self, canonical_team):
        """FG3% adjustment ≥ -15 pp from season average (spec requirement)."""
        base_fg3 = canonical_team[FEAT_IDX["fg3_pct"]]
        for sev in np.linspace(0, 1, 11):
            X_p = _apply_cold_shooting(canonical_team, sev, FEAT_IDX)
            delta = base_fg3 - X_p[FEAT_IDX["fg3_pct"]]
            assert delta <= 0.151, f"FG3% drop {delta:.3f} > 15pp at severity {sev:.2f}"

    def test_injury_reduces_ortg(self, canonical_team):
        """Injury perturbation always reduces ORTG."""
        base_ortg = canonical_team[FEAT_IDX["ortg"]]
        for sev in [0.0, 0.25, 0.5, 0.75, 1.0]:
            X_p = _apply_injury(canonical_team, sev, FEAT_IDX)
            assert X_p[FEAT_IDX["ortg"]] <= base_ortg, (
                f"Injury did not reduce ORTG at severity {sev}"
            )

    def test_hot_increases_cold_decreases_fg3(self, canonical_team):
        base = canonical_team[FEAT_IDX["fg3_pct"]]
        assert _apply_hot_shooting(canonical_team, 0.5, FEAT_IDX)[FEAT_IDX["fg3_pct"]] > base
        assert _apply_cold_shooting(canonical_team, 0.5, FEAT_IDX)[FEAT_IDX["fg3_pct"]] < base

    def test_rest_reduces_tov_rate(self, canonical_team):
        base_tov = canonical_team[FEAT_IDX["tov_rate"]]
        X_p = _apply_rest_advantage(canonical_team, 0.5, FEAT_IDX)
        assert X_p[FEAT_IDX["tov_rate"]] <= base_tov

    def test_perturbation_does_not_mutate_original(self, canonical_team):
        orig = canonical_team.copy()
        _apply_injury(canonical_team, 0.5, FEAT_IDX)
        np.testing.assert_array_equal(canonical_team, orig)


# ---------------------------------------------------------------------------
# TestCounterfactualGenerator
# ---------------------------------------------------------------------------

class TestCounterfactualGenerator:

    def test_generate_returns_list_of_scenarios(self, generator, canonical_team, opponent_team):
        scenarios = generator.generate(canonical_team, opponent_team)
        assert isinstance(scenarios, list)
        assert all(isinstance(s, CounterfactualScenario) for s in scenarios)

    def test_scenario_count(self, generator, canonical_team, opponent_team):
        """n_types × n_teams × n_severities scenarios."""
        scenarios = generator.generate(canonical_team, opponent_team)
        expected = len(ALL_PERTURBATION_TYPES) * len(BOTH_TEAMS) * len(DEFAULT_SEVERITIES)
        assert len(scenarios) == expected

    def test_base_prob_consistent(self, generator, canonical_team, opponent_team):
        """All scenarios share the same base_prob."""
        scenarios = generator.generate(canonical_team, opponent_team)
        base = scenarios[0].base_prob
        for s in scenarios:
            assert abs(s.base_prob - base) < 1e-9

    def test_delta_prob_consistency(self, generator, canonical_team, opponent_team):
        """delta_prob == prob - base_prob for every scenario."""
        scenarios = generator.generate(canonical_team, opponent_team)
        for s in scenarios:
            assert abs(s.delta_prob - (s.prob - s.base_prob)) < 1e-9

    def test_probs_in_range(self, generator, canonical_team, opponent_team):
        scenarios = generator.generate(canonical_team, opponent_team)
        for s in scenarios:
            assert 0.0 <= s.prob <= 1.0
            assert 0.0 <= s.base_prob <= 1.0

    def test_deterministic_same_seed(self, generator, canonical_team, opponent_team):
        """Same seed + same input → identical scenario traces."""
        s1 = generator.generate(canonical_team, opponent_team, seed=42)
        s2 = generator.generate(canonical_team, opponent_team, seed=42)
        for a, b in zip(s1, s2):
            np.testing.assert_array_equal(a.X1, b.X1)
            np.testing.assert_array_equal(a.X2, b.X2)
            assert a.prob == b.prob
            assert a.cvs == b.cvs

    def test_different_seeds_allowed_same_output(self, generator, canonical_team, opponent_team):
        """Seed does not affect deterministic perturbations (they're formulaic, not random)."""
        s1 = generator.generate(canonical_team, opponent_team, seed=0)
        s2 = generator.generate(canonical_team, opponent_team, seed=99)
        # Perturbations are deterministic given (ptype, team, severity) — seed unused for now
        for a, b in zip(s1, s2):
            np.testing.assert_array_almost_equal(a.X1, b.X1)

    def test_team_field(self, generator, canonical_team, opponent_team):
        """Each scenario records which team was perturbed."""
        scenarios = generator.generate(canonical_team, opponent_team)
        for s in scenarios:
            assert s.team in ("team1", "team2")

    def test_cvs_default_one_when_no_computer(self, generator, canonical_team, opponent_team):
        """Without a CVSComputer, CVS defaults to 1.0."""
        scenarios = generator.generate(canonical_team, opponent_team)
        for s in scenarios:
            assert s.cvs == 1.0
            assert s.is_valid is True

    def test_generate_single_perturbation_type(self, generator, canonical_team, opponent_team):
        scenarios = generator.generate(
            canonical_team, opponent_team, perturbation_types=[HOT_SHOOTING]
        )
        assert all(s.perturbation == HOT_SHOOTING for s in scenarios)

    def test_generate_single_team(self, generator, canonical_team, opponent_team):
        scenarios = generator.generate(
            canonical_team, opponent_team, teams=("team1",)
        )
        assert all(s.team == "team1" for s in scenarios)

    def test_net_rtg_model_injury_lowers_prob(self, net_rtg_model, canonical_team, opponent_team):
        """Injury on team1 reduces team1's win probability (net_rtg model)."""
        gen = CounterfactualGenerator(model=net_rtg_model, feature_cols=FEATURE_COLS)
        scenarios = gen.generate(
            canonical_team, opponent_team,
            perturbation_types=[INJURY],
            teams=("team1",),
        )
        for s in scenarios:
            assert s.delta_prob < 0, (
                f"Injury on team1 increased prob by {s.delta_prob:.4f}"
            )

    def test_net_rtg_model_injury_raises_opponent_prob(self, net_rtg_model, canonical_team, opponent_team):
        """Injury on team2 (the stronger team) benefits team1."""
        gen = CounterfactualGenerator(model=net_rtg_model, feature_cols=FEATURE_COLS)
        scenarios = gen.generate(
            canonical_team, opponent_team,
            perturbation_types=[INJURY],
            teams=("team2",),
        )
        for s in scenarios:
            assert s.delta_prob > 0, (
                f"Injury on team2 decreased team1 prob by {s.delta_prob:.4f}"
            )

    def test_generate_single_method(self, generator, canonical_team, opponent_team):
        s = generator.generate_single(canonical_team, opponent_team, HOT_SHOOTING, "team1", 0.5)
        assert isinstance(s, CounterfactualScenario)
        assert s.perturbation == HOT_SHOOTING
        assert s.team == "team1"
        assert abs(s.severity - 0.5) < 1e-9

    def test_adjusted_probability_returns_float(self, generator, canonical_team, opponent_team):
        p = generator.adjusted_probability(canonical_team, opponent_team)
        assert isinstance(p, float)
        assert 0.0 <= p <= 1.0

    def test_1d_and_2d_input_equivalent(self, generator, canonical_team, opponent_team):
        s1 = generator.generate(canonical_team, opponent_team)
        s2 = generator.generate(
            canonical_team.reshape(1, -1),
            opponent_team.reshape(1, -1),
        )
        for a, b in zip(s1, s2):
            np.testing.assert_array_almost_equal(a.X1, b.X1)

    def test_unknown_perturbation_raises(self, generator, canonical_team, opponent_team):
        with pytest.raises(ValueError, match="Unknown perturbation"):
            generator.generate(canonical_team, opponent_team,
                               perturbation_types=["invalid_type"])
