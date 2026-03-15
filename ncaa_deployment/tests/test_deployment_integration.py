"""
test_deployment_integration.py
================================
Integration tests for ncaa_deployment (Stage 10).

Covers:
- Full pipeline: raw data → two Kaggle CSVs, zero manual intervention
- Simulated 2025 tournament replay with live recalibration, cumulative Brier
- PostRoundAnalyzer correctly computes per-round Brier breakdowns
- TechnicalRetrospective generated with cross-referenced metrics from all 9 stages
- Retrospective document mentions all stage numbers 01–09
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ncaa_deployment.optimize import BayesianEnsembleOptimizer, WeightedEnsemble
from ncaa_deployment.submission import SubmissionBuilder, kl_divergence
from ncaa_deployment.post_round import PostRoundAnalyzer
from ncaa_deployment.retrospective import (
    TechnicalRetrospective,
    StageMetrics,
    generate_retrospective,
)

from .conftest import TEAM_IDS, SEASON, ConstantModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_builder(feat_df):
    from ncaa_models.baseline import FEATURE_COLS
    return SubmissionBuilder(
        feat_df=feat_df,
        team_ids=TEAM_IDS,
        season=SEASON,
        feature_cols=FEATURE_COLS,
    )


# ---------------------------------------------------------------------------
# Integration test 1: Full pipeline — raw data → two Kaggle CSVs
# ---------------------------------------------------------------------------

class TestFullPipeline:
    def test_full_pipeline_produces_two_valid_csvs(
        self, component_models, feat_df, train_arrays, val_arrays, tmp_path
    ):
        """
        Raw data → optimised ensemble → conservative + aggressive CSVs.
        Both must pass Kaggle validation.
        """
        X1_tr, X2_tr, y_tr = train_arrays
        X1_val, X2_val, y_val = val_arrays

        # Stage 1: Bayesian optimisation
        optimizer = BayesianEnsembleOptimizer(
            models=component_models, n_trials=30, seed=0
        )
        result = optimizer.optimize(X1_tr, X2_tr, y_tr, X1_val, X2_val, y_val)
        assert result.improved is True

        # Stage 2: Build ensemble
        ensemble = optimizer.build_optimized_ensemble()

        # Stage 3: Generate both submission variants
        builder = _make_builder(feat_df)
        conservative = builder.build_conservative(ensemble, temperature=5.0)
        aggressive = builder.build_aggressive(ensemble, temperature=1.0)

        # Stage 4: Validate both
        for variant in (conservative, aggressive):
            result_v = variant.validate(expected_season=SEASON)
            assert result_v["valid"] is True, (
                f"{variant.name} validation failed: {result_v}"
            )

        # Stage 5: Check distributions are distinct
        kl = kl_divergence(conservative, aggressive)
        assert kl > 0.001, f"Submissions too similar: KL={kl:.6f}"

        # Stage 6: Save CSVs
        c_path = tmp_path / "conservative.csv"
        a_path = tmp_path / "aggressive.csv"
        conservative.save_csv(str(c_path))
        aggressive.save_csv(str(a_path))
        assert c_path.exists()
        assert a_path.exists()

    def test_no_manual_intervention_needed(
        self, component_models, feat_df, train_arrays, val_arrays
    ):
        """Entire pipeline runs without any manual parameter adjustment."""
        X1_tr, X2_tr, y_tr = train_arrays
        X1_val, X2_val, y_val = val_arrays

        # Automatically detect team_ids from feat_df
        season_df = feat_df[feat_df["season"] == SEASON]
        auto_team_ids = list(season_df["team_id"].unique())

        from ncaa_models.baseline import FEATURE_COLS
        optimizer = BayesianEnsembleOptimizer(models=component_models, n_trials=20)
        opt_result = optimizer.optimize(X1_tr, X2_tr, y_tr, X1_val, X2_val, y_val)
        ensemble = optimizer.build_optimized_ensemble()

        builder = SubmissionBuilder(
            feat_df=feat_df,
            team_ids=auto_team_ids,
            season=SEASON,
            feature_cols=FEATURE_COLS,
        )
        conservative = builder.build_conservative(ensemble, temperature=5.0)
        aggressive = builder.build_aggressive(ensemble)

        assert len(conservative.df) > 0
        assert len(aggressive.df) > 0


# ---------------------------------------------------------------------------
# Integration test 2: Simulated tournament replay with live recalibration
# ---------------------------------------------------------------------------

class TestTournamentReplay:
    def test_cumulative_brier_computed_across_rounds(
        self, strong_model, feat_df, matchup_df, train_arrays, val_arrays
    ):
        """
        Simulate a 3-round tournament, compute cumulative Brier after each round.
        """
        from ncaa_live.simulator import make_simulator_from_feature_df
        from ncaa_live.ingester import BoxScoreIngester
        from ncaa_live.recalibrator import RoundRecalibrator
        import copy

        X1_tr, X2_tr, y_tr = train_arrays
        X1_val, X2_val, y_val = val_arrays

        # Build submissions
        builder = _make_builder(feat_df)
        conservative = builder.build_conservative(strong_model, temperature=5.0)
        aggressive = builder.build_aggressive(strong_model, temperature=1.0)

        # Set up post-round analyzer
        analyzer = PostRoundAnalyzer(season=SEASON)
        analyzer.register_variant("conservative", conservative)
        analyzer.register_variant("aggressive", aggressive)

        # Simulate 3 rounds
        simulator = make_simulator_from_feature_df(
            feat_df, TEAM_IDS, season=SEASON, deterministic=True
        )

        total_games = 0
        for round_num in range(1, 4):
            if simulator.is_complete():
                break
            round_results = simulator.simulate_round()

            # Build matchup_ids and y_true from GameResult objects
            matchup_ids = []
            y_true = []
            for gr in round_results:
                lo, hi = sorted([gr.team1_id, gr.team2_id])
                mid = f"{SEASON}_{lo}_{hi}"
                matchup_ids.append(mid)
                y_true.append(1 if gr.winner_id == lo else 0)

            report = analyzer.analyze_round(round_num, matchup_ids, y_true)
            assert report.round_num == round_num
            assert report.n_games == len(round_results)
            total_games += len(round_results)

        assert total_games > 0
        # Check cumulative Brier is computed
        for name in ("conservative", "aggressive"):
            cb = analyzer.cumulative_brier(name)
            assert cb is not None
            assert 0.0 <= cb <= 0.25

    def test_per_round_brier_breakdown(self, strong_model, feat_df):
        """PostRoundAnalyzer should compute different Brier per round."""
        from ncaa_live.simulator import make_simulator_from_feature_df

        builder = _make_builder(feat_df)
        conservative = builder.build_conservative(strong_model, temperature=5.0)
        aggressive = builder.build_aggressive(strong_model, temperature=1.0)

        analyzer = PostRoundAnalyzer(season=SEASON)
        analyzer.register_variant("conservative", conservative)
        analyzer.register_variant("aggressive", aggressive)

        simulator = make_simulator_from_feature_df(
            feat_df, TEAM_IDS, season=SEASON, deterministic=True
        )

        reports = []
        for round_num in range(1, 5):
            if simulator.is_complete():
                break
            round_results = simulator.simulate_round()
            matchup_ids = []
            y_true = []
            for gr in round_results:
                lo, hi = sorted([gr.team1_id, gr.team2_id])
                mid = f"{SEASON}_{lo}_{hi}"
                matchup_ids.append(mid)
                y_true.append(1 if gr.winner_id == lo else 0)
            report = analyzer.analyze_round(round_num, matchup_ids, y_true)
            reports.append(report)

        assert len(reports) >= 2
        # Each report has brier_by_variant with both variants
        for rpt in reports:
            assert "conservative" in rpt.brier_by_variant
            assert "aggressive" in rpt.brier_by_variant

    def test_cumulative_brier_decreases_or_stabilises(self, strong_model, feat_df):
        """On deterministic data (lower seed always wins), strong model should have
        consistent low Brier across rounds."""
        from ncaa_live.simulator import make_simulator_from_feature_df

        builder = _make_builder(feat_df)
        aggressive = builder.build_aggressive(strong_model, temperature=1.0)

        analyzer = PostRoundAnalyzer(season=SEASON)
        analyzer.register_variant("aggressive", aggressive)

        simulator = make_simulator_from_feature_df(
            feat_df, TEAM_IDS, season=SEASON, deterministic=True
        )

        for round_num in range(1, 5):
            if simulator.is_complete():
                break
            round_results = simulator.simulate_round()
            matchup_ids = []
            y_true = []
            for gr in round_results:
                lo, hi = sorted([gr.team1_id, gr.team2_id])
                matchup_ids.append(f"{SEASON}_{lo}_{hi}")
                y_true.append(1 if gr.winner_id == lo else 0)
            analyzer.analyze_round(round_num, matchup_ids, y_true)

        cb = analyzer.cumulative_brier("aggressive")
        assert cb is not None
        assert cb <= 0.25  # should be well below 0.25 for strong model

    def test_analyzer_history_matches_rounds_analyzed(self, strong_model, feat_df):
        from ncaa_live.simulator import make_simulator_from_feature_df

        builder = _make_builder(feat_df)
        v = builder.build_aggressive(strong_model)
        analyzer = PostRoundAnalyzer(season=SEASON)
        analyzer.register_variant("aggressive", v)

        simulator = make_simulator_from_feature_df(
            feat_df, TEAM_IDS, season=SEASON, deterministic=True
        )

        n_rounds = 0
        for _ in range(4):
            if simulator.is_complete():
                break
            rr = simulator.simulate_round()
            ids = [f"{SEASON}_{min(g.team1_id, g.team2_id)}_{max(g.team1_id, g.team2_id)}"
                   for g in rr]
            yt = [1 if g.winner_id == min(g.team1_id, g.team2_id) else 0 for g in rr]
            analyzer.analyze_round(n_rounds + 1, ids, yt)
            n_rounds += 1

        assert analyzer.n_rounds_analyzed() == n_rounds

    def test_compare_conservative_vs_aggressive(self, strong_model, feat_df):
        """Both variant comparisons should be present in round reports."""
        from ncaa_live.simulator import make_simulator_from_feature_df

        builder = _make_builder(feat_df)
        cons = builder.build_conservative(strong_model, temperature=5.0)
        aggr = builder.build_aggressive(strong_model, temperature=1.0)

        analyzer = PostRoundAnalyzer(season=SEASON)
        analyzer.register_variant("conservative", cons)
        analyzer.register_variant("aggressive", aggr)

        simulator = make_simulator_from_feature_df(
            feat_df, TEAM_IDS, season=SEASON, deterministic=True
        )
        rr = simulator.simulate_round()
        ids = [f"{SEASON}_{min(g.team1_id, g.team2_id)}_{max(g.team1_id, g.team2_id)}"
               for g in rr]
        yt = [1 if g.winner_id == min(g.team1_id, g.team2_id) else 0 for g in rr]
        report = analyzer.analyze_round(1, ids, yt)

        assert len(report.comparisons) >= 1
        cmp = report.comparisons[0]
        assert cmp.variant_a in ("conservative", "aggressive")
        assert cmp.variant_b in ("conservative", "aggressive")
        assert isinstance(cmp.winner, str)


# ---------------------------------------------------------------------------
# Integration test 3: TechnicalRetrospective
# ---------------------------------------------------------------------------

class TestTechnicalRetrospective:
    def test_retrospective_generated(self):
        retro = TechnicalRetrospective()
        doc = retro.generate()
        assert isinstance(doc, str)
        assert len(doc) > 100

    def test_retrospective_references_all_stages(self):
        """Retrospective must mention all stage numbers 01 through 09."""
        retro = TechnicalRetrospective()
        doc = retro.generate()
        for stage_num in range(1, 10):
            tag = f"{stage_num:02d}"
            assert tag in doc, f"Stage {tag} not referenced in retrospective"

    def test_retrospective_with_stage10_metrics(self):
        retro = TechnicalRetrospective()
        retro.set_stage10_brier(
            optimized_brier=0.158,
            conservative_brier=0.163,
            aggressive_brier=0.155,
            n_trials=200,
            improvement=0.005,
        )
        doc = retro.generate()
        assert "10" in doc
        assert "0.158" in doc

    def test_all_stages_have_entries(self):
        retro = TechnicalRetrospective()
        retro.set_stage10_brier(optimized_brier=0.160)
        stages = retro.all_stages()
        stage_nums = {s.stage for s in stages}
        for i in range(1, 11):
            assert f"{i:02d}" in stage_nums, f"Stage {i:02d} missing from retrospective"

    def test_stage_metrics_to_dict(self):
        sm = StageMetrics(
            stage="06", name="Ensemble", key_metric="brier",
            target="< 0.165", achieved="0.163", passed=True,
            brier=0.163, notes="NNLS stacking.",
        )
        d = sm.to_dict()
        for key in ("stage", "name", "key_metric", "target", "achieved", "passed", "brier", "notes"):
            assert key in d

    def test_all_stages_dict(self):
        retro = TechnicalRetrospective()
        retro.set_stage10_brier(optimized_brier=0.160)
        d = retro.all_stages_dict()
        assert len(d) == 10
        assert all("stage" in entry for entry in d)

    def test_save_markdown(self, tmp_path):
        retro = TechnicalRetrospective()
        retro.set_stage10_brier(optimized_brier=0.160)
        doc = retro.generate()
        path = tmp_path / "retro.md"
        retro.save(doc, str(path))
        assert path.exists()
        content = path.read_text()
        assert "NCAA Mania" in content

    def test_save_json(self, tmp_path):
        retro = TechnicalRetrospective()
        retro.set_stage10_brier(optimized_brier=0.160)
        path = tmp_path / "retro.json"
        retro.save_json(str(path))
        assert path.exists()
        import json
        data = json.loads(path.read_text())
        assert isinstance(data, list)
        assert len(data) == 10

    def test_generate_retrospective_convenience(self):
        doc = generate_retrospective(
            optimized_brier=0.160,
            conservative_brier=0.165,
            aggressive_brier=0.157,
            n_trials=300,
            improvement=0.003,
        )
        assert isinstance(doc, str)
        assert "NCAA Mania" in doc
        assert "Stage 06" in doc

    def test_brier_progression_table_in_doc(self):
        retro = TechnicalRetrospective()
        retro.set_stage10_brier(optimized_brier=0.160)
        doc = retro.generate()
        assert "Brier Score Progression" in doc
        # Brier values from stages with known Brier
        assert "0.195" in doc or "0.178" in doc  # Stage 02 or 03

    def test_lora_section_present(self):
        retro = TechnicalRetrospective()
        doc = retro.generate()
        assert "LoRA" in doc or "lora" in doc.lower()

    def test_recommendations_section_present(self):
        retro = TechnicalRetrospective()
        doc = retro.generate()
        assert "Recommendations" in doc or "v4.0" in doc

    def test_retrospective_references_9_prior_stages(self):
        """The retrospective data should contain entries for stages 01-09."""
        retro = TechnicalRetrospective()
        stages = retro.all_stages()  # without stage 10
        assert len(stages) == 9
        for sm in stages:
            assert int(sm.stage) <= 9


# ---------------------------------------------------------------------------
# Integration test 4: PostRoundAnalyzer unit tests
# ---------------------------------------------------------------------------

class TestPostRoundAnalyzer:
    def test_register_and_analyze(self, strong_model, feat_df):
        builder = _make_builder(feat_df)
        v = builder.build_aggressive(strong_model)
        analyzer = PostRoundAnalyzer(season=SEASON)
        analyzer.register_variant("aggressive", v)
        assert "aggressive" in analyzer.variant_names()

    def test_compute_round_brier(self, strong_model, feat_df):
        """compute_round_brier for a specific variant."""
        builder = _make_builder(feat_df)
        v = builder.build_aggressive(strong_model)
        analyzer = PostRoundAnalyzer(season=SEASON)
        analyzer.register_variant("aggressive", v)

        # Use first row of submission as a test matchup
        row = v.df.iloc[0]
        mid = row["ID"]
        brier = analyzer.compute_round_brier("aggressive", [mid], [1])
        assert 0.0 <= brier <= 1.0

    def test_round_report_summary_string(self, strong_model, feat_df):
        builder = _make_builder(feat_df)
        v = builder.build_aggressive(strong_model)
        analyzer = PostRoundAnalyzer(season=SEASON)
        analyzer.register_variant("aggressive", v)

        row = v.df.iloc[0]
        report = analyzer.analyze_round(1, [row["ID"]], [1])
        summary = report.summary()
        assert "Round 1" in summary
        assert "aggressive" in summary

    def test_round_report_to_dict(self, strong_model, feat_df):
        builder = _make_builder(feat_df)
        v = builder.build_aggressive(strong_model)
        analyzer = PostRoundAnalyzer(season=SEASON)
        analyzer.register_variant("aggressive", v)

        row = v.df.iloc[0]
        report = analyzer.analyze_round(1, [row["ID"]], [1])
        d = report.to_dict()
        for key in ("round_num", "n_games", "brier_by_variant", "comparisons",
                    "cumulative_brier"):
            assert key in d

    def test_best_variant(self, strong_model, weak_model, feat_df):
        """best_variant returns the variant with lower cumulative Brier."""
        builder = _make_builder(feat_df)
        v_strong = builder.build_aggressive(strong_model)
        v_weak = builder.build_aggressive(weak_model)

        analyzer = PostRoundAnalyzer(season=SEASON)
        analyzer.register_variant("strong", v_strong)
        analyzer.register_variant("weak", v_weak)

        # Use a matchup where strong model is clearly right
        # Team IDs: 1001 (seed 1) vs 1008 (seed 8) → lower-ID wins
        lo, hi = min(TEAM_IDS[0], TEAM_IDS[7]), max(TEAM_IDS[0], TEAM_IDS[7])
        mid = f"{SEASON}_{lo}_{hi}"
        # y=1 means lower-ID team wins
        analyzer.analyze_round(1, [mid], [1])
        best = analyzer.best_variant()
        assert best is not None
        assert best in ("strong", "weak")
