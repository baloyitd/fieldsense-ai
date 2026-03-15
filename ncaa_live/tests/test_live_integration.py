"""
test_live_integration.py
========================
Integration tests for the full ncaa_live Stage 08 pipeline.

Tests cover:
- Tournament simulation round-by-round with live recalibration
- Rollback scenario: upset round triggers recalibration, rollback engages, recovery succeeds
- Submission regeneration: excludes completed matchups, Kaggle-compliant
- End-to-end timing: full recalibration cycle < 30 minutes
- Public API completeness
"""

from __future__ import annotations

import copy
import time

import numpy as np
import pytest

import ncaa_live
from ncaa_live import (
    BoxScoreIngester,
    BracketTeam,
    FallbackManager,
    GameResult,
    PerformanceMonitor,
    RoundRecalibrator,
    SubmissionRegenerator,
    TournamentSimulator,
    brier_score,
    make_simulator_from_feature_df,
    should_rollback,
)
from ncaa_models.baseline import FEATURE_COLS, LogisticBaseline
from ncaa_models.submit import make_submission_id

from .conftest import SEASON, TEAM_IDS


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class TestPublicAPI:
    def test_stage_label(self):
        assert ncaa_live.__stage__ == "08/10"

    def test_game_result_exported(self):
        assert hasattr(ncaa_live, "GameResult")

    def test_box_score_ingester_exported(self):
        assert hasattr(ncaa_live, "BoxScoreIngester")

    def test_round_recalibrator_exported(self):
        assert hasattr(ncaa_live, "RoundRecalibrator")

    def test_performance_monitor_exported(self):
        assert hasattr(ncaa_live, "PerformanceMonitor")

    def test_fallback_manager_exported(self):
        assert hasattr(ncaa_live, "FallbackManager")

    def test_should_rollback_exported(self):
        assert hasattr(ncaa_live, "should_rollback")

    def test_submission_regenerator_exported(self):
        assert hasattr(ncaa_live, "SubmissionRegenerator")

    def test_tournament_simulator_exported(self):
        assert hasattr(ncaa_live, "TournamentSimulator")

    def test_brier_score_exported(self):
        assert hasattr(ncaa_live, "brier_score")


# ---------------------------------------------------------------------------
# TournamentSimulator unit tests
# ---------------------------------------------------------------------------

class TestTournamentSimulator:
    def test_simulate_round_returns_correct_game_count(self, fresh_simulator):
        results = fresh_simulator.simulate_round()
        assert len(results) == len(TEAM_IDS) // 2  # 4 games from 8 teams

    def test_simulate_round_returns_game_results(self, fresh_simulator):
        results = fresh_simulator.simulate_round()
        assert all(isinstance(r, GameResult) for r in results)

    def test_simulate_round_winners_advance(self, fresh_simulator):
        results = fresh_simulator.simulate_round()
        active = set(fresh_simulator.active_team_ids())
        for r in results:
            assert r.winner_id in active

    def test_deterministic_lower_seed_wins(self, fresh_simulator):
        """In deterministic mode, lower seed always wins."""
        seeds = {t.team_id: t.seed for t in fresh_simulator.teams}
        results = fresh_simulator.simulate_round()
        for r in results:
            s1 = seeds[r.team1_id]
            s2 = seeds[r.team2_id]
            if s1 < s2:
                assert r.winner_id == r.team1_id
            elif s2 < s1:
                assert r.winner_id == r.team2_id

    def test_simulate_all_rounds_completes_tournament(self, fresh_simulator):
        fresh_simulator.simulate_all_rounds()
        assert fresh_simulator.is_complete()
        assert fresh_simulator.champion_id() is not None

    def test_n_rounds_for_8_team_bracket(self, fresh_simulator):
        """8-team bracket = 3 rounds."""
        fresh_simulator.simulate_all_rounds()
        assert fresh_simulator.n_rounds_completed() == 3

    def test_get_all_results_flat_total_games(self, fresh_simulator):
        """8-team bracket = 7 games total."""
        fresh_simulator.simulate_all_rounds()
        all_results = fresh_simulator.get_all_results_flat()
        assert len(all_results) == 7

    def test_simulate_round_raises_when_complete(self, fresh_simulator):
        fresh_simulator.simulate_all_rounds()
        with pytest.raises(RuntimeError):
            fresh_simulator.simulate_round()

    def test_upset_round_higher_seed_wins(self, feat_df):
        teams = [
            BracketTeam(team_id=tid, seed=rank + 1)
            for rank, tid in enumerate(TEAM_IDS)
        ]
        sim = TournamentSimulator(teams, feat_df, season=SEASON, deterministic=True)
        seeds = {t.team_id: t.seed for t in sim.teams}
        results = sim.simulate_round_upset()
        for r in results:
            s1 = seeds[r.team1_id]
            s2 = seeds[r.team2_id]
            # Higher seed (worse team) should win
            expected_winner = r.team1_id if s1 > s2 else r.team2_id
            assert r.winner_id == expected_winner

    def test_make_simulator_from_feature_df(self, feat_df):
        sim = make_simulator_from_feature_df(
            feat_df, TEAM_IDS, season=SEASON, deterministic=True
        )
        assert isinstance(sim, TournamentSimulator)
        assert len(sim.teams) == len(TEAM_IDS)

    def test_invalid_team_count_raises(self, feat_df):
        teams = [BracketTeam(team_id=i, seed=i) for i in range(1, 6)]
        with pytest.raises(ValueError):
            TournamentSimulator(teams, feat_df, season=SEASON)


# ---------------------------------------------------------------------------
# Full round-by-round tournament pipeline
# ---------------------------------------------------------------------------

class TestRoundByRoundPipeline:
    def _build_pipeline(self, feat_df, fitted_model, val_arrays, fresh_simulator):
        ingester = BoxScoreIngester(feat_df, feature_cols=FEATURE_COLS)
        recal = RoundRecalibrator(base_model=fitted_model, feature_cols=FEATURE_COLS)
        monitor = PerformanceMonitor(threshold=0.005)
        fallback = FallbackManager()
        X1_v, X2_v, y_v = val_arrays
        return ingester, recal, monitor, fallback

    def test_round1_ingestion_and_recalibration(
        self, feat_df, fitted_model, train_arrays, val_arrays, fresh_simulator
    ):
        ingester, recal, monitor, fallback = self._build_pipeline(
            feat_df, fitted_model, val_arrays, fresh_simulator
        )
        X1_tr, X2_tr, y_tr = train_arrays
        X1_v, X2_v, y_v = val_arrays

        # Baseline
        baseline_brier = brier_score(fitted_model.predict_proba(X1_v, X2_v), y_v)
        fallback.save_version(fitted_model, brier=baseline_brier, round_num=0)
        monitor.record(baseline_brier, round_num=0, label="baseline")

        # Round 1
        r1_results = fresh_simulator.simulate_round()
        ingester.ingest_batch(r1_results)
        updated = ingester.get_updated_features()
        X1_t, X2_t, y_t = ingester.build_training_arrays(season=SEASON)

        assert len(y_t) == len(r1_results)

        version = recal.recalibrate(
            results_X1=X1_t, results_X2=X2_t, results_y=y_t,
            X1_val=X1_v, X2_val=X2_v, y_val=y_v,
            X1_hist=X1_tr, X2_hist=X2_tr, y_hist=y_tr,
            round_num=1,
        )
        monitor.record(version.brier, round_num=1, label="round1")
        fallback.save_version(version.model, brier=version.brier, round_num=1)

        assert np.isfinite(version.brier)
        assert fallback.n_versions() == 2

    def test_brier_trajectory_is_tracked(
        self, feat_df, fitted_model, train_arrays, val_arrays
    ):
        """After 3 rounds, monitor should have 4 records (baseline + 3 rounds)."""
        teams = [
            BracketTeam(team_id=tid, seed=rank + 1)
            for rank, tid in enumerate(TEAM_IDS)
        ]
        sim = TournamentSimulator(teams, feat_df, season=SEASON, deterministic=True)
        ingester = BoxScoreIngester(feat_df, feature_cols=FEATURE_COLS)
        recal = RoundRecalibrator(base_model=fitted_model, feature_cols=FEATURE_COLS)
        monitor = PerformanceMonitor()
        X1_tr, X2_tr, y_tr = train_arrays
        X1_v, X2_v, y_v = val_arrays

        baseline_brier = brier_score(fitted_model.predict_proba(X1_v, X2_v), y_v)
        monitor.record(baseline_brier, round_num=0, label="baseline")

        for rnd in range(1, 4):  # 3 rounds for 8-team bracket
            results = sim.simulate_round()
            ingester.ingest_batch(results)
            X1_t, X2_t, y_t = ingester.build_training_arrays(season=SEASON)
            if len(y_t) == 0:
                continue
            version = recal.recalibrate(
                results_X1=X1_t, results_X2=X2_t, results_y=y_t,
                X1_val=X1_v, X2_val=X2_v, y_val=y_v,
                X1_hist=X1_tr, X2_hist=X2_tr, y_hist=y_tr,
                round_num=rnd,
            )
            monitor.record(version.brier, round_num=rnd, label=f"round{rnd}")

        # 4 records: baseline + 3 rounds
        assert len(monitor.history()) == 4
        assert monitor.best_brier() is not None


# ---------------------------------------------------------------------------
# Rollback integration scenario
# ---------------------------------------------------------------------------

class TestRollbackIntegration:
    """
    Confirmed rollback+recovery scenario (spec requirement).

    Steps:
    1. Train baseline → good Brier
    2. Simulate extreme-upset Round of 32 → inject bad data
    3. Recalibrate on bad data → higher Brier
    4. Monitor triggers rollback
    5. FallbackManager reverts to baseline
    6. Recovered Brier < bad Brier (recovery confirmed)
    """

    def test_full_rollback_and_recovery(
        self, feat_df, fitted_model, train_arrays, val_arrays
    ):
        X1_tr, X2_tr, y_tr = train_arrays
        X1_v, X2_v, y_v = val_arrays

        monitor = PerformanceMonitor(threshold=0.005)
        fallback = FallbackManager()
        recal = RoundRecalibrator(base_model=fitted_model, feature_cols=FEATURE_COLS)

        # Step 1: baseline
        baseline_brier = brier_score(fitted_model.predict_proba(X1_v, X2_v), y_v)
        fallback.save_version(fitted_model, brier=baseline_brier, round_num=0, label="baseline")
        monitor.record(baseline_brier, round_num=0, label="baseline")

        # Step 2: bad data — all upsets (inverted labels relative to training)
        # Use the entire training set with flipped outcomes
        bad_X1 = X1_tr.copy()
        bad_X2 = X2_tr.copy()
        bad_y = 1 - y_tr  # all upsets

        # Step 3: recalibrate on bad data (NO historical data to dilute the damage)
        bad_version = recal.recalibrate(
            results_X1=bad_X1, results_X2=bad_X2, results_y=bad_y,
            X1_val=X1_v, X2_val=X2_v, y_val=y_v,
            round_num=1,
        )

        # Step 4: save bad version then monitor decides to rollback
        fallback.save_version(
            bad_version.model, brier=bad_version.brier, round_num=1, label="bad_round"
        )
        do_rollback = monitor.should_rollback(baseline_brier, bad_version.brier)
        assert do_rollback is True, (
            f"Expected rollback: baseline={baseline_brier:.4f}, "
            f"bad={bad_version.brier:.4f}, diff={bad_version.brier - baseline_brier:.4f}"
        )

        monitor.record(
            bad_version.brier, round_num=1, label="bad_round", triggered_rollback=True
        )

        # Step 5: revert
        recovered = fallback.rollback()
        assert recovered is not None, "Rollback should return previous version"
        assert recovered["label"] == "baseline"

        # Step 6: confirmed recovery
        recovered_brier = brier_score(
            recovered["model"].predict_proba(X1_v, X2_v), y_v
        )
        assert recovered_brier < bad_version.brier, (
            f"Recovered Brier {recovered_brier:.4f} should be < bad Brier {bad_version.brier:.4f}"
        )
        assert abs(recovered_brier - baseline_brier) < 1e-9
        assert monitor.n_rollbacks() == 1

    def test_rollback_does_not_engage_on_improvement(
        self, fitted_model, train_arrays, val_arrays
    ):
        """When recalibration helps, no rollback should fire."""
        X1_tr, X2_tr, y_tr = train_arrays
        X1_v, X2_v, y_v = val_arrays
        monitor = PerformanceMonitor(threshold=0.005)

        baseline_brier = brier_score(fitted_model.predict_proba(X1_v, X2_v), y_v)
        # Recalibrate on correct-label data (same signal as training)
        recal = RoundRecalibrator(base_model=fitted_model)
        new_version = recal.recalibrate(
            results_X1=X1_tr, results_X2=X2_tr, results_y=y_tr,
            X1_val=X1_v, X2_val=X2_v, y_val=y_v,
            X1_hist=X1_tr, X2_hist=X2_tr, y_hist=y_tr,
            round_num=1,
        )
        do_rollback = monitor.should_rollback(baseline_brier, new_version.brier)
        # Should NOT rollback (new Brier within threshold of baseline)
        assert do_rollback is False or new_version.brier > baseline_brier + 0.005


# ---------------------------------------------------------------------------
# SubmissionRegenerator
# ---------------------------------------------------------------------------

class TestSubmissionRegenerator:
    def test_regenerate_returns_dataframe(self, feat_df, fitted_model):
        regen = SubmissionRegenerator(
            model=fitted_model,
            feature_store=feat_df,
            feature_cols=FEATURE_COLS,
            season=SEASON,
        )
        sub = regen.regenerate(
            completed_game_ids=[],
            eligible_team_ids=TEAM_IDS,
        )
        import pandas as pd
        assert isinstance(sub, pd.DataFrame)
        assert "ID" in sub.columns
        assert "Pred" in sub.columns

    def test_regenerate_excludes_completed_matchups(self, feat_df, fitted_model):
        regen = SubmissionRegenerator(
            model=fitted_model,
            feature_store=feat_df,
            feature_cols=FEATURE_COLS,
            season=SEASON,
        )
        # Full submission
        full_sub = regen.regenerate(completed_game_ids=[], eligible_team_ids=TEAM_IDS)
        n_full = len(full_sub)

        # Exclude 2 games
        completed = set(full_sub["ID"].values[:2])
        filtered_sub = regen.regenerate(
            completed_game_ids=completed,
            eligible_team_ids=TEAM_IDS,
        )
        assert len(filtered_sub) == n_full - 2

    def test_regenerate_probs_in_valid_range(self, feat_df, fitted_model):
        regen = SubmissionRegenerator(
            model=fitted_model,
            feature_store=feat_df,
            feature_cols=FEATURE_COLS,
            season=SEASON,
        )
        sub = regen.regenerate(completed_game_ids=[], eligible_team_ids=TEAM_IDS)
        assert (sub["Pred"] >= 0.01).all()
        assert (sub["Pred"] <= 0.99).all()

    def test_regenerate_no_duplicate_ids(self, feat_df, fitted_model):
        regen = SubmissionRegenerator(
            model=fitted_model,
            feature_store=feat_df,
            feature_cols=FEATURE_COLS,
            season=SEASON,
        )
        sub = regen.regenerate(completed_game_ids=[], eligible_team_ids=TEAM_IDS)
        assert sub["ID"].nunique() == len(sub)

    def test_regenerate_correct_kaggle_id_format(self, feat_df, fitted_model):
        regen = SubmissionRegenerator(
            model=fitted_model,
            feature_store=feat_df,
            feature_cols=FEATURE_COLS,
            season=SEASON,
        )
        sub = regen.regenerate(completed_game_ids=[], eligible_team_ids=TEAM_IDS)
        for sid in sub["ID"]:
            parts = sid.split("_")
            assert len(parts) == 3
            assert parts[0] == str(SEASON)
            assert int(parts[1]) < int(parts[2])

    def test_update_model_swaps_model(self, feat_df, fitted_model, train_arrays, val_arrays):
        X1_tr, X2_tr, y_tr = train_arrays
        regen = SubmissionRegenerator(
            model=fitted_model,
            feature_store=feat_df,
            feature_cols=FEATURE_COLS,
            season=SEASON,
        )
        new_model = LogisticBaseline()
        new_model.fit(X1_tr, X2_tr, y_tr)
        regen.update_model(new_model)
        assert regen.model is new_model

    def test_update_feature_store(self, feat_df, fitted_model):
        regen = SubmissionRegenerator(
            model=fitted_model,
            feature_store=feat_df,
            feature_cols=FEATURE_COLS,
            season=SEASON,
        )
        ingester = BoxScoreIngester(feat_df, feature_cols=FEATURE_COLS)
        r = GameResult("g1", SEASON, 1, TEAM_IDS[0], TEAM_IDS[7], 80, 60)
        ingester.ingest(r)
        regen.update_feature_store(ingester.get_updated_features())
        # Should still be able to generate a submission
        sub = regen.regenerate(completed_game_ids=[], eligible_team_ids=TEAM_IDS)
        assert len(sub) > 0

    def test_validate_returns_valid_result(self, feat_df, fitted_model):
        regen = SubmissionRegenerator(
            model=fitted_model,
            feature_store=feat_df,
            feature_cols=FEATURE_COLS,
            season=SEASON,
        )
        sub = regen.regenerate(completed_game_ids=[], eligible_team_ids=TEAM_IDS)
        result = regen.validate(sub)
        assert result["valid"] is True

    def test_csv_written_to_disk(self, feat_df, fitted_model, tmp_path):
        regen = SubmissionRegenerator(
            model=fitted_model,
            feature_store=feat_df,
            feature_cols=FEATURE_COLS,
            season=SEASON,
        )
        output = tmp_path / "submission.csv"
        regen.regenerate(
            completed_game_ids=[],
            eligible_team_ids=TEAM_IDS,
            output_path=output,
        )
        assert output.exists()
        import pandas as pd
        df = pd.read_csv(output)
        assert "ID" in df.columns and "Pred" in df.columns


# ---------------------------------------------------------------------------
# End-to-end timing
# ---------------------------------------------------------------------------

class TestEndToEndTiming:
    def test_ingestion_plus_recalibration_within_30_minutes(
        self, feat_df, fitted_model, train_arrays, val_arrays
    ):
        """
        Full cycle: game result ingestion → recalibration → new submission.
        Must complete in < 1800 seconds (30 minutes).
        """
        X1_tr, X2_tr, y_tr = train_arrays
        X1_v, X2_v, y_v = val_arrays

        t0 = time.time()

        # Ingest 4 games
        ingester = BoxScoreIngester(feat_df, feature_cols=FEATURE_COLS)
        for i in range(4):
            r = GameResult(
                game_id=f"g{i}", season=SEASON, round_num=1,
                team1_id=TEAM_IDS[i], team2_id=TEAM_IDS[7 - i],
                team1_score=70 + i, team2_score=60,
            )
            ingester.ingest(r)

        # Recalibrate
        X1_t, X2_t, y_t = ingester.build_training_arrays(season=SEASON)
        recal = RoundRecalibrator(base_model=fitted_model, feature_cols=FEATURE_COLS)
        version = recal.recalibrate(
            results_X1=X1_t, results_X2=X2_t, results_y=y_t,
            X1_val=X1_v, X2_val=X2_v, y_val=y_v,
            X1_hist=X1_tr, X2_hist=X2_tr, y_hist=y_tr,
            round_num=1,
        )

        # Regenerate submission
        regen = SubmissionRegenerator(
            model=version.model,
            feature_store=ingester.get_updated_features(),
            feature_cols=FEATURE_COLS,
            season=SEASON,
        )
        sub = regen.regenerate(
            completed_game_ids=ingester.completed_matchup_ids(),
            eligible_team_ids=TEAM_IDS,
        )

        elapsed = time.time() - t0
        assert elapsed < RoundRecalibrator.MAX_RECAL_SECONDS, (
            f"Pipeline took {elapsed:.1f}s, limit is {RoundRecalibrator.MAX_RECAL_SECONDS}s"
        )
        assert len(sub) > 0
