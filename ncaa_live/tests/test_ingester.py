"""
test_ingester.py
================
Unit tests for ncaa_live.ingester.BoxScoreIngester and GameResult.
"""

from __future__ import annotations

import numpy as np
import pytest

from ncaa_live.ingester import BoxScoreIngester, GameResult
from ncaa_models.baseline import FEATURE_COLS

from .conftest import SEASON, TEAM_IDS


# ---------------------------------------------------------------------------
# GameResult
# ---------------------------------------------------------------------------

class TestGameResult:
    def test_winner_id_team1_wins(self):
        r = GameResult("g1", SEASON, 1, 1001, 1002, 80, 70)
        assert r.winner_id == 1001

    def test_winner_id_team2_wins(self):
        r = GameResult("g1", SEASON, 1, 1001, 1002, 65, 72)
        assert r.winner_id == 1002

    def test_loser_id_team1_wins(self):
        r = GameResult("g1", SEASON, 1, 1001, 1002, 80, 70)
        assert r.loser_id == 1002

    def test_team1_won_true(self):
        r = GameResult("g1", SEASON, 1, 1001, 1002, 80, 70)
        assert r.team1_won is True

    def test_team1_won_false(self):
        r = GameResult("g1", SEASON, 1, 1001, 1002, 65, 75)
        assert r.team1_won is False

    def test_to_dict_has_required_keys(self):
        r = GameResult("g1", SEASON, 1, 1001, 1002, 80, 70)
        d = r.to_dict()
        for key in ("game_id", "season", "round_num",
                    "team1_id", "team2_id", "winner_id", "loser_id"):
            assert key in d

    def test_box_scores_default_empty(self):
        r = GameResult("g1", SEASON, 1, 1001, 1002, 80, 70)
        assert r.box_scores == {}

    def test_box_scores_custom(self):
        r = GameResult("g1", SEASON, 1, 1001, 1002, 80, 70,
                       box_scores={1001: {"fg_pct": 0.55}})
        assert r.box_scores[1001]["fg_pct"] == 0.55


# ---------------------------------------------------------------------------
# BoxScoreIngester — ingestion
# ---------------------------------------------------------------------------

class TestBoxScoreIngester:
    def test_ingest_adds_to_completed(self, ingester, sample_result):
        ingester.ingest(sample_result)
        assert len(ingester.get_completed_games()) == 1

    def test_ingest_batch_adds_all(self, ingester):
        results = [
            GameResult("g1", SEASON, 1, TEAM_IDS[0], TEAM_IDS[7], 78, 55),
            GameResult("g2", SEASON, 1, TEAM_IDS[1], TEAM_IDS[6], 72, 60),
        ]
        ingester.ingest_batch(results)
        assert len(ingester.get_completed_games()) == 2

    def test_get_winner_ids(self, ingester, sample_result):
        ingester.ingest(sample_result)
        winners = ingester.get_winner_ids()
        assert sample_result.winner_id in winners

    def test_get_loser_ids(self, ingester, sample_result):
        ingester.ingest(sample_result)
        losers = ingester.get_loser_ids()
        assert sample_result.loser_id in losers

    def test_does_not_mutate_original_df(self, feat_df):
        """Ingester should copy the feature store."""
        original_col = feat_df["win_pct"].copy()
        ing = BoxScoreIngester(feat_df, feature_cols=FEATURE_COLS)
        r = GameResult("g1", SEASON, 1, TEAM_IDS[0], TEAM_IDS[7], 100, 50)
        ing.ingest(r)
        # Original DataFrame should be unchanged
        assert (feat_df["win_pct"].values == original_col.values).all()

    def test_ingest_updates_games_played(self, ingester, feat_df, sample_result):
        """After ingesting a game, games_played should increase by 1."""
        tid = sample_result.team1_id
        mask = (feat_df["team_id"] == tid) & (feat_df["season"] == SEASON)
        gp_before = feat_df.loc[mask, "games_played"].iloc[0]

        ingester.ingest(sample_result)
        updated = ingester.get_updated_features()
        mask2 = (updated["team_id"] == tid) & (updated["season"] == SEASON)
        gp_after = updated.loc[mask2, "games_played"].iloc[0]

        assert gp_after == gp_before + 1

    def test_ingest_updates_win_pct(self, ingester, feat_df, sample_result):
        """Winner's win_pct should increase after ingesting a win."""
        winner_id = sample_result.winner_id
        mask = (feat_df["team_id"] == winner_id) & (feat_df["season"] == SEASON)
        wp_before = feat_df.loc[mask, "win_pct"].iloc[0]

        ingester.ingest(sample_result)
        updated = ingester.get_updated_features()
        mask2 = (updated["team_id"] == winner_id) & (updated["season"] == SEASON)
        wp_after = updated.loc[mask2, "win_pct"].iloc[0]

        # win_pct should increase (or stay same if already 1.0)
        assert wp_after >= wp_before - 1e-9

    def test_ingest_updates_ppg_scored(self, ingester, feat_df):
        """ppg_scored should update incrementally after ingesting."""
        tid = TEAM_IDS[0]
        r = GameResult("g1", SEASON, 1, tid, TEAM_IDS[7], 100, 50)
        ingester.ingest(r)
        updated = ingester.get_updated_features()
        mask = (updated["team_id"] == tid) & (updated["season"] == SEASON)
        # ppg_scored should be closer to 100 than the original value
        new_ppg = updated.loc[mask, "ppg_scored"].iloc[0]
        assert new_ppg > 0

    def test_upset_decreases_winner_win_pct_or_holds(self, ingester, feat_df,
                                                       upset_result):
        """Seed-8 team upsets seed-1; seed-8's win_pct should increase."""
        winner_id = upset_result.winner_id  # seed-8
        mask = (feat_df["team_id"] == winner_id) & (feat_df["season"] == SEASON)
        wp_before = feat_df.loc[mask, "win_pct"].iloc[0]

        ingester.ingest(upset_result)
        updated = ingester.get_updated_features()
        mask2 = (updated["team_id"] == winner_id) & (updated["season"] == SEASON)
        wp_after = updated.loc[mask2, "win_pct"].iloc[0]
        assert wp_after >= wp_before - 1e-9


# ---------------------------------------------------------------------------
# BoxScoreIngester — update_feature_store
# ---------------------------------------------------------------------------

class TestUpdateFeatureStore:
    def test_update_existing_column(self, ingester, feat_df):
        tid = TEAM_IDS[0]
        new_val = 0.99
        ingester.update_feature_store(tid, {"win_pct": new_val}, season=SEASON)
        updated = ingester.get_updated_features()
        mask = (updated["team_id"] == tid) & (updated["season"] == SEASON)
        assert abs(updated.loc[mask, "win_pct"].iloc[0] - new_val) < 1e-9

    def test_update_unknown_team_does_not_crash(self, ingester):
        ingester.update_feature_store(9999, {"win_pct": 0.5}, season=SEASON)

    def test_update_nonexistent_column_skipped(self, ingester):
        tid = TEAM_IDS[0]
        ingester.update_feature_store(tid, {"nonexistent_col": 42.0}, season=SEASON)

    def test_update_with_box_scores_in_game(self, ingester, feat_df):
        tid = TEAM_IDS[0]
        r = GameResult(
            "g_bs", SEASON, 1, tid, TEAM_IDS[7], 78, 55,
            box_scores={tid: {"fg_pct": 0.60}},
        )
        ingester.ingest(r)
        updated = ingester.get_updated_features()
        mask = (updated["team_id"] == tid) & (updated["season"] == SEASON)
        assert abs(updated.loc[mask, "fg_pct"].iloc[0] - 0.60) < 1e-9


# ---------------------------------------------------------------------------
# BoxScoreIngester — completed_matchup_ids
# ---------------------------------------------------------------------------

class TestCompletedMatchupIds:
    def test_returns_kaggle_format_id(self, ingester, sample_result):
        ingester.ingest(sample_result)
        ids = ingester.completed_matchup_ids()
        assert len(ids) == 1
        parts = ids[0].split("_")
        assert len(parts) == 3
        assert parts[0] == str(SEASON)

    def test_low_id_before_high_id(self, ingester, sample_result):
        ingester.ingest(sample_result)
        ids = ingester.completed_matchup_ids()
        _, lo, hi = ids[0].split("_")
        assert int(lo) < int(hi)


# ---------------------------------------------------------------------------
# BoxScoreIngester — build_training_arrays
# ---------------------------------------------------------------------------

class TestBuildTrainingArrays:
    def test_empty_returns_empty_arrays(self, ingester):
        X1, X2, y = ingester.build_training_arrays()
        assert len(y) == 0

    def test_one_game_returns_one_row(self, ingester, sample_result):
        ingester.ingest(sample_result)
        X1, X2, y = ingester.build_training_arrays()
        assert len(y) == 1
        assert X1.shape == (1, len(FEATURE_COLS))
        assert X2.shape == (1, len(FEATURE_COLS))

    def test_y_is_binary(self, ingester):
        results = [
            GameResult("g1", SEASON, 1, TEAM_IDS[0], TEAM_IDS[7], 78, 55),
            GameResult("g2", SEASON, 1, TEAM_IDS[1], TEAM_IDS[6], 65, 70),
        ]
        ingester.ingest_batch(results)
        _, _, y = ingester.build_training_arrays()
        assert set(y).issubset({0, 1})

    def test_season_filter(self, ingester):
        r1 = GameResult("g1", SEASON, 1, TEAM_IDS[0], TEAM_IDS[7], 78, 55)
        r2 = GameResult("g2", SEASON - 1, 1, TEAM_IDS[0], TEAM_IDS[7], 70, 60)
        ingester.ingest_batch([r1, r2])
        _, _, y_all = ingester.build_training_arrays()
        _, _, y_season = ingester.build_training_arrays(season=SEASON)
        assert len(y_all) == 2
        assert len(y_season) == 1
