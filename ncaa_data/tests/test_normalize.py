"""
test_normalize.py
=================
Tests for ncaa_data.normalize:
  1. Every raw team ID maps to exactly one canonical identifier (no orphans).
  2. Canonical IDs are collision-free across genders.
  3. Seed parsing, conference encoding, game flattening.
  4. normalize_all produces a unified DataFrame with required columns.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ncaa_data.normalize import (
    ConferenceRegistry,
    flatten_games,
    make_canonical_id,
    normalize_all,
    parse_seed_number,
)
from ncaa_data.tests.conftest import TEAM_IDS, SEASONS, _compact, _detailed


# ---------------------------------------------------------------------------
# Canonical ID helpers
# ---------------------------------------------------------------------------

class TestCanonicalId:

    def test_format(self):
        assert make_canonical_id("M", 1234) == "M_1234"
        assert make_canonical_id("W", 5678) == "W_5678"

    def test_gender_uppercase(self):
        assert make_canonical_id("m", 1001) == "M_1001"

    def test_no_collision_across_genders(self):
        """Same integer team_id must produce different IDs for M vs W."""
        for tid in TEAM_IDS:
            assert make_canonical_id("M", tid) != make_canonical_id("W", tid)

    def test_no_orphaned_records(self, raw_data):
        """
        Every WTeamID / LTeamID in the results DataFrames must be representable
        as a valid canonical_id — i.e. not null or empty.
        """
        for gender in ("M", "W"):
            df = raw_data[f"{gender}_regular_compact"]
            for col in ("WTeamID", "LTeamID"):
                cids = df[col].apply(lambda x: make_canonical_id(gender, x))
                assert cids.notna().all()
                assert (cids.str.len() > 0).all()


# ---------------------------------------------------------------------------
# Seed parsing
# ---------------------------------------------------------------------------

class TestSeedParsing:

    @pytest.mark.parametrize("seed_str,expected", [
        ("W01",   1),
        ("X16",  16),
        ("Z08a",  8),
        ("Y04b",  4),
        ("W01a",  1),
    ])
    def test_known_seeds(self, seed_str, expected):
        assert parse_seed_number(seed_str) == expected

    @pytest.mark.parametrize("bad", ["", "abc", None])
    def test_bad_seed_returns_17(self, bad):
        assert parse_seed_number(bad) == 17


# ---------------------------------------------------------------------------
# Conference registry
# ---------------------------------------------------------------------------

class TestConferenceRegistry:

    def test_unknown_is_zero(self):
        reg = ConferenceRegistry()
        assert reg.encode("Unknown") == 0
        assert reg.encode(None) == 0

    def test_distinct_conferences_get_distinct_ids(self):
        reg = ConferenceRegistry()
        ids = [reg.encode(c) for c in ("ACC", "B10", "SEC", "B12", "PAC")]
        assert len(set(ids)) == len(ids)

    def test_same_conference_stable_id(self):
        reg = ConferenceRegistry()
        acc1 = reg.encode("ACC")
        acc2 = reg.encode("ACC")
        assert acc1 == acc2

    def test_encode_series(self):
        reg = ConferenceRegistry()
        s = pd.Series(["ACC", "B10", None, "ACC"])
        encoded = reg.encode_series(s)
        assert encoded[0] == encoded[3]   # same conference → same id
        assert encoded[2] == 0            # None → Unknown → 0


# ---------------------------------------------------------------------------
# Game flattening
# ---------------------------------------------------------------------------

class TestFlattenGames:

    def test_doubles_rows(self, raw_data):
        df = raw_data["M_regular_compact"]
        flat = flatten_games(df, "M")
        assert len(flat) == 2 * len(df)

    def test_required_columns_present(self, raw_data):
        flat = flatten_games(raw_data["M_regular_compact"], "M")
        for col in (
            "season", "day_num", "gender", "team_id", "opp_team_id",
            "canonical_id", "opp_canonical_id",
            "score", "opp_score", "margin", "won", "loc", "num_ot", "is_tourney",
        ):
            assert col in flat.columns, f"Missing: {col}"

    def test_no_null_canonical_ids(self, raw_data):
        flat = flatten_games(raw_data["M_regular_compact"], "M")
        assert flat["canonical_id"].notna().all()
        assert flat["opp_canonical_id"].notna().all()

    def test_winner_margin_positive(self, raw_data):
        flat = flatten_games(raw_data["M_regular_compact"], "M")
        winners = flat[flat["won"]]
        assert (winners["margin"] > 0).all()

    def test_loser_margin_negative(self, raw_data):
        flat = flatten_games(raw_data["M_regular_compact"], "M")
        losers = flat[~flat["won"]]
        assert (losers["margin"] < 0).all()

    def test_location_flip_home_to_away(self):
        """W location 'H' → loser sees 'A'."""
        src = pd.DataFrame([_compact(2023, 10, 1001, 75, 1002, 66, "H")])
        flat = flatten_games(src, "M")
        winner = flat[flat["team_id"] == 1001].iloc[0]
        loser  = flat[flat["team_id"] == 1002].iloc[0]
        assert winner["loc"] == "H"
        assert loser["loc"]  == "A"

    def test_location_neutral_unchanged(self):
        src = pd.DataFrame([_compact(2023, 10, 1001, 75, 1002, 66, "N")])
        flat = flatten_games(src, "M")
        for _, row in flat.iterrows():
            assert row["loc"] == "N"

    def test_detailed_stats_attached(self, raw_data):
        flat = flatten_games(raw_data["M_regular_detailed"], "M")
        for col in ("team_fgm", "team_fga", "opp_fgm", "opp_fga",
                    "team_or", "team_dr"):
            assert col in flat.columns, f"Missing detailed col: {col}"

    def test_tourney_flag(self):
        src = pd.DataFrame([_compact(2023, 145, 1001, 80, 1002, 70)])
        flat_t = flatten_games(src, "M", is_tourney=True)
        flat_r = flatten_games(src, "M", is_tourney=False)
        assert flat_t["is_tourney"].all()
        assert not flat_r["is_tourney"].any()

    def test_symmetric_scores(self):
        """Winner's score == loser's opp_score and vice versa."""
        src = pd.DataFrame([_compact(2023, 10, 1001, 75, 1002, 66)])
        flat = flatten_games(src, "M")
        w = flat[flat["team_id"] == 1001].iloc[0]
        l = flat[flat["team_id"] == 1002].iloc[0]
        assert w["score"] == l["opp_score"] == 75
        assert l["score"] == w["opp_score"] == 66


# ---------------------------------------------------------------------------
# normalize_all (unified pass)
# ---------------------------------------------------------------------------

class TestNormalizeAll:

    def test_returns_dataframe(self, games_df):
        assert isinstance(games_df, pd.DataFrame)

    def test_both_genders_present(self, games_df):
        assert set(games_df["gender"].unique()) == {"M", "W"}

    def test_all_seasons_present(self, games_df):
        assert set(SEASONS).issubset(set(games_df["season"].unique()))

    def test_no_null_canonical_ids(self, games_df):
        assert games_df["canonical_id"].notna().all()
        assert games_df["opp_canonical_id"].notna().all()

    def test_one_canonical_id_per_raw_team_id(self, games_df):
        """
        Each (gender, team_id) pair must map to exactly one canonical_id.
        This validates zero orphaned records.
        """
        mapping = (
            games_df
            .groupby(["gender", "team_id"])["canonical_id"]
            .nunique()
        )
        assert (mapping == 1).all(), "Some team_id maps to multiple canonical_ids"

    def test_seed_column_present(self, games_df):
        tourn = games_df[games_df["is_tourney"]]
        assert "seed" in tourn.columns

    def test_team_1001_seed_1(self, games_df):
        tourn_m = games_df[games_df["is_tourney"] & (games_df["gender"] == "M")]
        t1 = tourn_m[tourn_m["team_id"] == 1001]
        assert len(t1) > 0
        assert (t1["seed"] == 1).all()

    def test_conf_id_non_negative(self, games_df):
        if "conf_id" in games_df.columns:
            assert (games_df["conf_id"] >= 0).all()

    def test_tourney_and_regular_both_present(self, games_df):
        assert games_df["is_tourney"].any()
        assert (~games_df["is_tourney"]).any()

    def test_sorted_by_gender_season_day(self, games_df):
        """normalize_all should return rows sorted by (gender, season, day_num)."""
        df = games_df.reset_index(drop=True)
        expected = df.sort_values(
            ["gender", "season", "day_num"]
        ).reset_index(drop=True)
        pd.testing.assert_frame_equal(df, expected, check_like=False)
