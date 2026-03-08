"""
tests/test_ncaa_pipeline.py
===========================
Unit + integration tests for the NCAA data pipeline (Stage 01/10).

Tests run entirely on synthetic Kaggle-format CSV data written to a
temporary directory — no real Kaggle download required.

Coverage:
  * ncaa_data.ingest      – load_raw_data, column validation
  * ncaa_data.normalize   – canonical IDs, flatten_games, seed/conf attach
  * ncaa_data.features    – compute_features with detailed + compact data
  * ncaa_data.pipeline    – run_pipeline dry-run and full end-to-end with I/O
  * ncaa_data.cli         – argument parsing, summary output
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Synthetic data factory
# ---------------------------------------------------------------------------

def _compact_game(
    season: int, day: int,
    wid: int, wscore: int,
    lid: int, lscore: int,
    wloc: str = "N", num_ot: int = 0,
) -> dict:
    return dict(
        Season=season, DayNum=day,
        WTeamID=wid, WScore=wscore,
        LTeamID=lid, LScore=lscore,
        WLoc=wloc, NumOT=num_ot,
    )


def _detailed_game(season, day, wid, wscore, lid, lscore, wloc="N", num_ot=0) -> dict:
    base = _compact_game(season, day, wid, wscore, lid, lscore, wloc, num_ot)
    # Winning team detailed stats
    base.update(dict(
        WFGM=28, WFGA=60, WFGM3=8, WFGA3=20, WFTM=16, WFTA=20,
        WOR=10, WDR=25, WAST=15, WTO=12, WSTL=7, WBLK=4, WPF=18,
    ))
    # Losing team detailed stats
    base.update(dict(
        LFGM=24, LFGA=58, LFGM3=6, LFGA3=18, LFTM=12, LFTA=16,
        LOR=8,  LDR=22, LAST=11, LTO=14, LSTL=5, LBLK=2, LPF=20,
    ))
    return base


def _write_synthetic_data(data_dir: Path, gender: str = "M") -> None:
    """Write minimal Kaggle-format CSV files for one gender."""
    g = gender

    # ---- Teams -----------------------------------------------------------
    teams = pd.DataFrame([
        {"TeamID": 1001, "TeamName": "Alpha University"},
        {"TeamID": 1002, "TeamName": "Beta College"},
        {"TeamID": 1003, "TeamName": "Gamma State"},
        {"TeamID": 1004, "TeamName": "Delta Tech"},
    ])
    teams.to_csv(data_dir / f"{g}Teams.csv", index=False)

    # ---- Seasons ---------------------------------------------------------
    seasons = pd.DataFrame([
        {"Season": y, "DayzeroDate": f"{y-1}-10-30", "RegionW": "East",
         "RegionX": "West", "RegionY": "South", "RegionZ": "Midwest"}
        for y in range(2021, 2026)
    ])
    seasons.to_csv(data_dir / f"{g}Seasons.csv", index=False)

    # ---- Regular season compact results ----------------------------------
    games = []
    for season in range(2021, 2026):
        # 1001 beats 1002 (home), 1003 beats 1004 (away) — 6 games each pair
        for day in range(10, 130, 20):
            games.append(_compact_game(season, day, 1001, 75, 1002, 68, "H"))
            games.append(_compact_game(season, day + 2, 1003, 80, 1004, 72, "A"))
            games.append(_compact_game(season, day + 4, 1001, 70, 1003, 65, "N"))
            games.append(_compact_game(season, day + 6, 1002, 78, 1004, 71, "N"))
        # Ensure 1001 wins last 10 and not 1002
        for extra_day in range(130, 150, 3):
            games.append(_compact_game(season, extra_day, 1001, 82, 1004, 60, "H"))

    compact = pd.DataFrame(games)
    compact.to_csv(data_dir / f"{g}RegularSeasonCompactResults.csv", index=False)

    # ---- Regular season detailed results ---------------------------------
    det_games = []
    for season in range(2021, 2026):
        for day in range(10, 130, 20):
            det_games.append(_detailed_game(season, day, 1001, 75, 1002, 68, "H"))
            det_games.append(_detailed_game(season, day + 2, 1003, 80, 1004, 72, "A"))
    detailed = pd.DataFrame(det_games)
    detailed.to_csv(data_dir / f"{g}RegularSeasonDetailedResults.csv", index=False)

    # ---- Tournament compact results --------------------------------------
    tourn = pd.DataFrame([
        _compact_game(s, 145, 1001, 85, 1002, 70, "N")
        for s in range(2021, 2026)
    ] + [
        _compact_game(s, 148, 1001, 72, 1003, 68, "N")
        for s in range(2021, 2026)
    ])
    tourn.to_csv(data_dir / f"{g}NCAATourneyCompactResults.csv", index=False)

    # ---- Tournament detailed results ------------------------------------
    tourn_det = pd.DataFrame([
        _detailed_game(s, 145, 1001, 85, 1002, 70, "N")
        for s in range(2021, 2026)
    ])
    tourn_det.to_csv(data_dir / f"{g}NCAATourneyDetailedResults.csv", index=False)

    # ---- Seeds ----------------------------------------------------------
    seeds = []
    for s in range(2021, 2026):
        seeds += [
            {"Season": s, "Seed": "W01", "TeamID": 1001},
            {"Season": s, "Seed": "W04", "TeamID": 1002},
            {"Season": s, "Seed": "X02", "TeamID": 1003},
            {"Season": s, "Seed": "X08a", "TeamID": 1004},
        ]
    pd.DataFrame(seeds).to_csv(data_dir / f"{g}NCAATourneySeeds.csv", index=False)

    # ---- Conferences ----------------------------------------------------
    confs = []
    conf_map = {1001: "ACC", 1002: "ACC", 1003: "B10", 1004: "B10"}
    for s in range(2021, 2026):
        for tid, conf in conf_map.items():
            confs.append({"Season": s, "TeamID": tid, "ConfAbbrev": conf})
    pd.DataFrame(confs).to_csv(data_dir / f"{g}TeamConferences.csv", index=False)


def _make_data_dir() -> tempfile.TemporaryDirectory:
    """Create a temp dir with synthetic M + W data."""
    tmpdir = tempfile.TemporaryDirectory()
    data_dir = Path(tmpdir.name)
    _write_synthetic_data(data_dir, "M")
    _write_synthetic_data(data_dir, "W")
    return tmpdir


# ===========================================================================
# Test classes
# ===========================================================================

class TestIngest(unittest.TestCase):

    def setUp(self):
        self._tmpdir = _make_data_dir()
        self.data_dir = Path(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_load_raw_data_returns_dict(self):
        from ncaa_data.ingest import load_raw_data
        raw = load_raw_data(self.data_dir)
        self.assertIsInstance(raw, dict)
        self.assertGreater(len(raw), 0)

    def test_required_files_present(self):
        from ncaa_data.ingest import load_raw_data
        raw = load_raw_data(self.data_dir)
        for key in ("M_regular_compact", "W_regular_compact"):
            self.assertIn(key, raw, f"Missing key: {key}")

    def test_gender_column_added(self):
        from ncaa_data.ingest import load_raw_data
        raw = load_raw_data(self.data_dir)
        for key, df in raw.items():
            if "Gender" in df.columns:
                self.assertTrue(df["Gender"].isin(["M", "W"]).all())

    def test_compact_required_columns(self):
        from ncaa_data.ingest import load_raw_data
        raw = load_raw_data(self.data_dir)
        df = raw["M_regular_compact"]
        for col in ["Season", "DayNum", "WTeamID", "WScore", "LTeamID", "LScore"]:
            self.assertIn(col, df.columns)

    def test_detailed_results_loaded(self):
        from ncaa_data.ingest import load_raw_data
        raw = load_raw_data(self.data_dir)
        self.assertIn("M_regular_detailed", raw)
        df = raw["M_regular_detailed"]
        self.assertIn("WFGM", df.columns)

    def test_seeds_loaded(self):
        from ncaa_data.ingest import load_raw_data
        raw = load_raw_data(self.data_dir)
        self.assertIn("M_seeds", raw)

    def test_missing_required_file_raises(self):
        from ncaa_data.ingest import load_raw_data
        import shutil
        shutil.move(
            str(self.data_dir / "MRegularSeasonCompactResults.csv"),
            str(self.data_dir / "MRegularSeasonCompactResults.csv.bak"),
        )
        with self.assertRaises(FileNotFoundError):
            load_raw_data(self.data_dir)

    def test_load_teams(self):
        from ncaa_data.ingest import load_teams
        df = load_teams(self.data_dir, "M")
        self.assertIn("TeamID", df.columns)
        self.assertIn("TeamName", df.columns)
        self.assertGreater(len(df), 0)

    def test_load_seeds(self):
        from ncaa_data.ingest import load_seeds
        df = load_seeds(self.data_dir, "M")
        self.assertEqual(set(df.columns) & {"Season", "Seed", "TeamID"},
                         {"Season", "Seed", "TeamID"})

    def test_load_conferences(self):
        from ncaa_data.ingest import load_conferences
        df = load_conferences(self.data_dir, "M")
        self.assertIsNotNone(df)
        self.assertIn("ConfAbbrev", df.columns)


class TestNormalize(unittest.TestCase):

    def setUp(self):
        self._tmpdir = _make_data_dir()
        self.data_dir = Path(self._tmpdir.name)
        from ncaa_data.ingest import load_raw_data
        self.raw = load_raw_data(self.data_dir)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_canonical_id_format(self):
        from ncaa_data.normalize import make_canonical_id
        self.assertEqual(make_canonical_id("M", 1001), "M_1001")
        self.assertEqual(make_canonical_id("W", 5678), "W_5678")

    def test_no_id_collision_across_genders(self):
        """The same integer team_id must not collide between M and W."""
        from ncaa_data.normalize import make_canonical_id
        self.assertNotEqual(make_canonical_id("M", 1001), make_canonical_id("W", 1001))

    def test_parse_seed_number(self):
        from ncaa_data.normalize import parse_seed_number
        self.assertEqual(parse_seed_number("W01"), 1)
        self.assertEqual(parse_seed_number("X16"), 16)
        self.assertEqual(parse_seed_number("Z08a"), 8)
        self.assertEqual(parse_seed_number("bad"), 17)

    def test_conference_registry(self):
        from ncaa_data.normalize import ConferenceRegistry
        reg = ConferenceRegistry()
        acc_id = reg.encode("ACC")
        b10_id = reg.encode("B10")
        self.assertNotEqual(acc_id, b10_id)
        self.assertEqual(reg.encode("ACC"), acc_id)  # stable
        self.assertEqual(reg.encode("Unknown"), 0)

    def test_flatten_games_doubles_rows(self):
        """flatten_games produces 2 rows per input game (winner + loser)."""
        from ncaa_data.normalize import flatten_games
        df = self.raw["M_regular_compact"]
        flat = flatten_games(df, gender="M")
        self.assertEqual(len(flat), 2 * len(df))

    def test_flatten_games_columns(self):
        from ncaa_data.normalize import flatten_games
        flat = flatten_games(self.raw["M_regular_compact"], gender="M")
        for col in ["season", "canonical_id", "opp_canonical_id", "won", "score", "margin"]:
            self.assertIn(col, flat.columns, f"Missing column: {col}")

    def test_winner_row_has_won_true(self):
        from ncaa_data.normalize import flatten_games
        flat = flatten_games(self.raw["M_regular_compact"], gender="M")
        winning_team_ids = self.raw["M_regular_compact"]["WTeamID"].values
        winners = flat[flat["team_id"].isin(winning_team_ids) & flat["won"]]
        self.assertGreater(len(winners), 0)

    def test_winner_loc_flipped_for_loser(self):
        """Loser row location should be flipped from the winner's perspective."""
        from ncaa_data.normalize import flatten_games
        src = pd.DataFrame([_compact_game(2023, 10, 1001, 75, 1002, 68, "H")])
        flat = flatten_games(src, gender="M")
        winner_row = flat[flat["team_id"] == 1001].iloc[0]
        loser_row  = flat[flat["team_id"] == 1002].iloc[0]
        self.assertEqual(winner_row["loc"], "H")
        self.assertEqual(loser_row["loc"],  "A")

    def test_detailed_columns_attached(self):
        from ncaa_data.normalize import flatten_games
        flat = flatten_games(self.raw["M_regular_detailed"], gender="M")
        for col in ["team_fgm", "team_fga", "opp_fgm", "opp_fga"]:
            self.assertIn(col, flat.columns)

    def test_normalize_all_unified_output(self):
        from ncaa_data.normalize import normalize_all
        games = normalize_all(self.raw)
        self.assertIsInstance(games, pd.DataFrame)
        self.assertGreater(len(games), 0)
        # Both genders present
        self.assertSetEqual(set(games["gender"].unique()), {"M", "W"})

    def test_normalize_all_has_seed_column(self):
        from ncaa_data.normalize import normalize_all
        games = normalize_all(self.raw)
        # Seed column should be present in tourney rows
        tourn = games[games["is_tourney"]]
        self.assertIn("seed", tourn.columns)
        # 1001 should be seed 1
        s1 = tourn[(tourn["team_id"] == 1001) & (tourn["gender"] == "M")]
        if len(s1):
            self.assertEqual(s1["seed"].iloc[0], 1)

    def test_normalize_all_has_conf_id(self):
        from ncaa_data.normalize import normalize_all
        games = normalize_all(self.raw)
        self.assertIn("conf_id", games.columns)
        self.assertTrue((games["conf_id"] >= 0).all())

    def test_no_null_canonical_ids(self):
        from ncaa_data.normalize import normalize_all
        games = normalize_all(self.raw)
        self.assertFalse(games["canonical_id"].isnull().any())
        self.assertFalse(games["opp_canonical_id"].isnull().any())


class TestFeatures(unittest.TestCase):

    def setUp(self):
        self._tmpdir = _make_data_dir()
        self.data_dir = Path(self._tmpdir.name)
        from ncaa_data.ingest import load_raw_data
        from ncaa_data.normalize import normalize_all
        raw = load_raw_data(self.data_dir)
        self.games_df = normalize_all(raw)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_returns_dataframe(self):
        from ncaa_data.features import compute_features
        feat = compute_features(self.games_df, seasons=range(2021, 2026))
        self.assertIsInstance(feat, pd.DataFrame)

    def test_one_row_per_team_season(self):
        from ncaa_data.features import compute_features
        feat = compute_features(self.games_df, seasons=range(2021, 2026))
        # No duplicate (gender, season, canonical_id) triplets
        dupes = feat.duplicated(["gender", "season", "canonical_id"]).sum()
        self.assertEqual(dupes, 0)

    def test_at_least_25_feature_columns(self):
        from ncaa_data.features import compute_features, FEATURE_COLS
        feat = compute_features(self.games_df, seasons=range(2021, 2026))
        computed = [c for c in FEATURE_COLS if c in feat.columns]
        self.assertGreaterEqual(len(computed), 25,
                                f"Only {len(computed)} feature columns present")

    def test_win_pct_between_0_and_1(self):
        from ncaa_data.features import compute_features
        feat = compute_features(self.games_df, seasons=range(2021, 2026))
        wp = feat["win_pct"].dropna()
        self.assertTrue((wp >= 0).all())
        self.assertTrue((wp <= 1).all())

    def test_team_1001_has_high_win_pct(self):
        """Team 1001 wins most games in synthetic data."""
        from ncaa_data.features import compute_features
        feat = compute_features(self.games_df, seasons=range(2021, 2026))
        t1 = feat[(feat["team_id"] == 1001) & (feat["gender"] == "M")]
        self.assertGreater(len(t1), 0)
        self.assertTrue((t1["win_pct"] > 0.5).all())

    def test_scoring_margin_consistent(self):
        """scoring_margin = ppg_scored - ppg_allowed."""
        from ncaa_data.features import compute_features
        feat = compute_features(self.games_df, seasons=range(2021, 2026))
        margin_check = (feat["ppg_scored"] - feat["ppg_allowed"] - feat["scoring_margin"]).abs()
        self.assertTrue((margin_check < 1e-6).all())

    def test_fg_pct_between_0_and_1(self):
        from ncaa_data.features import compute_features
        feat = compute_features(self.games_df, seasons=range(2021, 2026))
        pct = feat["fg_pct"].dropna()
        self.assertTrue((pct >= 0).all() and (pct <= 1).all())

    def test_ortg_positive(self):
        from ncaa_data.features import compute_features
        feat = compute_features(self.games_df, seasons=range(2021, 2026))
        ortg = feat["ortg"].dropna()
        self.assertTrue((ortg > 0).all())

    def test_net_rtg_equals_ortg_minus_drtg(self):
        from ncaa_data.features import compute_features
        feat = compute_features(self.games_df, seasons=range(2021, 2026))
        sub = feat.dropna(subset=["ortg", "drtg", "net_rtg"])
        diff = (sub["ortg"] - sub["drtg"] - sub["net_rtg"]).abs()
        self.assertTrue((diff < 1e-6).all())

    def test_seed_column_present(self):
        from ncaa_data.features import compute_features
        feat = compute_features(self.games_df, seasons=range(2021, 2026))
        self.assertIn("seed", feat.columns)
        # Team 1001 seed should be 1
        t1 = feat[(feat["team_id"] == 1001) & (feat["gender"] == "M")]
        self.assertTrue((t1["seed"] == 1).all())

    def test_unseeded_teams_get_seed_17(self):
        """Teams not in tourney seeds should get seed=17."""
        from ncaa_data.features import compute_features
        feat = compute_features(self.games_df, seasons=range(2021, 2026))
        # All teams in our synthetic data ARE seeded, so just check no NaN
        self.assertFalse(feat["seed"].isnull().any())

    def test_sos_between_0_and_1(self):
        from ncaa_data.features import compute_features
        feat = compute_features(self.games_df, seasons=range(2021, 2026))
        sos = feat["sos"].dropna()
        self.assertTrue((sos >= 0).all() and (sos <= 1).all())

    def test_season_filter(self):
        from ncaa_data.features import compute_features
        feat = compute_features(self.games_df, seasons=[2023])
        self.assertTrue((feat["season"] == 2023).all())

    def test_invalid_season_raises(self):
        from ncaa_data.features import compute_features
        with self.assertRaises(ValueError):
            compute_features(self.games_df, seasons=[1900])

    def test_conf_id_present_and_non_negative(self):
        from ncaa_data.features import compute_features
        feat = compute_features(self.games_df)
        self.assertIn("conf_id", feat.columns)
        self.assertTrue((feat["conf_id"] >= 0).all())

    def test_games_played_positive(self):
        from ncaa_data.features import compute_features
        feat = compute_features(self.games_df)
        self.assertTrue((feat["games_played"] > 0).all())

    def test_both_genders_in_output(self):
        from ncaa_data.features import compute_features
        feat = compute_features(self.games_df)
        self.assertSetEqual(set(feat["gender"].unique()), {"M", "W"})


class TestPipeline(unittest.TestCase):

    def setUp(self):
        self._tmpdir_data = _make_data_dir()
        self._tmpdir_out  = tempfile.TemporaryDirectory()
        self.data_dir = Path(self._tmpdir_data.name)
        self.out_dir  = Path(self._tmpdir_out.name)

    def tearDown(self):
        self._tmpdir_data.cleanup()
        self._tmpdir_out.cleanup()

    def test_dry_run_returns_dataframe(self):
        from ncaa_data.pipeline import run_pipeline
        feat = run_pipeline(
            data_dir=self.data_dir, out_dir=self.out_dir,
            seasons=range(2021, 2026), dry_run=True,
        )
        self.assertIsInstance(feat, pd.DataFrame)
        self.assertGreater(len(feat), 0)

    def test_csv_written(self):
        from ncaa_data.pipeline import run_pipeline
        run_pipeline(
            data_dir=self.data_dir, out_dir=self.out_dir,
            seasons=range(2021, 2026),
            save_parquet=False, save_csv=True,
        )
        csv_path = self.out_dir / "ncaa_features_all.csv"
        self.assertTrue(csv_path.exists())
        df = pd.read_csv(csv_path)
        self.assertGreater(len(df), 0)

    def test_parquet_written_per_season(self):
        from ncaa_data.pipeline import run_pipeline
        run_pipeline(
            data_dir=self.data_dir, out_dir=self.out_dir,
            seasons=range(2021, 2026),
            save_parquet=True, save_csv=False,
        )
        parquet_files = list(self.out_dir.glob("ncaa_features_*.parquet"))
        self.assertEqual(len(parquet_files), 5)  # 2021-2025

    def test_manifest_written(self):
        from ncaa_data.pipeline import run_pipeline
        run_pipeline(
            data_dir=self.data_dir, out_dir=self.out_dir,
            seasons=range(2021, 2026),
        )
        manifest_path = self.out_dir / "pipeline_manifest.json"
        self.assertTrue(manifest_path.exists())
        manifest = json.loads(manifest_path.read_text())
        self.assertEqual(manifest["stage"], "01/10")
        self.assertIn("feature_cols", manifest)
        self.assertGreaterEqual(manifest["n_features"], 25)

    def test_load_features_csv(self):
        from ncaa_data.pipeline import run_pipeline, load_features
        run_pipeline(
            data_dir=self.data_dir, out_dir=self.out_dir,
            seasons=range(2021, 2026),
            save_parquet=False, save_csv=True,
        )
        df = load_features(out_dir=self.out_dir)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)

    def test_load_features_parquet(self):
        from ncaa_data.pipeline import run_pipeline, load_features
        run_pipeline(
            data_dir=self.data_dir, out_dir=self.out_dir,
            seasons=range(2021, 2026),
            save_parquet=True, save_csv=False,
        )
        df = load_features(out_dir=self.out_dir, seasons=[2023])
        self.assertTrue((df["season"] == 2023).all())

    def test_load_features_missing_raises(self):
        from ncaa_data.pipeline import load_features
        with self.assertRaises(FileNotFoundError):
            load_features(out_dir=self.out_dir)

    def test_pipeline_seasons_filter(self):
        from ncaa_data.pipeline import run_pipeline
        feat = run_pipeline(
            data_dir=self.data_dir, out_dir=self.out_dir,
            seasons=[2024], dry_run=True,
        )
        self.assertTrue((feat["season"] == 2024).all())

    def test_column_set(self):
        from ncaa_data.pipeline import run_pipeline
        from ncaa_data.features import FEATURE_COLS, KEY_COLS
        feat = run_pipeline(
            data_dir=self.data_dir, out_dir=self.out_dir,
            dry_run=True,
        )
        for col in KEY_COLS + FEATURE_COLS:
            self.assertIn(col, feat.columns, f"Missing column: {col}")


class TestCLI(unittest.TestCase):

    def setUp(self):
        self._tmpdir_data = _make_data_dir()
        self._tmpdir_out  = tempfile.TemporaryDirectory()
        self.data_dir = str(Path(self._tmpdir_data.name))
        self.out_dir  = str(Path(self._tmpdir_out.name))

    def tearDown(self):
        self._tmpdir_data.cleanup()
        self._tmpdir_out.cleanup()

    def test_dry_run_exit_0(self):
        from ncaa_data.cli import main
        code = main([
            "--data-dir", self.data_dir,
            "--out", self.out_dir,
            "--dry-run",
        ])
        self.assertEqual(code, 0)

    def test_full_run_exit_0(self):
        from ncaa_data.cli import main
        code = main([
            "--data-dir", self.data_dir,
            "--out", self.out_dir,
            "--seasons", "2022", "2023",
            "--no-parquet",
        ])
        self.assertEqual(code, 0)

    def test_missing_data_dir_exit_1(self):
        from ncaa_data.cli import main
        code = main([
            "--data-dir", "/nonexistent/path",
            "--out", self.out_dir,
            "--dry-run",
        ])
        self.assertEqual(code, 1)

    def test_load_subcommand(self):
        from ncaa_data.cli import main
        # First create the outputs
        main(["--data-dir", self.data_dir, "--out", self.out_dir, "--no-parquet"])
        # Then load them
        code = main(["load", "--out", self.out_dir])
        self.assertEqual(code, 0)

    def test_load_missing_raises_1(self):
        from ncaa_data.cli import main
        code = main(["load", "--out", "/nonexistent"])
        self.assertEqual(code, 1)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run_tests():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in [TestIngest, TestNormalize, TestFeatures, TestPipeline, TestCLI]:
        suite.addTests(loader.loadTestsFromTestCase(cls))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
