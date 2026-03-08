"""
test_matchup.py
===============
Tests for ncaa_model.matchup:
  1. matchup_vector produces correct differential features.
  2. Symmetry: x(A, B) = -x(B, A) for all columns not in _FLIP_SIGN_COLS.
  3. build_training_data returns correctly shaped (X, y, meta).
  4. Label convention: y=1 iff lower TeamID team won.
  5. build_prediction_pairs: C(n,2) rows, lower-ID convention, no leakage.
  6. impute_features fills NaN with per-season medians.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ncaa_model.matchup import (
    DEFAULT_FEATURE_COLS,
    _FLIP_SIGN_COLS,
    build_prediction_pairs,
    build_training_data,
    impute_features,
    matchup_vector,
)
from ncaa_model.tests.conftest import (
    M_TEAM_IDS, W_TEAM_IDS, TRAIN_SEASONS, VAL_SEASON,
    make_feat_df, make_tourney_df,
)


# ---------------------------------------------------------------------------
# matchup_vector
# ---------------------------------------------------------------------------

class TestMatchupVector:

    def test_length_matches_feature_cols(self):
        feat_a = pd.Series({c: 1.0 for c in DEFAULT_FEATURE_COLS})
        feat_b = pd.Series({c: 0.0 for c in DEFAULT_FEATURE_COLS})
        x = matchup_vector(feat_a, feat_b, DEFAULT_FEATURE_COLS)
        assert len(x) == len(DEFAULT_FEATURE_COLS)

    def test_sign_convention_per_col_type(self):
        """
        Normal cols (higher = better): A raw > B raw → positive delta.
        Flip-sign cols (lower = better: drtg, seed, tov_rate):
            A raw > B raw means A is *worse* → negative delta.
        """
        feat_a = pd.Series({c: 10.0 for c in DEFAULT_FEATURE_COLS})
        feat_b = pd.Series({c: 5.0  for c in DEFAULT_FEATURE_COLS})
        x = matchup_vector(feat_a, feat_b, DEFAULT_FEATURE_COLS)
        for i, col in enumerate(DEFAULT_FEATURE_COLS):
            if col in _FLIP_SIGN_COLS:
                assert x[i] < 0, (
                    f"Flip-sign col '{col}': A worse (higher raw value) "
                    f"should yield negative delta, got {x[i]}"
                )
            else:
                assert x[i] > 0, (
                    f"Normal col '{col}': A better (higher raw value) "
                    f"should yield positive delta, got {x[i]}"
                )

    def test_zero_when_equal(self):
        feat = pd.Series({c: 3.14 for c in DEFAULT_FEATURE_COLS})
        x = matchup_vector(feat, feat, DEFAULT_FEATURE_COLS)
        assert np.allclose(x, 0.0)

    def test_antisymmetry(self):
        """x(A, B) = -x(B, A) elementwise."""
        feat_a = pd.Series({c: float(i) for i, c in enumerate(DEFAULT_FEATURE_COLS)})
        feat_b = pd.Series({c: 0.0 for c in DEFAULT_FEATURE_COLS})
        x_ab = matchup_vector(feat_a, feat_b, DEFAULT_FEATURE_COLS)
        x_ba = matchup_vector(feat_b, feat_a, DEFAULT_FEATURE_COLS)
        assert np.allclose(x_ab, -x_ba), "Antisymmetry violated"

    def test_custom_feature_subset(self):
        subset = ["net_rtg", "win_pct"]
        feat_a = pd.Series({"net_rtg": 10.0, "win_pct": 0.7})
        feat_b = pd.Series({"net_rtg": 5.0,  "win_pct": 0.4})
        x = matchup_vector(feat_a, feat_b, subset)
        assert len(x) == 2

    def test_missing_feature_treated_as_zero(self):
        """A feature not in the Series is treated as 0."""
        feat_a = pd.Series({"net_rtg": 10.0})   # missing other columns
        feat_b = pd.Series({"net_rtg": 5.0})
        x = matchup_vector(feat_a, feat_b, DEFAULT_FEATURE_COLS)
        # net_rtg should be positive; others should be 0
        net_rtg_idx = DEFAULT_FEATURE_COLS.index("net_rtg")
        assert x[net_rtg_idx] > 0


# ---------------------------------------------------------------------------
# impute_features
# ---------------------------------------------------------------------------

class TestImpute:

    def test_no_nan_after_impute(self, feat_df):
        with_nan = feat_df.copy()
        for col in ["net_rtg", "win_pct", "ortg"]:
            if col in with_nan.columns:
                with_nan.loc[with_nan.index[:5], col] = np.nan
        result = impute_features(with_nan, DEFAULT_FEATURE_COLS)
        for col in ["net_rtg", "win_pct", "ortg"]:
            if col in result.columns:
                assert result[col].notna().all(), f"{col} still has NaN after impute"

    def test_non_nan_values_unchanged(self, feat_df):
        """Values that are not NaN should not be modified."""
        result = impute_features(feat_df, DEFAULT_FEATURE_COLS)
        for col in DEFAULT_FEATURE_COLS:
            if col in feat_df.columns:
                orig_notna = feat_df[col].notna()
                pd.testing.assert_series_equal(
                    feat_df.loc[orig_notna, col].reset_index(drop=True),
                    result.loc[orig_notna, col].reset_index(drop=True),
                )

    def test_all_nan_column_imputed_to_zero(self, feat_df):
        """A column that is entirely NaN imputes to 0 (not median of empty)."""
        df = feat_df.copy()
        df["net_rtg"] = np.nan
        result = impute_features(df, ["net_rtg"])
        assert (result["net_rtg"] == 0.0).all()


# ---------------------------------------------------------------------------
# build_training_data
# ---------------------------------------------------------------------------

class TestBuildTrainingData:

    def test_returns_correct_types(self, feat_df, tourney_df):
        X, y, meta = build_training_data(
            feat_df, tourney_df, TRAIN_SEASONS, DEFAULT_FEATURE_COLS
        )
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert isinstance(meta, pd.DataFrame)

    def test_shapes_consistent(self, feat_df, tourney_df):
        X, y, meta = build_training_data(
            feat_df, tourney_df, TRAIN_SEASONS, DEFAULT_FEATURE_COLS
        )
        assert X.shape[1] == len(DEFAULT_FEATURE_COLS)
        assert len(X) == len(y) == len(meta)

    def test_positive_game_count(self, feat_df, tourney_df):
        X, y, meta = build_training_data(
            feat_df, tourney_df, TRAIN_SEASONS, DEFAULT_FEATURE_COLS
        )
        assert len(X) > 0

    def test_labels_binary(self, feat_df, tourney_df):
        _, y, _ = build_training_data(
            feat_df, tourney_df, TRAIN_SEASONS, DEFAULT_FEATURE_COLS
        )
        assert set(y.tolist()).issubset({0, 1})

    def test_label_convention_lower_id_wins(self, feat_df, tourney_df):
        """
        y=1 iff the lower-ID team won *that specific game*.
        We join meta rows back to tourney_df on (season, gender, pair) to
        verify — avoids false positives when the same pair plays across
        multiple seasons with different outcomes.
        """
        X, y, meta = build_training_data(
            feat_df, tourney_df, TRAIN_SEASONS, DEFAULT_FEATURE_COLS
        )
        # winner rows: team_id = winner, opp_team_id = loser
        tourn_winners = tourney_df[
            tourney_df["season"].isin(TRAIN_SEASONS) & tourney_df["won"]
        ]

        for _, row in meta.iterrows():
            season = row["season"]
            gender = row["gender"]
            tlo = int(row["team_id_low"])
            thi = int(row["team_id_high"])
            label = int(row["y"])

            # Find the winner of this specific game
            game = tourn_winners[
                (tourn_winners["season"] == season) &
                (tourn_winners["gender"] == gender) &
                (
                    ((tourn_winners["team_id"] == tlo) & (tourn_winners["opp_team_id"] == thi)) |
                    ((tourn_winners["team_id"] == thi) & (tourn_winners["opp_team_id"] == tlo))
                )
            ]
            if game.empty:
                continue   # game not in tourney_df; skip

            winner_id = int(game.iloc[0]["team_id"])
            expected_label = 1 if winner_id == tlo else 0
            assert label == expected_label, (
                f"season={season} {gender}: lower_id={tlo} vs {thi}, "
                f"winner={winner_id}, expected label={expected_label}, got {label}"
            )

    def test_meta_columns(self, feat_df, tourney_df):
        _, _, meta = build_training_data(
            feat_df, tourney_df, TRAIN_SEASONS, DEFAULT_FEATURE_COLS
        )
        for col in ("season", "gender", "team_id_low", "team_id_high", "y"):
            assert col in meta.columns, f"Missing meta column: {col}"

    def test_team_id_low_le_high(self, feat_df, tourney_df):
        _, _, meta = build_training_data(
            feat_df, tourney_df, TRAIN_SEASONS, DEFAULT_FEATURE_COLS
        )
        assert (meta["team_id_low"] <= meta["team_id_high"]).all()

    def test_invalid_season_raises(self, feat_df, tourney_df):
        with pytest.raises(ValueError):
            build_training_data(feat_df, tourney_df, [1900], DEFAULT_FEATURE_COLS)

    def test_single_season(self, feat_df, tourney_df):
        X, y, _ = build_training_data(
            feat_df, tourney_df, [TRAIN_SEASONS[-1]], DEFAULT_FEATURE_COLS
        )
        assert len(X) > 0


# ---------------------------------------------------------------------------
# build_prediction_pairs
# ---------------------------------------------------------------------------

class TestBuildPredictionPairs:

    def test_n_pairs_correct(self, feat_df):
        n_teams = 8
        team_ids = M_TEAM_IDS[:n_teams]
        X, meta = build_prediction_pairs(
            feat_df, team_ids, "M", VAL_SEASON, DEFAULT_FEATURE_COLS
        )
        expected = n_teams * (n_teams - 1) // 2
        assert len(X) == expected == len(meta), (
            f"Expected {expected} pairs, got {len(X)}"
        )

    def test_lower_id_always_first(self, feat_df):
        X, meta = build_prediction_pairs(
            feat_df, M_TEAM_IDS[:6], "M", VAL_SEASON, DEFAULT_FEATURE_COLS
        )
        assert (meta["team_id_low"] < meta["team_id_high"]).all()

    def test_no_self_pairs(self, feat_df):
        _, meta = build_prediction_pairs(
            feat_df, M_TEAM_IDS[:6], "M", VAL_SEASON, DEFAULT_FEATURE_COLS
        )
        assert (meta["team_id_low"] != meta["team_id_high"]).all()

    def test_shape_consistent(self, feat_df):
        X, meta = build_prediction_pairs(
            feat_df, M_TEAM_IDS[:6], "M", VAL_SEASON, DEFAULT_FEATURE_COLS
        )
        assert X.shape[1] == len(DEFAULT_FEATURE_COLS)
        assert len(X) == len(meta)

    def test_women_pairs(self, feat_df):
        X, meta = build_prediction_pairs(
            feat_df, W_TEAM_IDS[:6], "W", VAL_SEASON, DEFAULT_FEATURE_COLS
        )
        assert len(X) > 0
        assert (meta["gender"] == "W").all()

    def test_season_in_meta(self, feat_df):
        _, meta = build_prediction_pairs(
            feat_df, M_TEAM_IDS[:4], "M", VAL_SEASON, DEFAULT_FEATURE_COLS
        )
        assert (meta["season"] == VAL_SEASON).all()

    def test_antisymmetric_pairs(self, feat_df):
        """
        For any pair (A, B) in the prediction set:
        x(A, B) == -x(B, A).
        """
        team_ids = M_TEAM_IDS[:4]
        X, meta = build_prediction_pairs(
            feat_df, team_ids, "M", VAL_SEASON, DEFAULT_FEATURE_COLS
        )
        # Reverse the feat_df lookup and recompute
        feat_imp = impute_features(feat_df, DEFAULT_FEATURE_COLS)
        fidx = (
            feat_imp[(feat_imp["gender"] == "M") & (feat_imp["season"] == VAL_SEASON)]
            .set_index("team_id")[DEFAULT_FEATURE_COLS]
        )
        for i, row in meta.iterrows():
            x_ab = X[i]
            fa = fidx.loc[row["team_id_low"]]
            fb = fidx.loc[row["team_id_high"]]
            x_ba = matchup_vector(fb, fa, DEFAULT_FEATURE_COLS)
            assert np.allclose(x_ab, -x_ba, atol=1e-9), (
                f"Antisymmetry violated for pair {row['team_id_low']} vs {row['team_id_high']}"
            )
