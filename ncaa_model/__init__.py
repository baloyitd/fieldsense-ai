"""
ncaa_model - NCAA Tournament Prediction Models for FieldSense AI v3.0
======================================================================
Stage 02 / 10 of the 2026 NCAA March Machine Learning Mania pipeline.

Builds on ncaa_data (Stage 01) which supplies team-season feature matrices.

Modules
-------
matchup   Build matchup feature vectors (differential representation).
baseline  Logistic regression baseline model (target Brier score < 0.20).
evaluate  Model-agnostic evaluation harness used by all subsequent stages.
submit    Kaggle submission CSV generation.

Quick start::

    from ncaa_data.pipeline import run_pipeline, load_features
    from ncaa_model.baseline import LogisticBaseline
    from ncaa_model.evaluate import cross_validate, generate_submission

    feat_df = load_features("data/ncaa/features")

    model = LogisticBaseline()
    model.fit(feat_df, tourney_df, train_seasons=range(2021, 2025))

    result = evaluate_season(model, feat_df, tourney_df, val_season=2025)
    print(f"Brier score: {result.brier_score:.4f}")   # target < 0.20

    sub_df = generate_submission(model, feat_df, team_ids_m, team_ids_w, season=2026)
    sub_df.to_csv("submission.csv", index=False)

Version: 1.0.0  (Stage 02 / 10)
"""

__version__ = "1.0.0"
__all__ = ["matchup", "baseline", "evaluate", "submit"]
