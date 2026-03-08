"""
ncaa_data - NCAA Basketball Data Pipeline for FieldSense AI v3.0
================================================================
Adapts FieldSense AI for the 2026 NCAA March Machine Learning Mania
Kaggle competition.

Modules:
    ingest    - Load raw Kaggle CSV files (M/W compact + detailed results,
                seeds, teams, conferences)
    normalize - Canonical team ID schema, conference encoding, gender tagging
    features  - Compute 28 basketball-specific aggregate features per team-season
    pipeline  - End-to-end orchestration: ingest → normalize → features → output
    cli       - Command-line entry point

Quick start::

    from ncaa_data.pipeline import run_pipeline
    features_df = run_pipeline(data_dir="data/ncaa/raw", seasons=range(2021, 2026))

CLI::

    python -m ncaa_data --data-dir data/ncaa/raw --seasons 2021-2025 --out data/ncaa/features

Version: 1.0.0  (Stage 01 / 10)
"""

__version__ = "1.0.0"
__all__ = ["ingest", "normalize", "features", "pipeline", "cli"]
