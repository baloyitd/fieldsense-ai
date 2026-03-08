"""
ncaa_data.cli
=============
Command-line interface for the NCAA data pipeline.

Usage examples::

    # Run for all available seasons
    python -m ncaa_data --data-dir data/ncaa/raw --out data/ncaa/features

    # Run for specific seasons
    python -m ncaa_data --data-dir data/ncaa/raw --out data/ncaa/features \\
        --seasons 2021 2022 2023 2024 2025

    # Dry-run (no file I/O)
    python -m ncaa_data --data-dir data/ncaa/raw --dry-run

    # Via installed entry point
    ncaa-pipeline --data-dir data/ncaa/raw --out data/ncaa/features

    # Reload already-computed features and print summary
    python -m ncaa_data load --out data/ncaa/features
"""

from __future__ import annotations

import argparse
import logging
import sys


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ncaa_data",
        description=(
            "NCAA Basketball Data Pipeline — Stage 01/10\n"
            "Ingests Kaggle competition CSVs, normalizes team IDs, and\n"
            "computes 28 aggregate features per team-season."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    sub = parser.add_subparsers(dest="command", help="Sub-commands")

    # ---- run (default) ---------------------------------------------------
    run_p = sub.add_parser("run", help="Run the full pipeline (default)")
    _add_run_args(run_p)

    # ---- load (inspect existing output) ----------------------------------
    load_p = sub.add_parser("load", help="Load and summarise existing features")
    load_p.add_argument(
        "--out", dest="out_dir", default="data/ncaa/features",
        metavar="DIR", help="Features directory (default: data/ncaa/features)",
    )
    load_p.add_argument(
        "--seasons", nargs="+", type=int, metavar="YEAR",
        help="Filter to specific seasons",
    )

    # ---- Top-level args (run is the implicit default) -------------------
    _add_run_args(parser)

    return parser


def _add_run_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--data-dir", dest="data_dir", default="data/ncaa/raw",
        metavar="DIR",
        help="Directory containing raw Kaggle CSV files (default: data/ncaa/raw)",
    )
    p.add_argument(
        "--out", dest="out_dir", default="data/ncaa/features",
        metavar="DIR",
        help="Output directory for feature files (default: data/ncaa/features)",
    )
    p.add_argument(
        "--seasons", nargs="+", type=int, metavar="YEAR",
        help="Seasons to include (e.g. 2021 2022 2023 2024 2025). "
             "Default: all seasons in the data.",
    )
    p.add_argument(
        "--no-parquet", dest="save_parquet", action="store_false", default=True,
        help="Skip per-season Parquet output",
    )
    p.add_argument(
        "--no-csv", dest="save_csv", action="store_false", default=True,
        help="Skip combined CSV output",
    )
    p.add_argument(
        "--dry-run", dest="dry_run", action="store_true", default=False,
        help="Run pipeline in-memory without writing any files",
    )
    p.add_argument(
        "--log-level", dest="log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging verbosity (default: INFO)",
    )


def main(argv: list[str] | None = None) -> int:
    """Entry point; returns exit code (0 = success)."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level, logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    command = getattr(args, "command", None) or "run"

    if command == "load":
        return _cmd_load(args)

    # Default: run pipeline
    return _cmd_run(args)


def _cmd_run(args: argparse.Namespace) -> int:
    from .pipeline import run_pipeline

    seasons = args.seasons if args.seasons else None

    try:
        feat_df = run_pipeline(
            data_dir=args.data_dir,
            out_dir=args.out_dir,
            seasons=seasons,
            save_parquet=args.save_parquet,
            save_csv=args.save_csv,
            dry_run=args.dry_run,
        )
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print(
            "Hint: download the Kaggle CSV files and place them in the data directory.\n"
            "  kaggle competitions download -c march-machine-learning-mania-2026\n"
            f"  unzip the archive into {args.data_dir}",
            file=sys.stderr,
        )
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"Pipeline failed: {exc}", file=sys.stderr)
        raise

    _print_summary(feat_df)
    return 0


def _cmd_load(args: argparse.Namespace) -> int:
    from .pipeline import load_features

    seasons = args.seasons if args.seasons else None

    try:
        feat_df = load_features(out_dir=args.out_dir, seasons=seasons)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    _print_summary(feat_df)
    return 0


def _print_summary(feat_df) -> None:
    """Print a human-readable summary of the feature DataFrame."""
    print("\n" + "=" * 60)
    print("NCAA Feature Matrix Summary")
    print("=" * 60)
    print(f"  Team-seasons : {len(feat_df):,}")
    print(f"  Seasons      : {sorted(feat_df['season'].unique().tolist())}")
    genders = feat_df["gender"].value_counts().to_dict()
    print(f"  Genders      : {genders}")
    print(f"  Features     : {len(feat_df.columns) - 4} computed columns")
    print(f"  Missing vals : {feat_df.isnull().sum().sum()} total NaN cells")
    print()

    # Quick per-feature NaN counts for transparency
    from .features import FEATURE_COLS
    nan_counts = {c: int(feat_df[c].isnull().sum()) for c in FEATURE_COLS if c in feat_df.columns}
    any_nan = {k: v for k, v in nan_counts.items() if v > 0}
    if any_nan:
        print("  Features with missing values:")
        for col, cnt in any_nan.items():
            pct = 100 * cnt / len(feat_df)
            print(f"    {col:<22} {cnt:4d}  ({pct:.1f}%)")
    else:
        print("  No missing values in feature columns.")
    print("=" * 60 + "\n")
